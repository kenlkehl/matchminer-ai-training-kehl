#!/usr/bin/env python3
"""Download public ClinicalTrials.gov protocol PDFs for oncology trials.

The ClinicalTrials.gov API v2 lists studies with an uploaded protocol; each
document flagged hasProtocol (Prot, Prot_SAP, Prot_ICF, Prot_SAP_ICF) is fetched
from the public large-docs CDN. Standalone SAPs and consent forms are skipped.
Every protocol version is kept, because amendments change eligibility.

Scope defaults to the term query behind the oncology export parsed by
0a_parse_ctgov_json.py; --nct-ids-from restricts it to a table's nct_id column
(e.g. ctgov_trials.csv or sample_trial_cohort_lineitems.csv). Files are checked
against the registry size and the PDF header, written atomically and skipped on
reruns when the size already matches. manifest.jsonl records registry
identifiers, document/upload dates, retrieval time and each file's outcome.

Sponsors can hold copyright in these documents; check terms before
redistributing them or derived text.
"""
import argparse
import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import urllib.parse

import httpx
import pandas as pd

API_URL = "https://clinicaltrials.gov/api/v2/studies"
CDN_URL = "https://cdn.clinicaltrials.gov/large-docs"
ONCOLOGY_TERMS = ("cancer", "tumor", "oncology", "lymphoma", "leukemia", "carcinoma", "sarcoma",
                  "melanoma", "myelodysplastic", "myeloproliferative", "myeloma", "malignancy")
FIELDS = "NCTId,BriefTitle,Condition,OverallStatus,LastUpdatePostDate,LargeDocumentModule"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "data" / "no_phi" / "ctgov_protocol_pdfs"
NCT_ID = re.compile(r"NCT\d{8}")
RETRY_STATUS = {429, 500, 502, 503, 504}
IDS_PER_REQUEST = 200


@dataclass
class ProtocolDoc:
    nct_id: str
    brief_title: str
    conditions: list
    overall_status: str
    last_update_post_date: str
    type_abbrev: str
    label: str
    doc_date: str
    upload_date: str
    filename: str
    size_bytes: int
    url: str
    local_path: str
    status: str = "pending"
    error: str = ""
    retrieved_at: str = ""


def cdn_url(nct_id, filename):
    """The CDN shards documents by the NCT ID's last two digits."""
    return f"{CDN_URL}/{nct_id[-2:]}/{nct_id}/{urllib.parse.quote(filename)}"


async def get_json(client, params, attempts=6):
    for attempt in range(attempts):
        try:
            response = await client.get(API_URL, params=params, timeout=60)
            if response.status_code not in RETRY_STATUS:
                response.raise_for_status()
                return response.json()
            retry_after = response.headers.get("Retry-After", "")
            delay = float(retry_after) if retry_after.isdigit() else 2 ** attempt
        except httpx.TransportError:
            if attempt == attempts - 1:
                raise
            delay = 2 ** attempt
        await asyncio.sleep(delay)
    response.raise_for_status()


def protocol_docs(study):
    protocol = study.get("protocolSection", {})
    ident = protocol.get("identificationModule", {})
    status = protocol.get("statusModule", {})
    nct_id = ident.get("nctId", "")
    if not NCT_ID.fullmatch(nct_id):
        return
    for doc in study.get("documentSection", {}).get("largeDocumentModule", {}).get("largeDocs", []):
        filename = Path(doc.get("filename") or "").name
        if not doc.get("hasProtocol") or not filename:
            continue
        yield ProtocolDoc(
            nct_id=nct_id,
            brief_title=ident.get("briefTitle", ""),
            conditions=protocol.get("conditionsModule", {}).get("conditions", []),
            overall_status=status.get("overallStatus", ""),
            last_update_post_date=(status.get("lastUpdatePostDateStruct") or {}).get("date", ""),
            type_abbrev=doc.get("typeAbbrev", ""),
            label=doc.get("label", ""),
            doc_date=doc.get("date", ""),
            upload_date=doc.get("uploadDate", ""),
            filename=filename,
            size_bytes=int(doc.get("size") or 0),
            url=cdn_url(nct_id, filename),
            local_path=f"{nct_id}/{filename}",
        )


async def discover(client, query, nct_ids, max_studies):
    """Page through studies that have a protocol document (aggFilters docs:prot)."""
    base = {"aggFilters": "docs:prot", "fields": FIELDS, "pageSize": 1000}
    requests = ([{**base, "filter.ids": ",".join(nct_ids[i:i + IDS_PER_REQUEST])}
                 for i in range(0, len(nct_ids), IDS_PER_REQUEST)]
                if nct_ids is not None else [{**base, "query.term": query, "countTotal": "true"}])
    docs, studies = [], 0
    for params in requests:
        token = None
        while True:
            payload = await get_json(client, {**params, **({"pageToken": token} if token else {})})
            if "totalCount" in payload and not token:
                print(f"[*] {payload['totalCount']} matching studies have an uploaded protocol", flush=True)
            for study in payload.get("studies", []):
                docs.extend(protocol_docs(study))
                studies += 1
                if max_studies and studies >= max_studies:
                    return docs, studies
            token = payload.get("nextPageToken")
            if not token:
                break
    return docs, studies


async def download(client, doc, output_dir, semaphore, attempts=4):
    dest = output_dir / doc.local_path
    if doc.size_bytes and dest.exists() and dest.stat().st_size == doc.size_bytes:
        doc.status = "existing"
        return doc
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    async with semaphore:
        for attempt in range(attempts):
            try:
                async with client.stream("GET", doc.url, timeout=120) as response:
                    response.raise_for_status()
                    with open(partial, "wb") as handle:
                        header = b""
                        async for chunk in response.aiter_bytes(1 << 16):
                            if len(header) < 5:
                                header += chunk[:5 - len(header)]
                            handle.write(chunk)
                written = partial.stat().st_size
                if not header.startswith(b"%PDF-"):
                    raise ValueError(f"not a PDF (starts {header!r})")
                if doc.size_bytes and written != doc.size_bytes:
                    raise ValueError(f"size {written} != registry size {doc.size_bytes}")
                partial.replace(dest)
                doc.status, doc.error = "downloaded", ""
                doc.retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
                return doc
            except (httpx.HTTPError, OSError, ValueError) as exc:
                partial.unlink(missing_ok=True)
                doc.error = f"{type(exc).__name__}: {exc}"
                permanent = isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code not in RETRY_STATUS
                if permanent or attempt == attempts - 1:
                    doc.status = "failed"
                    return doc
                await asyncio.sleep(2 ** attempt)


def write_manifest(path, docs):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            for doc in docs:
                handle.write(json.dumps(asdict(doc)) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


async def main(args):
    nct_ids = None
    if args.nct_ids_from:
        nct_ids = sorted(set(pd.read_csv(args.nct_ids_from, usecols=["nct_id"]).nct_id.dropna()))
        nct_ids = [n for n in nct_ids if NCT_ID.fullmatch(n)]
        print(f"[*] Restricting to {len(nct_ids)} NCT IDs from {args.nct_ids_from}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "manifest.jsonl"
    headers = {"User-Agent": "MatchMiner-AI-training/0.1 (academic research; protocol eligibility text)"}
    limits = httpx.Limits(max_connections=args.concurrency, max_keepalive_connections=args.concurrency)
    async with httpx.AsyncClient(headers=headers, limits=limits, follow_redirects=True) as client:
        docs, studies = await discover(client, " OR ".join(ONCOLOGY_TERMS), nct_ids, args.max_studies)
        total_mb = sum(d.size_bytes for d in docs) / 2 ** 20
        print(f"[*] {len(docs)} protocol PDFs across {studies} studies (~{total_mb:,.0f} MB)", flush=True)
        if args.discover_only:
            write_manifest(manifest, docs)
            print(f"[+] Manifest only: {manifest}")
            return
        semaphore = asyncio.Semaphore(args.concurrency)
        tasks = [asyncio.create_task(download(client, d, args.output_dir, semaphore)) for d in docs]
        for done, task in enumerate(asyncio.as_completed(tasks), 1):
            await task
            if done % 100 == 0 or done == len(tasks):
                counts = pd.Series([d.status for d in docs]).value_counts().to_dict()
                print(f"[*] {done}/{len(tasks)} finished: {counts}", flush=True)
    write_manifest(manifest, docs)
    failed = [d for d in docs if d.status == "failed"]
    for doc in failed[:20]:
        print(f"[!] {doc.local_path}: {doc.error}")
    print(f"[+] {len(docs) - len(failed)}/{len(docs)} PDFs present in {args.output_dir}; manifest {manifest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--nct-ids-from", type=Path, help="CSV with an nct_id column to restrict discovery")
    parser.add_argument("--max-studies", type=int, help="Stop after this many studies (for trial runs)")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--discover-only", action="store_true", help="Write the manifest without downloading")
    asyncio.run(main(parser.parse_args()))
