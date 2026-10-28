"""Generate synthetic protocols for many seed trials across several vLLM servers.

Seed registry records are fetched once from the ClinicalTrials.gov API in
batches and saved with their retrieval time (seeds.jsonl). Each server gets its
own pool of workers sharing one queue; a server that does not answer is checked
again every minute, so one that is still launching joins when it is ready.
Every trial gets a directory with the same files as generate_from_registry.py.
A trial whose response_meta.json records finish_reason "stop" is skipped, so
the batch resumes after interruption; failed attempts are retried up to
--attempts times and then recorded in failures.jsonl.
"""
import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time

import httpx
from openai import APIConnectionError, AsyncOpenAI
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_from_registry import (API_URL, FIELDS, MISSING, RECORD_FIELDS, choose_layout, now,
                                    record_from_protocol, render)

IDS_PER_REQUEST = 200
EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": True}}


def fetch_seeds(nct_ids, path):
    if path.exists():
        return {r["nct_id"]: r for r in map(json.loads, path.read_text().splitlines())}
    seeds = {}
    with httpx.Client(timeout=120) as client:
        for i in range(0, len(nct_ids), IDS_PER_REQUEST):
            params = {"filter.ids": ",".join(nct_ids[i:i + IDS_PER_REQUEST]), "fields": FIELDS, "pageSize": 1000}
            for attempt in range(6):
                response = client.get(API_URL.format("").rstrip("/"), params=params)
                if response.status_code not in (429, 500, 502, 503, 504):
                    break
                time.sleep(2 ** attempt)
            response.raise_for_status()
            for study in response.json()["studies"]:
                record = record_from_protocol(study["protocolSection"])
                seeds[record["nct_id"]] = record
    path.write_text("".join(json.dumps(seeds[n]) + "\n" for n in nct_ids if n in seeds))
    return seeds


def api_base(url):
    url = url.strip().rstrip("/")
    return url if url.endswith("/v1") else url + "/v1"


class Server:
    def __init__(self, spec):
        url, _, concurrency = spec.partition("=")
        self.base_url = api_base(url)
        self.concurrency = int(concurrency or 8)
        self.model, self.checked = None, 0
        self.client = AsyncOpenAI(base_url=self.base_url, api_key="EMPTY", timeout=3600, max_retries=0)

    async def ready(self):
        if self.model:
            return True
        if time.time() - self.checked < 60:
            return False
        self.checked = time.time()
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(self.base_url + "/models")
                response.raise_for_status()
                self.model = response.json()["data"][0]["id"]
            print(f"[{now()}] server ready: {self.base_url} {self.model}", flush=True)
        except (httpx.HTTPError, KeyError, IndexError, ValueError):
            pass
        return bool(self.model)


class Batch:
    def __init__(self, args, seeds, pending):
        self.args, self.seeds = args, seeds
        self.queue = asyncio.Queue()
        for nct_id in pending:
            self.queue.put_nowait((nct_id, 0))
        self.in_flight, self.done, self.failed = 0, Counter(), 0
        self.started = time.time()
        self.template = args.prompt.read_text()
        self.disabled, self.disabled_checked = set(), 0

    def is_disabled(self, server):
        """Servers listed in disabled_servers.txt get no new requests; in-flight ones finish."""
        if time.time() - self.disabled_checked > 10:
            self.disabled_checked = time.time()
            path = self.args.batch_dir / "disabled_servers.txt"
            disabled = {api_base(line) for line in path.read_text().splitlines() if line.strip()} if path.exists() else set()
            for url in disabled - self.disabled:
                print(f"[{now()}] server disabled: {url}", flush=True)
            self.disabled = disabled
        return server.base_url in self.disabled

    def finished(self):
        return self.queue.empty() and self.in_flight == 0

    async def generate(self, server, nct_id, attempt):
        record = self.seeds[nct_id]
        fields = {key: record[key].strip() or MISSING for key in RECORD_FIELDS}
        layout_index = None
        if "{{eligibility_layout}}" in self.template:
            layout_index, fields["eligibility_layout"] = choose_layout(self.args.prompt, nct_id, None)
        messages = render(self.template, fields)
        settings = dict(temperature=self.args.temperature, top_p=self.args.top_p, max_tokens=self.args.max_tokens)
        started = time.time()
        response = await server.client.chat.completions.create(model=server.model, messages=messages,
                                                               extra_body=EXTRA_BODY, **settings)
        choice = response.choices[0]
        message = choice.message
        content = message.content or ""
        reasoning = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None) or ""
        if choice.finish_reason != "stop" or not content.strip():
            raise RuntimeError(f"finish_reason={choice.finish_reason}, {len(content)} content chars")
        run_dir = self.args.batch_dir / "protocols" / nct_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "registry_record.json").write_text(json.dumps(record, indent=2))
        (run_dir / "request.json").write_text(json.dumps(
            dict(base_url=server.base_url, model=server.model, prompt=str(self.args.prompt), layout_index=layout_index,
                 messages=messages, **settings), indent=2))
        (run_dir / "protocol.md").write_text(content)
        (run_dir / "reasoning.txt").write_text(reasoning)
        generation = dict(finish_reason=choice.finish_reason, usage=response.usage.model_dump() if response.usage else None,
                          elapsed_seconds=round(time.time() - started, 1), completed_at=now(), attempt=attempt,
                          extra_body=EXTRA_BODY, **settings)
        meta = dict(model=server.model, base_url=server.base_url, layout_index=layout_index, generation=generation)
        (run_dir / "response_meta.json").write_text(json.dumps(meta, indent=2))

    async def worker(self, server):
        while not self.finished():
            if self.is_disabled(server) or not await server.ready() or self.queue.empty():
                await asyncio.sleep(5)
                continue
            nct_id, attempt = self.queue.get_nowait()
            self.in_flight += 1
            try:
                await self.generate(server, nct_id, attempt)
                self.done[server.base_url] += 1
            except Exception as exc:
                if isinstance(exc, (APIConnectionError, httpx.ConnectError)):
                    server.model, server.checked = None, time.time()
                error = dict(nct_id=nct_id, attempt=attempt, server=server.base_url, at=now(),
                             error=f"{type(exc).__name__}: {str(exc)[:500]}")
                with open(self.args.batch_dir / "errors.jsonl", "a") as handle:
                    handle.write(json.dumps(error) + "\n")
                if attempt + 1 < self.args.attempts:
                    self.queue.put_nowait((nct_id, attempt + 1))
                else:
                    self.failed += 1
                    with open(self.args.batch_dir / "failures.jsonl", "a") as handle:
                        handle.write(json.dumps(error) + "\n")
            finally:
                self.in_flight -= 1

    async def report(self, total):
        while not self.finished():
            await asyncio.sleep(60)
            done = sum(self.done.values())
            rate = done / max((time.time() - self.started) / 60, 1e-9)
            remaining = self.queue.qsize() + self.in_flight
            progress = dict(at=now(), done_this_run=done, failed=self.failed, queued=self.queue.qsize(),
                            in_flight=self.in_flight, per_minute=round(rate, 2),
                            eta_minutes=round(remaining / rate) if rate else None, total_pending_at_start=total,
                            per_server=dict(self.done))
            (self.args.batch_dir / "progress.json").write_text(json.dumps(progress, indent=2))
            print(json.dumps(progress), flush=True)


def completed(batch_dir, nct_id):
    meta = batch_dir / "protocols" / nct_id / "response_meta.json"
    return meta.exists() and json.loads(meta.read_text())["generation"]["finish_reason"] == "stop"


async def run(args):
    nct_ids = list(dict.fromkeys(pd.read_csv(args.nct_ids_from, usecols=["nct_id"]).nct_id.dropna()))
    args.batch_dir.mkdir(parents=True, exist_ok=True)
    seeds = fetch_seeds(nct_ids, args.batch_dir / "seeds.jsonl")
    missing = [n for n in nct_ids if n not in seeds]
    pending = [n for n in nct_ids if n in seeds and not completed(args.batch_dir, n)]
    servers = [Server(spec) for spec in args.server]
    config_path = args.batch_dir / "batch_config.json"
    if not config_path.exists():
        code = args.batch_dir / "code"
        code.mkdir(exist_ok=True)
        for source in (Path(__file__), Path(__file__).with_name("generate_from_registry.py"), args.prompt,
                       args.prompt.with_name(args.prompt.stem + ".layouts.json")):
            if source.exists():
                shutil.copy2(source, code / source.name)
        config_path.write_text(json.dumps(dict(
            created_at=now(), nct_ids_from=str(args.nct_ids_from), seed_trials=len(nct_ids), prompt=str(args.prompt),
            servers=[dict(base_url=s.base_url, concurrency=s.concurrency) for s in servers],
            temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens, attempts=args.attempts), indent=2))
    print(f"[{now()}] {len(nct_ids)} seed trials, {len(missing)} missing from the API, "
          f"{len(nct_ids) - len(missing) - len(pending)} already complete, {len(pending)} pending", flush=True)
    batch = Batch(args, seeds, pending)
    workers = [batch.worker(server) for server in servers for _ in range(server.concurrency)]
    await asyncio.gather(batch.report(len(pending)), *workers)
    print(f"[{now()}] finished: {sum(batch.done.values())} generated, {batch.failed} failed "
          f"(see failures.jsonl), per server {dict(batch.done)}", flush=True)
    return 0 if batch.failed == 0 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--nct-ids-from", type=Path, required=True, help="CSV with an nct_id column")
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--server", action="append", required=True, help="BASE_URL[=CONCURRENCY]; repeatable")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=120_000)
    parser.add_argument("--attempts", type=int, default=3)
    sys.exit(asyncio.run(run(parser.parse_args())))
