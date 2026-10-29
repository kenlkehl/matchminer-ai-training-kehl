"""Generate a full synthetic protocol seeded by a trial's ClinicalTrials.gov registry text.

The prompt template (prompts/*.txt, "[system]" then "[user]" blocks) receives the
official title, brief summary, detailed description and eligibility criteria.
A template that asks for {{eligibility_layout}} gets one entry of its sibling
<stem>.layouts.json, chosen by a stable hash of the NCT ID (weighted when
entries carry weights) unless --layout-index is given. Each run gets its own
directory holding the template, registry inputs with their retrieval date, the
rendered request, protocol, reasoning and usage.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import re
import shutil
import sys
import time

import httpx
from openai import OpenAI

HERE = Path(__file__).resolve().parent
API_URL = "https://clinicaltrials.gov/api/v2/studies/{}"
FIELDS = "NCTId,BriefTitle,OfficialTitle,BriefSummary,DetailedDescription,EligibilityCriteria,LastUpdatePostDate"
RECORD_FIELDS = ("title", "brief_summary", "detailed_description", "eligibility_criteria")
MISSING = "Not provided in the registry record."
DEFAULT_OUTPUT = HERE.parents[1] / "data" / "no_phi" / "synthetic_protocols_from_registry"
PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def registry_record(nct_id):
    for attempt in range(6):
        response = httpx.get(API_URL.format(nct_id), params={"fields": FIELDS}, timeout=60)
        if response.status_code not in (429, 500, 502, 503, 504):
            break
        time.sleep(2 ** attempt)
    response.raise_for_status()
    return record_from_protocol(response.json()["protocolSection"])


def record_from_protocol(protocol):
    ident, description = protocol["identificationModule"], protocol.get("descriptionModule", {})
    return dict(
        nct_id=ident["nctId"],
        title=ident.get("officialTitle") or ident.get("briefTitle", ""),
        brief_summary=description.get("briefSummary", ""),
        detailed_description=description.get("detailedDescription", ""),
        eligibility_criteria=protocol.get("eligibilityModule", {}).get("eligibilityCriteria", ""),
        last_update_post_date=protocol.get("statusModule", {}).get("lastUpdatePostDateStruct", {}).get("date", ""),
        retrieved_at=now(),
    )


def render(template, fields):
    system, user = template.split("[user]\n", 1)
    system = system.removeprefix("[system]\n").strip()
    missing = set(PLACEHOLDER.findall(user)) - fields.keys()
    if missing:
        raise ValueError(f"unfilled placeholders in prompt template: {sorted(missing)}")
    user = PLACEHOLDER.sub(lambda m: fields[m.group(1)], user)
    return [{"role": "system", "content": system}, {"role": "user", "content": user.strip()}]


def choose_layout(prompt, nct_id, index):
    """Entries are strings or {"weight": w, "text": ...}; weights default to 1."""
    layouts = json.loads(prompt.with_name(prompt.stem + ".layouts.json").read_text())
    if index is None:
        weights = [entry.get("weight", 1) if isinstance(entry, dict) else 1 for entry in layouts]
        point = int(hashlib.sha256(nct_id.encode()).hexdigest(), 16) % sum(weights)
        index = next(i for i, bound in enumerate(itertools.accumulate(weights)) if point < bound)
    entry = layouts[index]
    return index, entry["text"] if isinstance(entry, dict) else entry


def complete(client, args, messages, settings, label):
    extra_body = {"chat_template_kwargs": {"enable_thinking": True}}
    started, last_report = time.time(), 0
    content, reasoning, finish, usage = [], [], None, None
    stream = client.chat.completions.create(model=args.model, messages=messages, stream=True,
                                            stream_options={"include_usage": True}, extra_body=extra_body, **settings)
    for chunk in stream:
        if chunk.usage:
            usage = chunk.usage.model_dump()
        for choice in chunk.choices:
            reasoning.append(getattr(choice.delta, "reasoning_content", None) or getattr(choice.delta, "reasoning", None) or "")
            content.append(choice.delta.content or "")
            finish = choice.finish_reason or finish
        if time.time() - last_report > 60:
            last_report = time.time()
            print(f"[{label} {time.time() - started:,.0f}s] reasoning {len(''.join(reasoning)):,} chars, "
                  f"output {len(''.join(content)):,} chars", flush=True)
    meta = dict(finish_reason=finish, usage=usage, elapsed_seconds=round(time.time() - started, 1), completed_at=now(),
                extra_body=extra_body, **settings)
    return "".join(content), "".join(reasoning), meta


def main(args):
    record = registry_record(args.nct_id)
    fields = {key: record[key].strip() or MISSING for key in RECORD_FIELDS}
    template = args.prompt.read_text()
    layout_index = None
    if "{{eligibility_layout}}" in template:
        layout_index, fields["eligibility_layout"] = choose_layout(args.prompt, record["nct_id"], args.layout_index)
    messages = render(template, fields)
    run_dir = args.output_root / f"{record['nct_id']}__{args.prompt.stem}__{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    run_dir.mkdir(parents=True)
    shutil.copy2(args.prompt, run_dir / "prompt_template.txt")
    (run_dir / "registry_record.json").write_text(json.dumps(record, indent=2))
    settings = dict(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens)
    (run_dir / "request.json").write_text(json.dumps(
        dict(base_url=args.base_url, model=args.model, prompt=str(args.prompt), layout_index=layout_index,
             messages=messages, **settings), indent=2))
    print(f"[*] {run_dir} (layout {layout_index})", flush=True)

    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=7200)
    protocol, reasoning, generation = complete(client, args, messages, settings, "generate")
    (run_dir / "reasoning.txt").write_text(reasoning)
    meta = dict(model=args.model, base_url=args.base_url, layout_index=layout_index, generation=generation)
    ok = generation["finish_reason"] == "stop"

    (run_dir / "protocol.md").write_text(protocol)
    (run_dir / "response_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--nct-id", required=True)
    parser.add_argument("--prompt", type=Path, required=True, help="Versioned template under prompts/")
    parser.add_argument("--layout-index", type=int, help="Override the hashed eligibility-layout choice")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-url", default="http://sn4622130540:8000/v1")
    parser.add_argument("--model", default="nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=120_000)
    sys.exit(main(parser.parse_args()))
