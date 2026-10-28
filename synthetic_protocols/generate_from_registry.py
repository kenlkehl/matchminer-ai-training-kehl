"""Generate a full synthetic protocol seeded by a trial's ClinicalTrials.gov registry text.

The prompt template (prompts/*.txt, "[system]" then "[user]" blocks) receives the
official title, brief summary, detailed description and eligibility criteria.
A template that asks for {{eligibility_layout}} gets one entry of its sibling
<stem>.layouts.json, chosen by a stable hash of the NCT ID (weighted when
entries carry weights) unless --layout-index is given. An optional editor template then reviews the draft and returns JSON
edits; each is applied only where its target occurs exactly once (allowing for
whitespace differences), so the editor cannot silently rewrite the document.
Each run gets its own directory holding the templates, registry inputs with
their retrieval date, rendered requests, draft, edits, protocol, reasoning and
usage, so prompt versions stay comparable as they change.
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
HEADING = re.compile(r"^(#{1,6})\s")


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


def parse_edits(text):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    return json.loads(text[text.find("{"):text.rfind("}") + 1])["edits"]


def delete_section(text, heading):
    lines = text.split("\n")
    starts = [i for i, line in enumerate(lines) if line.strip() == heading.strip() and HEADING.match(line)]
    if len(starts) != 1:
        return text, "not_found" if not starts else "ambiguous"
    start, level = starts[0], len(HEADING.match(lines[starts[0]]).group(1))
    end = next((i for i in range(start + 1, len(lines))
                if HEADING.match(lines[i]) and len(HEADING.match(lines[i]).group(1)) <= level), len(lines))
    return "\n".join(lines[:start] + lines[end:]), "applied"


def replace_once(text, find, replace):
    if text.count(find) == 1:
        return text.replace(find, replace), "applied"
    matches = list(re.finditer(r"\s+".join(map(re.escape, find.split())), text)) if find.strip() else []
    if len(matches) == 1:
        return text[:matches[0].start()] + replace + text[matches[0].end():], "applied_whitespace"
    return text, "not_found" if not matches else "ambiguous"


def apply_edits(text, edits):
    results = []
    for edit in edits:
        if edit.get("delete_heading"):
            text, status = delete_section(text, edit["delete_heading"])
        else:
            text, status = replace_once(text, edit.get("find", ""), edit.get("replace", ""))
        results.append({**edit, "status": status})
    return text, results


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

    if args.editor_prompt and ok:
        (run_dir / "protocol_draft.md").write_text(protocol)
        shutil.copy2(args.editor_prompt, run_dir / "editor_template.txt")
        editor_messages = render(args.editor_prompt.read_text(), {"protocol": protocol})
        editor_settings = dict(temperature=args.editor_temperature, top_p=args.top_p, max_tokens=args.editor_max_tokens)
        (run_dir / "editor_request.json").write_text(json.dumps(
            dict(prompt=str(args.editor_prompt), messages=editor_messages, **editor_settings), indent=2))
        response, editor_reasoning, editing = complete(client, args, editor_messages, editor_settings, "edit")
        (run_dir / "editor_response.txt").write_text(response)
        (run_dir / "editor_reasoning.txt").write_text(editor_reasoning)
        try:
            protocol, results = apply_edits(protocol, parse_edits(response))
        except (ValueError, KeyError, TypeError) as exc:
            results, editing["parse_error"] = [], f"{type(exc).__name__}: {exc}"
            ok = False
        (run_dir / "edits.json").write_text(json.dumps(results, indent=2))
        editing["edits"] = {status: sum(r["status"] == status for r in results) for status in {r["status"] for r in results}}
        meta["editing"] = editing
        ok = ok and editing["finish_reason"] == "stop"

    (run_dir / "protocol.md").write_text(protocol)
    (run_dir / "response_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--nct-id", required=True)
    parser.add_argument("--prompt", type=Path, required=True, help="Versioned template under prompts/")
    parser.add_argument("--editor-prompt", type=Path, help="Optional versioned editor template under prompts/")
    parser.add_argument("--layout-index", type=int, help="Override the hashed eligibility-layout choice")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-url", default="http://sn4622130540:8000/v1")
    parser.add_argument("--model", default="nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=120_000)
    parser.add_argument("--editor-temperature", type=float, default=0.6)
    parser.add_argument("--editor-max-tokens", type=int, default=32_768)
    sys.exit(main(parser.parse_args()))
