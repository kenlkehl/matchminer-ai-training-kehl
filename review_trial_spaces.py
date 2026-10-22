"""Opt-in, audited refinement of completed trial spaces against their source.

Input JSON is a list of {case_id, nct_id, trial_text, space_text} objects, with
one completed space per object. Results are separate from production outputs.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import re
import time

from trial_space_refinement import (
    VERSION,
    build_refinement_messages,
    format_space,
    load_refinement_prompts,
    parse_refinement_response,
    parse_space_text,
    refinement_retry_messages,
)


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


async def run(args):
    from openai import AsyncOpenAI
    from remote_vllm_pool import CompletionSampling, make_completion_work_fn
    from vllm_reasoning_utils import (
        load_tokenizer, resolve_parser_name, resolve_sampling_defaults,
        resolve_chat_template_kwargs,
    )

    if min(args.concurrency, args.max_attempts, args.max_tokens, args.max_model_len) < 1:
        raise ValueError("Concurrency, attempts, and token budgets must be positive")
    cases = json.loads(args.input.read_text())
    if not isinstance(cases, list) or not cases:
        raise ValueError("Input must be a nonempty list of cases")
    prompts = load_refinement_prompts(Path(__file__).parent / "trial_prompts")
    prepared = []
    seen = set()
    for case in cases:
        case_id = case["case_id"]
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError("case_id must be a unique nonempty string")
        seen.add(case_id)
        original = parse_space_text(case["space_text"])
        prepared.append((case, original, build_refinement_messages(case["trial_text"], original, prompts)))
    parser = resolve_parser_name(args.model)
    template = resolve_chat_template_kwargs(args.model, enable_thinking=True)
    defaults = resolve_sampling_defaults(args.model, enable_thinking=True)
    manifest = {
        "version": VERSION, "cases": cases, "prompts": prompts,
        "model": args.model, "tokenizer": args.tokenizer, "endpoint": args.endpoint,
        "reasoning_parser": parser, "chat_template_kwargs": template,
        "sampling": defaults, "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len, "max_attempts": args.max_attempts,
        "implementation_sha256": hashlib.sha256(
            Path(__file__).read_bytes() + Path(__file__).with_name("trial_space_refinement.py").read_bytes()
        ).hexdigest(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Refinement inputs/settings changed; use a fresh output directory")
    else:
        if any(args.output_dir.iterdir()):
            raise ValueError("Nonempty output directory lacks a refinement manifest")
        save(manifest_path, manifest)
    tokenizer = load_tokenizer(args.tokenizer, trust_remote_code=True)
    work = make_completion_work_fn(
        CompletionSampling(model=args.model, request_timeout=args.timeout, **defaults),
        parser, tokenizer,
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    async with AsyncOpenAI(
        base_url=args.endpoint, api_key=os.environ.get(args.api_key_env, "not-needed"),
        max_retries=0,
    ) as client:
        models = await client.models.list()
        if args.model not in {m.id for m in models.data}:
            raise ValueError("Requested model is not served by this endpoint")
        save(args.output_dir / "server_models.json", models.model_dump())

        async def review(index, case, original, initial_messages):
            directory = args.output_dir / f"case_{index:04d}"
            directory.mkdir(exist_ok=True)
            target = directory / "result.json"
            if target.exists():
                return json.loads(target.read_text())
            save(directory / "input.json", case)
            messages, started = initial_messages, time.time()
            failures = sorted(p for p in directory.glob("attempt_*.json")
                              if re.fullmatch(r"attempt_\d+\.json", p.name))
            # Resume an interrupted test without overwriting earlier raw traces.
            if failures:
                last = json.loads(failures[-1].read_text())
                if "validation_error" in last:
                    messages = refinement_retry_messages(initial_messages, last["final"], last["validation_error"])
                elif "final" in last:
                    result = parse_refinement_response(last["final"], original, last["finish_reason"])
                    record = {"case_id": case["case_id"], "nct_id": case["nct_id"],
                              "original_space": original, "result": result,
                              "refined_space_texts": [format_space(s) for s in result["spaces"]],
                              "attempts": len(failures), "restored_from_saved_response": True}
                    save(target, record)
                    return record
            async with semaphore:
                for attempt in range(len(failures), args.max_attempts):
                    # All attempts retain reasoning; never truncate source/primer.
                    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **template)
                    tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
                    available = args.max_model_len - tokens - 256
                    if available < 1:
                        raise ValueError("Refinement prompt exceeds context budget")
                    payload = {"prompt": prompt, "max_tokens": min(args.max_tokens, available),
                               "enable_thinking": True, "include_completion_metadata": True,
                               "include_raw_text": True}
                    stem = directory / f"attempt_{attempt + 1:02d}"
                    save(stem.with_suffix(".messages.json"), messages)
                    stem.with_suffix(".prompt.txt").write_text(prompt)
                    print(f"START {case['case_id']} attempt={attempt + 1} input_tokens={tokens}", flush=True)
                    try:
                        reasoning, final, metadata = await work(client, payload)
                    except Exception as exc:
                        save(stem.with_suffix(".json"), {"transport_error": str(exc)})
                        if attempt + 1 == args.max_attempts:
                            raise
                        await asyncio.sleep(min(2 ** attempt, 15))
                        continue
                    audit = {"final": final, "reasoning": reasoning, **metadata,
                             "prompt_tokens": tokens, "enable_thinking": True}
                    save(stem.with_suffix(".json"), audit)
                    try:
                        result = parse_refinement_response(final, original, metadata["finish_reason"])
                    except ValueError as exc:
                        audit["validation_error"] = str(exc)
                        save(stem.with_suffix(".json"), audit)
                        messages = refinement_retry_messages(initial_messages, final, str(exc))
                        print(f"RETRY {case['case_id']}: {exc}", flush=True)
                        continue
                    record = {"case_id": case["case_id"], "nct_id": case["nct_id"],
                              "original_space": original, "result": result,
                              "refined_space_texts": [format_space(s) for s in result["spaces"]],
                              "attempts": attempt + 1, "elapsed_seconds": time.time() - started}
                    save(target, record)
                    print(f"DONE {case['case_id']} status={result['status']} spaces={len(result['spaces'])}", flush=True)
                    return record
                raise ValueError(f"Refinement retries exhausted for {case['case_id']}; all attempts saved")

        outcomes = await asyncio.gather(
            *(review(i, *item) for i, item in enumerate(prepared)), return_exceptions=True,
        )
    records, errors = [], []
    for item, outcome in zip(prepared, outcomes, strict=True):
        if isinstance(outcome, BaseException):
            errors.append({"case_id": item[0]["case_id"], "error": str(outcome)})
        else:
            records.append(outcome)
    save(args.output_dir / "results.json", {"results": records, "errors": errors})
    if errors:
        raise RuntimeError(f"{len(errors)} refinement cases failed; see results.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--endpoint", required=True, help="Explicit existing vLLM /v1 URL")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=50000)
    parser.add_argument("--max-model-len", type=int, default=250000)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
