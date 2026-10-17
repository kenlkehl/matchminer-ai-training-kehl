"""Training transports for the shared hierarchical trial-space contract."""

import asyncio
import json
import time
from pathlib import Path

import pandas as pd

from trial_hierarchy import (
    VERSION,
    Completion,
    Hierarchy,
    ReadyHierarchy,
    ResponseCache,
    load_prompts,
    parse_response,
    retry_messages,
    run_ready_hierarchies,
)


def prompt_bundle():
    return load_prompts(Path(__file__).parent / "trial_prompts")


def _prepare_payload(
    request,
    tokenizer,
    max_tokens,
    max_model_len,
    *,
    beam=False,
    failures=0,
    messages=None,
):
    thinking = (failures // 2) % 2 == 0
    template = (
        {"reasoning_effort": "max" if thinking else "low"}
        if beam
        else {"enable_thinking": thinking}
    )
    messages = messages or request.messages
    prompt = tokenizer.apply_chat_template(
        conversation=messages, add_generation_prompt=True, tokenize=False, **template
    )
    count = len(tokenizer.encode(prompt, add_special_tokens=False))
    available = max_model_len - count - 256
    if available < 1:
        raise ValueError(
            f"{request.field_name} prompt requires {count} tokens; context {max_model_len} leaves no output budget"
        )
    return {
        "prompt": prompt,
        "max_tokens": min(max_tokens, available),
        "enable_thinking": thinking,
        "include_completion_metadata": True,
        "include_raw_text": True,
        **({"messages": messages, "chat_template_kwargs": template} if beam else {}),
    }


def _states(trials, args):
    prompts = prompt_bundle()
    return [
        Hierarchy(
            i,
            str(row.trial_text),
            prompts,
            max_nodes=getattr(args, "hierarchy_max_nodes", 4096),
            max_calls=getattr(args, "hierarchy_max_calls", 20000),
        )
        for i, row in enumerate(trials.itertuples())
    ]


def _save_results(states, trials, directory, output_name):
    directory = Path(directory)
    errors = {
        str(row.nct_id): state.error
        for state, row in zip(states, trials.itertuples(), strict=True)
        if state.error
    }
    status = {
        "method": VERSION,
        "trials": len(states),
        "errors": errors,
        "calls": sum(s.calls for s in states),
        "complete": not errors,
    }
    (directory / f"{output_name}.status.json").write_text(json.dumps(status, indent=2))
    # Full per-node source context, reasoning and final answers are retained even
    # when a different branch fails. Successful nodes are also cached individually.
    trace_dir = directory / "hierarchy_traces"
    trace_dir.mkdir(exist_ok=True)
    rows = []
    for state, (_, row) in zip(states, trials.iterrows(), strict=True):
        trace = json.dumps(
            {
                "version": VERSION,
                "nct_id": str(row.nct_id),
                "nodes": state.trace,
                "error": state.error,
            },
            ensure_ascii=False,
        )
        (trace_dir / f"{output_name}-{state.trial_index:06d}.json").write_text(trace)
        if state.error:
            continue
        final = state.output()
        spaces, boilerplate = final.split("\nBoilerplate exclusions:\n", 1)
        rows.append(
            {
                **row.to_dict(),
                "space_reasoning_and_output": trace,
                "space_output_no_reasoning": final,
                "space_text": spaces,
                "trial_boilerplate_text": boilerplate,
            }
        )
    if errors:
        raise RuntimeError(
            f"Hierarchical extraction incomplete: {errors}. Successful nodes are checkpointed; no partial final CSVs were written."
        )
    target = directory / output_name
    temporary = target.with_suffix(".tmp")
    pd.DataFrame(rows).to_parquet(temporary, index=False)
    temporary.replace(target)
    return [str(target)]


def remote_main(args, trials, reasoning_parser):
    from transformers import AutoTokenizer

    from remote_vllm_pool import (
        CompletionSampling,
        OutputValidationError,
        RequestSession,
        build_registry_from_args,
        make_chat_completion_work_fn,
        make_completion_work_fn,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        getattr(args, "tokenizer_model", None) or args.model,
        cache_dir=args.download_dir or None,
        trust_remote_code=True,
    )
    beam = reasoning_parser == "beam"
    sampling = CompletionSampling(
        **{
            key: getattr(args, key)
            for key in (
                "model",
                "temperature",
                "top_p",
                "top_k",
                "min_p",
                "presence_penalty",
                "repetition_penalty",
            )
        },
        request_timeout=float(getattr(args, "request_timeout", 600)),
    )
    completion_fn = (
        make_chat_completion_work_fn(sampling)
        if beam
        else make_completion_work_fn(sampling, reasoning_parser, tokenizer)
    )
    registry = build_registry_from_args(args)
    cache = ResponseCache(
        Path(args.work_dir) / "hierarchy_nodes",
        json.loads((Path(args.work_dir) / "extraction_manifest.json").read_text()),
    )
    audit_path = Path(args.work_dir) / "validation_failures.jsonl"

    async def work_fn(client, payload):
        request = payload["request"]
        failures = payload.get("failures", 0)
        outgoing = _prepare_payload(
            request,
            tokenizer,
            args.max_tokens,
            args.max_model_len,
            beam=beam,
            failures=failures,
            messages=payload.get("messages"),
        )
        reasoning, final, metadata = await completion_fn(client, outgoing)
        result = Completion(
            final, reasoning, metadata["finish_reason"], metadata.get("raw_text", "")
        )
        try:
            parse_response(final, request.field_name, result.finish_reason)
        except ValueError as exc:
            payload["failures"] = failures + 1
            payload["messages"] = retry_messages(request, result, str(exc))
            with audit_path.open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "time": time.time(),
                            "nct_id": payload["nct_id"],
                            "field": request.field_name,
                            "request_key": request.key,
                            "failure_number": failures + 1,
                            "error": str(exc),
                            "enable_thinking": outgoing["enable_thinking"],
                            **vars(result),
                        }
                    )
                    + "\n"
                )
            # Content errors must not reduce pool concurrency.
            raise OutputValidationError(str(exc)) from exc
        cache.put(
            request, result
        )  # Persist immediately, without waiting for a full wave/shard.
        return result

    async def run():
        paths = {}
        prompts = prompt_bundle()
        states = (
            ReadyHierarchy(
                i,
                str(row.trial_text),
                prompts,
                max_nodes=getattr(args, "hierarchy_max_nodes", 4096),
                max_calls=getattr(args, "hierarchy_max_calls", 20000),
            )
            for i, row in enumerate(trials.itertuples())
        )

        def finished(state):
            index = state.trial_index
            paths[index] = _save_results(
                [state],
                trials.iloc[[index]],
                args.work_dir,
                f"remote_output_shard_{index:06d}.parquet",
            )[0]
            print(
                f"[hierarchy] completed trial {index + 1}/{len(trials)}; "
                f"{state.calls} nodes",
                flush=True,
            )

        async with RequestSession(
            registry, work_fn, max_attempts=int(getattr(args, "max_attempts", 4))
        ) as session:
            entries = await registry.active_entries()
            capacity = max(1, len(entries)) * int(
                getattr(args, "max_concurrent_per_server", 100)
            )
            capacity_limit = getattr(registry, "request_capacity", lambda: capacity)

            async def complete(request):
                cached = cache.get(request)
                if cached is not None:
                    return cached
                result = await session.submit(
                    {
                        "request": request,
                        "nct_id": str(trials.iloc[request.trial_index].nct_id),
                    }
                )
                return (
                    result
                    if isinstance(result, Completion)
                    else Completion(str(result), finish_reason="error")
                )

            await run_ready_hierarchies(
                states,
                complete,
                concurrency=capacity_limit,
                active_trials=lambda: max(
                    capacity_limit(),
                    int(getattr(args, "hierarchy_trial_batch_size", 32)),
                ),
                on_complete=finished,
            )
        return [paths[index] for index in sorted(paths)]

    return asyncio.run(run())


def local_extract(
    llm, tokenizer, sampling, trials, args, reasoning_parser, output_path
):
    """Keep one local model resident while batching parent requests per field."""
    from copy import copy

    from vllm_reasoning_utils import parse_reasoning_output

    states = _states(trials, args)
    directory = Path(output_path).parent
    cache = ResponseCache(
        directory / "hierarchy_nodes",
        json.loads((directory / "extraction_manifest.json").read_text()),
    )
    for stage in range(10):
        groups = [s.requests() for s in states]
        requests = [r for group in groups for r in group]
        responses = []
        for offset in range(0, len(requests), args.batch_size):
            chunk = requests[offset : offset + args.batch_size]
            answers = {i: cache.get(r) for i, r in enumerate(chunk)}
            feedback = {}
            for attempt in range(getattr(args, "max_attempts", 4)):
                pending = [i for i, answer in answers.items() if answer is None]
                if not pending:
                    break
                payloads = [
                    _prepare_payload(
                        chunk[i],
                        tokenizer,
                        args.max_tokens,
                        args.max_model_len,
                        beam=reasoning_parser == "beam",
                        failures=attempt,
                        messages=feedback.get(i),
                    )
                    for i in pending
                ]
                params = copy(sampling)
                params.max_tokens = min(p["max_tokens"] for p in payloads)
                generated = llm.generate([p["prompt"] for p in payloads], params)
                for i, payload, generation in zip(
                    pending, payloads, generated, strict=True
                ):
                    output = generation.outputs[0]
                    reasoning, final = parse_reasoning_output(
                        output.text,
                        reasoning_parser,
                        tokenizer,
                        enable_thinking=payload["enable_thinking"],
                    )
                    answer = Completion(
                        final, reasoning, str(output.finish_reason), output.text
                    )
                    try:
                        parse_response(final, chunk[i].field_name, answer.finish_reason)
                    except ValueError as exc:
                        feedback[i] = retry_messages(chunk[i], answer, str(exc))
                        with (
                            directory
                            / (Path(output_path).stem + ".validation_failures.jsonl")
                        ).open("a") as stream:
                            stream.write(
                                json.dumps(
                                    {
                                        "request_key": chunk[i].key,
                                        "field": chunk[i].field_name,
                                        "error": str(exc),
                                        "attempt": attempt + 1,
                                        **vars(answer),
                                    }
                                )
                                + "\n"
                            )
                        if attempt + 1 == getattr(args, "max_attempts", 4):
                            answers[i] = Completion(
                                str(exc), reasoning, "error", output.text
                            )
                    else:
                        answers[i] = answer
                        cache.put(chunk[i], answer)
            responses.extend(answers[i] for i in range(len(chunk)))
        offset = 0
        for state, group in zip(states, groups, strict=True):
            if group:
                state.accept(group, responses[offset : offset + len(group)])
            offset += len(group)
    return _save_results(states, trials, directory, Path(output_path).name)
