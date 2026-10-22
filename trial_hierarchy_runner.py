"""Training transports for the shared hierarchical trial-space contract."""

import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import pandas as pd
from vllm_reasoning_utils import load_tokenizer, resolve_chat_template_kwargs

from trial_hierarchy import (
    VERSION,
    FIELDS,
    Completion,
    Hierarchy,
    ReadyHierarchy,
    ResponseCache,
    load_prompts,
    validate_completion,
    retry_messages,
    retry_thinking,
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
    max_attempts=4,
    messages=None,
    always_think=False,
):
    thinking = True if always_think else retry_thinking(True, failures, max_attempts)
    template = resolve_chat_template_kwargs("beam" if beam else "", enable_thinking=thinking)
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


def _save_results(states, trials, directory, output_name, *, raise_errors=True):
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
        "rejections": {str(row.nct_id): state.rejection
                       for state, row in zip(states, trials.itertuples(), strict=True)
                       if state.rejection},
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
                "rejection": state.rejection,
            },
            ensure_ascii=False,
        )
        (trace_dir / f"{output_name}-{state.trial_index:06d}.json").write_text(trace)
        if state.error:
            continue
        final = state.output()
        spaces, boilerplate = ("", "") if state.rejection else final.split("\nBoilerplate exclusions:\n", 1)
        rows.append(
            {
                **row.to_dict(),
                "space_reasoning_and_output": trace,
                "space_output_no_reasoning": final,
                "space_text": spaces,
                "trial_boilerplate_text": boilerplate,
                "space_extraction_status": "rejected" if state.rejection else "completed",
                "space_rejection_reason": state.rejection.get("reason", ""),
                "space_rejection_evidence": state.rejection.get("evidence", ""),
            }
        )
    if errors:
        if not raise_errors:
            return []
        raise RuntimeError(
            f"Hierarchical extraction incomplete: {errors}. Successful nodes are checkpointed; no partial final CSVs were written."
        )
    target = directory / output_name
    temporary = target.with_suffix(".tmp")
    pd.DataFrame(rows).to_parquet(temporary, index=False)
    temporary.replace(target)
    return [str(target)]


def remote_main(args, trials, reasoning_parser):
    from remote_vllm_pool import (
        CompletionSampling,
        OutputValidationError,
        RequestSession,
        build_registry_from_args,
        make_completion_work_fn,
    )

    tokenizer = load_tokenizer(
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
    completion_fn = make_completion_work_fn(sampling, reasoning_parser, tokenizer)
    registry = build_registry_from_args(args)
    cache = ResponseCache(
        Path(args.work_dir) / "hierarchy_nodes",
        json.loads((Path(args.work_dir) / "extraction_manifest.json").read_text()),
    )
    audit_path = Path(args.work_dir) / "validation_failures.jsonl"

    async def work_fn(client, payload):
        request = payload["request"]
        failures = payload.get("failures", 0)
        outgoing = await asyncio.get_running_loop().run_in_executor(
            preparation,
            partial(
                _prepare_payload, request, tokenizer, args.max_tokens,
                args.max_model_len, beam=beam, failures=failures,
                max_attempts=int(getattr(args, "max_attempts", 4)),
                messages=payload.get("messages"),
                always_think=getattr(args, "hierarchy_always_think", False),
            ),
        )
        reasoning, final, metadata = await completion_fn(client, outgoing)
        result = Completion(
            final, reasoning, metadata["finish_reason"], metadata.get("raw_text", "")
        )
        try:
            validate_completion(request, result)
        except ValueError as exc:
            payload["failures"] = failures + 1
            payload["messages"] = retry_messages(request, result, str(exc), failures=failures + 1)
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
                            "next_enable_thinking": (True if getattr(args, "hierarchy_always_think", False)
                                else retry_thinking(True, failures + 1, int(getattr(args, "max_attempts", 4)))),
                            "retry_feedback": payload["messages"][-1]["content"],
                            **vars(result),
                        }
                    )
                    + "\n"
                )
            # Content errors must not reduce pool concurrency.
            raise OutputValidationError(str(exc)) from exc
        await asyncio.to_thread(
            cache.put, request, result
        )  # Persist immediately, without waiting for a full wave/shard.
        return result

    async def run():
        # The caller has validated the prompt/runtime and selected-input manifests.
        # Reuse atomic completed shards without replaying every accepted node or
        # rewriting their original completion timestamps on a worker restart.
        semaphore = asyncio.Semaphore(16)

        async def existing(index, nct_id):
            async with semaphore:
                return await asyncio.to_thread(
                    _completed_trial_path, Path(args.work_dir), index, nct_id
                )

        saved = await asyncio.gather(*(
            existing(i, str(row.nct_id)) for i, row in enumerate(trials.itertuples())
        ))
        paths = {i: path for i, path in enumerate(saved) if path is not None}
        if paths:
            print(f"[hierarchy] reusing {len(paths)} completed trial shards", flush=True)
        errors = {}
        # Include previously checkpointed rejections in the run-level audit.
        rejections = {}
        for path in paths.values():
            rejections.update(json.loads(Path(path).with_suffix('.parquet.status.json').read_text()).get('rejections', {}))

        def save_rejections():
            target = Path(args.work_dir) / "extraction_rejections.json"
            temporary = target.with_suffix('.tmp')
            temporary.write_text(json.dumps(rejections, indent=2))
            temporary.replace(target)

        save_rejections()
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
            if i not in paths
        )

        async def finished(state):
            index = state.trial_index
            saved = await asyncio.to_thread(
                _save_results,
                [state],
                trials.iloc[[index]],
                args.work_dir,
                f"remote_output_shard_{index:06d}.parquet",
                raise_errors=False,
            )
            if state.error:
                errors[str(trials.iloc[index].nct_id)] = state.error
                failure_path = Path(args.work_dir) / "extraction_failures.json"
                temporary = failure_path.with_suffix(".tmp")
                temporary.write_text(json.dumps(errors, indent=2))
                temporary.replace(failure_path)
                print(f"[hierarchy] trial {index + 1}/{len(trials)} needs review: {state.error}; "
                      "continuing remaining trials", flush=True)
                return
            paths[index] = saved[0]
            if state.rejection:
                rejections[str(trials.iloc[index].nct_id)] = state.rejection
                save_rejections()
            print(
                f"[hierarchy] {'rejected' if state.rejection else 'completed'} trial {index + 1}/{len(trials)}; "
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
                cached = await asyncio.to_thread(cache.get, request)
                if cached is not None:
                    return cached
                if request.recovery_round:
                    print(f"[hierarchy] recovering {trials.iloc[request.trial_index].nct_id} "
                          f"field={request.field_name} round={request.recovery_round} "
                          f"replacing={request.recovery_of}", flush=True)
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
        failure_path = Path(args.work_dir) / "extraction_failures.json"
        temporary = failure_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(errors, indent=2))
        temporary.replace(failure_path)
        if errors:
            raise RuntimeError(
                f"Hierarchical extraction incomplete: {len(errors)} trials need review; "
                f"{len(paths)} complete trials remain checkpointed. See {failure_path}. "
                "No partial final CSVs were written."
            )
        return [paths[index] for index in sorted(paths)]

    # Serialize access to the tokenizer on a dedicated worker; do not let a
    # burst of CPU preparation block HTTP dispatch or checkpoint I/O workers.
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="trial-prompts") as preparation:
        return asyncio.run(run())


def _completed_trial_path(directory, index, nct_id):
    """Verify an already atomically published one-trial shard before skipping it."""
    import pyarrow.parquet as parquet

    path = directory / f"remote_output_shard_{index:06d}.parquet"
    if not path.exists():
        return None
    status = json.loads(path.with_suffix('.parquet.status.json').read_text())
    if status.get('method') != VERSION or status.get('complete') is not True or status.get('trials') != 1:
        raise ValueError(f"Completed trial shard has incompatible status: {path}")
    saved = parquet.ParquetFile(path)
    required = {'nct_id', 'space_text', 'trial_boilerplate_text', 'space_reasoning_and_output',
                'space_output_no_reasoning'}
    if saved.metadata.num_rows != 1 or not required.issubset(saved.schema_arrow.names):
        raise ValueError(f"Completed trial shard has invalid schema/row count: {path}")
    if str(saved.read(columns=['nct_id']).column('nct_id')[0].as_py()) != nct_id:
        raise ValueError(f"Completed trial shard belongs to a different trial: {path}")
    return str(path)


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
    while True:
        groups = [s.requests() for s in states]
        requests = [r for group in groups for r in group]
        if not requests:
            break
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
                        max_attempts=getattr(args, "max_attempts", 4),
                        messages=feedback.get(i),
                        always_think=getattr(args, "hierarchy_always_think", False),
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
                        validate_completion(chunk[i], answer)
                    except ValueError as exc:
                        feedback[i] = retry_messages(chunk[i], answer, str(exc), failures=attempt + 1)
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
                                        "enable_thinking": payload["enable_thinking"],
                                        "retry_feedback": feedback[i][-1]["content"],
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
