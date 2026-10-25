"""Checkpointed serial patient summaries on the shared adaptive endpoint pool."""
import argparse
import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import uuid

from remote_vllm_pool import (CompletionSampling, OutputValidationError, RequestSession,
                              build_registry_from_args, make_completion_work_fn)
from matchminer_ai.patients.postprocess import normalize_patient_summary


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def add_arguments(parser):
    parser.add_argument('--scheduler-adaptive-concurrency', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--initial-concurrent-per-server', type=int, default=16)
    parser.add_argument('--scheduler-metrics-file')
    parser.add_argument('--pending-requests', type=int, default=8192)


def checkpoint_key(prompt, sampling, parser_name, template):
    value = dict(prompt=prompt, sampling=asdict(sampling), parser=parser_name, template=str(template))
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def traced_work(base, root, sampling):
    async def work(client, payload):
        reasoning, answer, metadata = await base(client, payload)
        trace = dict(chunk_index=payload['chunk_index'], endpoint=str(client.base_url),
                     prompt=payload['prompt'], reasoning=reasoning, answer=answer,
                     sampling=asdict(sampling), generated_at=datetime.now(timezone.utc).isoformat(), **metadata)
        path = root / 'attempts' / f'{payload["chunk_index"] // 256:05d}' / f'{payload["chunk_index"]}_{uuid.uuid4()}.json'
        await asyncio.to_thread(save, path, trace)
        if metadata['finish_reason'] != 'stop' or not answer.strip() or answer.lstrip().startswith('ERROR:'):
            raise OutputValidationError('Patient summary is empty, failed, or incomplete')
        return dict(reasoning=reasoning, summary=normalize_patient_summary(answer), trace_path=str(path))
    return work


async def run_rounds(*, rounds, patient_chunk_order, patient_summaries, all_results,
                     completed_rounds, args, prompt_pool, parser_name, tokenizer,
                     build_prompt, save_round):
    root = Path(args.shard_dir)
    pending = getattr(args, 'pending_requests', 8192)
    if pending < 1:
        raise ValueError('Summary pending requests must be positive')
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    needed = 2 * pending + 2048
    if hard != resource.RLIM_INFINITY and hard < needed:
        raise RuntimeError('Open-file hard limit is too small for the summary pool')
    if soft != resource.RLIM_INFINITY and soft < needed:
        resource.setrlimit(resource.RLIMIT_NOFILE, (needed, hard))
    args.scheduler_metrics_file = getattr(args, 'scheduler_metrics_file', None) or str(root / 'concurrency_status.json')
    if args.max_concurrent_per_server is None:
        args.max_concurrent_per_server = int(args.max_concurrent_requests)
    sampling = CompletionSampling(model=args.model, request_timeout=float(args.request_timeout),
        **{k: getattr(args, k) for k in ('temperature', 'top_k', 'top_p', 'presence_penalty', 'min_p', 'repetition_penalty')})
    work = traced_work(make_completion_work_fn(sampling, parser_name, tokenizer), root, sampling)
    chunk_to_patient = {cidx: pid for pid, chunks in patient_chunk_order.items() for cidx in chunks}
    total_chunks = sum(map(len, rounds))
    for round_idx in range(completed_rounds, len(rounds)):
        items = rounds[round_idx]
        priors = {idx: patient_summaries.get(pid) or 'None - this is the first segment for this patient'
                  for pid, idx, first, last, text in items}
        if any(value.lstrip().startswith('ERROR:') for value in priors.values()):
            raise RuntimeError('A failed prior summary must be repaired before continuing')
        save(root / 'status.json', dict(stage='preparing_prompts', round=round_idx + 1,
             rounds=len(rounds), completed_chunks=len(all_results), expected_chunks=total_chunks))
        inputs = [(idx, patient_summaries.get(pid), first, last, text, args.max_model_len)
                  for pid, idx, first, last, text in items]
        prompts = prompt_pool.map(build_prompt, inputs, chunksize=max(1, len(inputs) // (prompt_pool._processes * 4)))
        iterator, completed = iter(prompts), {}
        registry = build_registry_from_args(args)
        async with RequestSession(registry, work, max_attempts=int(args.max_attempts)) as session:
            async def worker():
                while (item := next(iterator, None)) is not None:
                    idx, prompt, tokens = item
                    available = args.max_model_len - tokens - 256
                    if available <= 0:
                        raise ValueError('Full summary prompt exceeds context')
                    max_tokens = min(available, args.max_tokens) if args.max_tokens is not None else available
                    identity = checkpoint_key(prompt, sampling, parser_name, tokenizer.chat_template)
                    # Output budget and the prior summary are part of checkpoint identity.
                    identity = hashlib.sha256((identity + str(max_tokens) + priors[idx]).encode()).hexdigest()
                    path = root / 'chunks' / f'{idx // 256:05d}' / f'{idx:09d}.json'
                    if path.exists():
                        result = await asyncio.to_thread(lambda: json.loads(path.read_text()))
                        if result['identity'] != identity:
                            raise ValueError('Saved summary prompt or sampling changed')
                        if not result['summary'].strip() or result['summary'].lstrip().startswith('ERROR:'):
                            raise ValueError('Saved patient summary is invalid')
                    else:
                        result = await session.submit(dict(chunk_index=idx, prompt=prompt, max_tokens=max_tokens,
                            include_completion_metadata=True, include_raw_text=True))
                        if not isinstance(result, dict):
                            raise RuntimeError(f'Patient summary failed for chunk {idx}: {result}')
                        result.update(identity=identity, prior_summary=priors[idx], patient_id=chunk_to_patient[idx])
                        await asyncio.to_thread(save, path, result)
                    completed[idx] = (result['reasoning'], result['summary'], priors[idx])
                    if len(completed) % 32 == 0 or len(completed) == len(items):
                        save(root / 'status.json', dict(stage='summarizing', round=round_idx + 1,
                             rounds=len(rounds), completed_chunks=len(all_results) + len(completed), expected_chunks=total_chunks))
            tasks = [asyncio.create_task(worker()) for _ in range(min(pending, len(items)))]
            try:
                await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        if set(completed) != {item[1] for item in items}:
            raise RuntimeError('Incomplete summary round')
        save_round(args.shard_dir, round_idx, [(idx, *completed[idx]) for idx in sorted(completed)])
        all_results.update(completed)
        for idx, (_, summary, _) in completed.items():
            patient_summaries[chunk_to_patient[idx]] = summary
    save(root / 'status.json', dict(stage='completed', completed_chunks=len(all_results), expected_chunks=total_chunks))
