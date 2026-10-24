"""Resumable five-question teacher labels using the existing adaptive vLLM pool."""
import asyncio
from dataclasses import asdict, replace
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import uuid

from remote_vllm_pool import (CompletionSampling, OutputValidationError, RequestSession,
                              build_registry_from_args, make_completion_work_fn)
from vllm_reasoning_utils import (load_tokenizer, resolve_chat_template_kwargs,
                                 resolve_parser_name, resolve_sampling_defaults)
from .contracts import TASKS, contract_hash, judge_messages, parse_judgment, prompt_sources
from .data import digest, load_corpus, read_jsonl, write_json, write_jsonl


def validate_served_teacher(models, *, requested, tokenizer):
    """Check the advertised root, including the approved equivalent Gemma quantizations."""
    matches = [model for model in models if model['id'] == requested]
    if len(matches) != 1:
        raise ValueError('Requested teacher model is not uniquely advertised: ' + requested)
    served = matches[0]
    gemma31 = {'google/gemma-4-31B-it', 'nvidia/Gemma-4-31B-IT-NVFP4',
               'RedHatAI/Gemma-4-31B-IT-FP8-Dynamic', 'RedHatAI/gemma-4-31B-it-FP8-Dynamic'}
    allowed = gemma31 if tokenizer in gemma31 else {tokenizer}
    if served.get('root', served['id']) not in allowed:
        raise ValueError('Served teacher root differs from the configured model family')
    return served


def label_key(row):
    return digest([row['patient_id'], row['cohort_id'], row['task']])


def questions(pairs, patients, cohorts, *, sources=None):
    patients, cohorts = {r['id']: r for r in patients}, {r['id']: r for r in cohorts}
    seen = set()
    for pair in pairs:
        p, c = patients[pair['patient_id']], cohorts[pair['cohort_id']]
        if p['split'] != c['split'] or pair['split'] != p['split']:
            raise ValueError('Teacher pairs must stay within their corpus split')
        for task in TASKS:
            item = {**pair, 'tasks': list(TASKS), 'task': task}
            key = label_key(item)
            if key in seen:
                continue
            seen.add(key)
            yield {**item, 'key': key, 'messages': judge_messages(task, p['text'], c['text'], sources=sources)}


async def label(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'labeling.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return await _label(args, output)


async def _label(args, output):
    patients, cohorts = load_corpus(args.corpus)
    pairs = read_jsonl(args.candidates)
    sources = prompt_sources()
    tokenizer = load_tokenizer(args.teacher_tokenizer, cache_dir=args.cache_dir)
    defaults = resolve_sampling_defaults(args.teacher_tokenizer)
    sampling = CompletionSampling(model=args.teacher_model, request_timeout=args.request_timeout, **defaults)
    signature = dict(contract=contract_hash(sources), patients=digest(patients), cohorts=digest(cohorts),
                     pairs=digest(pairs), sampling=asdict(sampling), tokenizer=args.teacher_tokenizer,
                     template=digest(tokenizer.chat_template), max_tokens=args.max_tokens,
                     context_length=args.teacher_context)
    gated = getattr(args, 'gated_regression', False)
    if gated:
        from .regression import score_contract
        signature['aggregation_contract'] = score_contract(signature['contract'])
    manifest = output / 'labeling_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != signature:
        raise ValueError('Teacher identity changed; use a new label directory')
    write_json(manifest, signature)
    write_json(output / 'prompt_sources.json', sources)
    checkpoint_dir = output / 'responses'
    checkpoint_dir.mkdir(exist_ok=True)
    served_name = getattr(args, 'teacher_served_model', None) or args.teacher_model
    base = make_completion_work_fn(replace(sampling, model=served_name),
                                   resolve_parser_name(args.teacher_tokenizer, 'auto'), tokenizer)
    server_models = {}

    async def work(client, payload):
        endpoint = str(client.base_url)
        if endpoint not in server_models:
            catalog = await client.models.list()
            served = validate_served_teacher([m.model_dump() for m in catalog.data],
                                             requested=served_name, tokenizer=args.teacher_tokenizer)
            if served.get('max_model_len', args.teacher_context) < args.teacher_context:
                raise ValueError('Teacher server context is smaller than the configured budget')
            server_models[endpoint] = served
        reasoning, answer, metadata = await base(client, payload)
        trace = dict(endpoint=endpoint, served_model=server_models[endpoint], reasoning=reasoning,
                     raw_answer=answer, **metadata)
        try:
            if metadata['finish_reason'] != 'stop':
                raise ValueError('Incomplete teacher output: ' + metadata['finish_reason'])
            result = parse_judgment(answer)
        except (ValueError, TypeError, IndexError) as error:
            write_json(output / 'invalid_responses' / (str(uuid.uuid4()) + '.json'),
                       dict(key=payload['key'], error=str(error), **trace))
            raise OutputValidationError(str(error)) from error
        return dict(**result, **trace, generated_at=datetime.now(timezone.utc).isoformat())

    registry = build_registry_from_args(args)
    def groups():
        seen = set()
        for pair in pairs:
            key = (pair['patient_id'], pair['cohort_id'])
            if key not in seen:
                seen.add(key)
                yield list(questions([pair], patients, cohorts, sources=sources))
    iterator = iter(groups() if gated else questions(pairs, patients, cohorts, sources=sources))
    completed, failures, scores = {}, {}, {}
    async with RequestSession(registry, work, max_attempts=args.max_attempts) as session:
        async def ask(item):
            path = checkpoint_dir / (item['key'] + '.json')
            if path.exists():
                row = json.loads(path.read_text())
                if row['key'] != item['key'] or row['prompt_sha256'] != digest(item['messages']):
                    raise ValueError('Checkpoint key or clinical text mismatch')
                parse_judgment(json.dumps({k: row[k] for k in ('answer', 'evidence_status', 'reason')}))
                completed[item['key']] = row
                return row
            prompt = tokenizer.apply_chat_template(item['messages'], tokenize=False,
                add_generation_prompt=True, **resolve_chat_template_kwargs(args.teacher_tokenizer))
            tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
            if tokens + args.max_tokens + 256 > args.teacher_context:
                raise ValueError('Full teacher input and output exceed context; truncation is disabled')
            result = await session.submit(dict(prompt=prompt, key=item['key'], max_tokens=args.max_tokens,
                enable_thinking=True, include_completion_metadata=True, include_raw_text=True))
            if not isinstance(result, dict):
                failures[item['key']] = str(result)
                return None
            row = {k: v for k, v in item.items() if k != 'messages'}
            row.update(result, prompt_sha256=digest(item['messages']), contract=signature['contract'],
                       teacher_model=args.teacher_model)
            write_json(path, row)
            completed[item['key']] = row
            write_json(output / 'status.json', dict(state='labeling', completed=len(completed), failures=failures))
            return row

        async def worker():
            while (item := next(iterator, None)) is not None:
                if not gated:
                    await ask(item)
                    continue
                from .regression import gated_judgments, aggregate_pair
                rows = await gated_judgments(item, ask)
                if rows is not None:
                    score = aggregate_pair(item[0], rows, question_contract=signature['contract'])
                    scores[digest([score['patient_id'], score['cohort_id']])] = score
                    write_json(output / 'status.json', dict(state='labeling', completed=len(completed),
                        scored_pairs=len(scores), failures=failures))

        tasks = [asyncio.create_task(worker()) for _ in range(args.pending_requests)]
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    write_jsonl(output / 'labels.jsonl', [completed[k] for k in sorted(completed)])
    if gated:
        from .regression import score_balance
        write_jsonl(output / 'scores.jsonl', [scores[k] for k in sorted(scores)])
        write_json(output / 'score_balance.json', score_balance(list(scores.values())))
    write_json(output / 'status.json', dict(state='incomplete' if failures else 'complete',
                                          completed=len(completed), failures=failures,
                                          **(dict(scored_pairs=len(scores)) if gated else {})))
    if failures:
        raise RuntimeError(f'{len(failures)} unresolved teacher labels; resume this command to retry')
    return list(completed.values())
