#!/usr/bin/env python3
"""Resumable event lists and notes using the existing 1a/1b/2 prompts.

One shared live endpoint pool serves both classes of histories and all notes.
Bounded preparation and atomic shards avoid materializing millions of prompts.
This command does not train models, summarize patients, or stop serving machines.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import re
import uuid

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from remote_vllm_pool import (
    CompletionSampling, OutputValidationError, RequestSession,
    add_remote_cli_args, build_registry_from_args, make_completion_work_fn,
)
from vllm_reasoning_utils import resolve_parser_name, resolve_sampling_defaults

HERE = Path(__file__).resolve().parent
EVENT_TYPES = {'demographics', 'diagnosis', 'systemic', 'surgery', 'radiation',
               'adverse_event', 'clinical_note', 'imaging_report', 'pathology_report', 'ngs_report'}
REPORT_TYPES = {'clinical_note', 'imaging_report', 'pathology_report', 'ngs_report'}
PATIENTS_PER_CLASS = 5


def load_stage(filename):
    spec = importlib.util.spec_from_file_location(Path(filename).stem, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def now():
    return datetime.now(timezone.utc).isoformat()


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def save_frame(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


async def finish_checkpoint(coroutine):
    """Do not leave a thread writing a shard after its writer lock is released."""
    task = asyncio.create_task(coroutine)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        raise


def checkpoint_paths(directory):
    # Keep each directory small on NFS while retaining flat-checkpoint readability.
    return sorted([*directory.glob('shard_*.parquet'), *directory.glob('*/shard_*.parquet')],
                  key=lambda path: path.name)


def parse_histories(answer):
    """Validate the requested five patients without rewriting clinical content."""
    histories = []
    for section in answer.split('<new_patient>'):
        # A literal inequality (e.g. "margin < 1 mm") is clinical content;
        # only a complete tag begins the next event. Unknown tags still fail below.
        events = re.findall(r'<(\w+)>(.*?)(?=<\w+>|\Z)', section, re.DOTALL)
        if not events:
            continue  # A heading or a leading/trailing delimiter is not a patient.
        if any(kind not in EVENT_TYPES or not text.strip() for kind, text in events):
            raise OutputValidationError('Unknown event tag or empty event')
        kinds = {kind for kind, _ in events}
        if 'demographics' not in kinds or not kinds.intersection(REPORT_TYPES):
            raise OutputValidationError('Each patient needs demographics and report events')
        histories.append([(kind, text.strip().replace('\n', ' ')) for kind, text in events])
    if len(histories) != 5:
        raise OutputValidationError(f'Expected five patient histories; got {len(histories)}')
    return histories


def tagged_events(text):
    """Parse flat event tags, preserving inequalities and clinical content."""
    text = text.strip()
    if text.startswith('```') and text.endswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    # Balanced XML-style event closers are harmless formatting. Remove only
    # matched, non-nested pairs; mismatched or nested markup is still rejected.
    text = re.sub(r'<(\w+)>(.*?)</\1>',
                  lambda match: match[0] if re.search(r'</?\w+>', match[2]) else f'<{match[1]}>{match[2]}',
                  text, flags=re.DOTALL)
    if re.search(r'</\w+>', text):
        raise OutputValidationError('Use flat event tags without mismatched or nested closing tags')
    if not text.startswith('<'):
        raise OutputValidationError('Return tagged events only, without headings or commentary')
    events = re.findall(r'<(\w+)>(.*?)(?=<\w+>|\Z)', text, re.DOTALL)
    if not events or any(kind not in EVENT_TYPES or not value.strip() for kind, value in events):
        raise OutputValidationError('Unknown event tag or empty event')
    return [(kind, ' '.join(value.split())) for kind, value in events]


def check_event_order(events):
    previous = None
    for kind, text in events:
        if kind == 'demographics':
            continue
        match = re.match(r'At age (\d+(?:\.\d+)?)\s*(months?|years?)?', text, re.I)
        if not match:
            raise OutputValidationError(f'Every {kind} event must begin with At age N years')
        age = float(match[1]) / (12 if (match[2] or '').lower().startswith('month') else 1)
        if previous is not None and age < previous:
            raise OutputValidationError(f'Events are out of chronological order: age {age:g} follows {previous:g}')
        previous = age
        if re.search(r'\b(?:study (?:entry|enrollment)|trial (?:screening|enrollment)|(?:this|the) trial)\b', text, re.I):
            raise OutputValidationError('Remove references to trial screening, study entry, or enrollment')


def parse_seeds(answer):
    sections = [s.strip() for s in answer.split('<new_patient>') if s.strip()]
    # A single outer markdown fence is harmless; never reinterpret event tags.
    if answer.strip().startswith('```') and answer.strip().endswith('```'):
        return parse_seeds(answer.strip().split('\n', 1)[1].rsplit('```', 1)[0])
    if len(sections) != PATIENTS_PER_CLASS:
        raise OutputValidationError(f'Expected five patient seeds; got {len(sections)}')
    seeds = []
    demographics = set()
    for section in sections:
        events = tagged_events(section)
        kinds = Counter(kind for kind, _ in events)
        if kinds['demographics'] != 1 or kinds['diagnosis'] < 1 or set(kinds) - {'demographics', 'diagnosis'}:
            raise OutputValidationError('Each seed needs exactly one demographics line and diagnosis lines only')
        if events[0][0] != 'demographics':
            raise OutputValidationError('Begin each seed with its demographics line')
        if events[0][1] in demographics:
            raise OutputValidationError('The five seeds must have distinct patient identities')
        demographics.add(events[0][1])
        check_event_order(events)
        seeds.append('\n'.join(f'<{kind}> {text}' for kind, text in events))
    return seeds


def parse_patient_history(answer, seed_text):
    if '<new_patient>' in answer:
        raise OutputValidationError('Generate exactly one patient, without a new_patient separator')
    events = tagged_events(answer)
    required = tagged_events(seed_text)
    if events[0][0] != 'demographics' or sum(kind == 'demographics' for kind, _ in events) != 1:
        raise OutputValidationError('Generate one patient with one initial demographics line')
    # A missing terminal sentence period is formatting, not a different seed.
    # Restore the saved seed spelling after comparison; all factual text must
    # still match, including names, ages, stages, and biomarkers.
    identity = lambda event: (event[0], event[1].rstrip('.'))
    counts = Counter(map(identity, events))
    if any(counts[identity(event)] != 1 for event in required):
        raise OutputValidationError('Copy every supplied seed demographics/diagnosis line verbatim exactly once')
    if not REPORT_TYPES.intersection(kind for kind, _ in events):
        raise OutputValidationError('The patient history needs report events for note generation')
    check_event_order(events)
    originals = {identity(event): event for event in required}
    return [originals.get(identity(event), event) for event in events]


def validate_completion(answer, metadata, stage, seed_text=None):
    if metadata.get('finish_reason') != 'stop':
        raise OutputValidationError(f"Incomplete completion: {metadata.get('finish_reason')}")
    if not answer.strip() or answer.lstrip().startswith('ERROR:'):
        raise OutputValidationError('Empty or error output')
    if stage == 'events':
        parse_histories(answer)
    elif stage == 'seeds':
        parse_seeds(answer)
    elif stage == 'patient_events':
        parse_patient_history(answer, seed_text)


def event_rows(records, run_identity):
    """Stable global patient IDs shared by event lists, notes, and resume."""
    rows = []
    for record in records:
        space = int(record['space_index'])
        kind_offset = 0 if record['patient_class'] == 'positive' else 5
        for patient_number, events in enumerate(parse_histories(record['answer'])):
            mrn = 10 * space + kind_offset + patient_number + 1
            patient_id = 'patient_' + str(uuid.uuid5(uuid.NAMESPACE_URL, f'{run_identity}/{mrn}'))
            for event_number, (event_type, event_text) in enumerate(events):
                rows.append(dict(pseudo_mrn=mrn, patient_id=patient_id, space_index=space,
                                 nct_id=record['nct_id'], patient_class=record['patient_class'],
                                 event_number=event_number, event_type=event_type, event_text=event_text))
    return pd.DataFrame(rows)


def compose_parquet(paths, destination, expected_rows=None):
    """Stream assembly with an explicit row-count check; never skip unreadable files."""
    destination = Path(destination)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    count, writer = 0, None
    try:
        for path in paths:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
                table = pa.Table.from_batches([batch])
                if writer is None:
                    writer = pq.ParquetWriter(temporary, table.schema, compression='zstd')
                writer.write_table(table)
                count += len(table)
        if writer is None:
            raise ValueError('No rows to assemble')
    finally:
        if writer is not None:
            writer.close()
    if expected_rows is not None and count != expected_rows:
        raise ValueError(f'Assembly has {count} rows, expected {expected_rows}')
    temporary.replace(destination)
    return count


class Generator:
    def __init__(self, args, tokenizer):
        self.args, self.tokenizer = args, tokenizer
        self.out = Path(args.output_dir)
        self.positive = load_stage('1a_make_synthetic_enrollee_prompts.py')
        self.negative = load_stage('1b_make_synthetic_negative_enrollee_prompts.py')
        self.notes = load_stage('2_make_synthetic_notes_sharded.py')
        self.sampling = CompletionSampling(
            model=args.model, request_timeout=args.request_timeout,
            **resolve_sampling_defaults(args.tokenizer_model or args.model,
                fallback=dict(temperature=1.0, top_p=.95, top_k=64, repetition_penalty=1.0)),
        )
        self.status = {'started_at': now(), 'stage': 'preparing', 'automatic_vm_shutdown': False}
        self.write_status()

    def write_status(self, **values):
        self.status.update(updated_at=now(), **values)
        save_json(self.out / 'status.json', self.status)

    def manifest(self):
        sources = ['generate_synthetic_data.py', '1a_make_synthetic_enrollee_prompts.py',
                   '1b_make_synthetic_negative_enrollee_prompts.py', '2_make_synthetic_notes_sharded.py',
                   'synthetic_event_prompts.py', 'remote_vllm_pool.py', 'vllm_reasoning_utils.py']
        value = dict(version='synthetic-seeds-patient-events-notes-v2', input_sha256=sha_file(self.args.input_csv),
                     source_sha256={name: sha_file(HERE / name) for name in sources},
                     sampling=asdict(self.sampling), tokenizer=self.args.tokenizer_model or self.args.model,
                     chat_template_sha256=hashlib.sha256(str(self.tokenizer.chat_template).encode()).hexdigest(),
                     seed=self.args.seed, enable_thinking=True, patients_per_space_per_class=5,
                     seed_max_tokens=self.args.seed_max_tokens,
                     event_max_tokens=self.args.event_max_tokens, note_max_tokens=self.args.note_max_tokens,
                     context_limit=self.args.max_model_len, spaces_per_note_chunk=self.args.spaces_per_note_chunk)
        path = self.out / 'manifest.json'
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError('Generation manifest changed; use a fresh output directory')
        save_json(path, value)
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    def build_payload(self, item, stage):
        messages = None
        if stage in ('seeds', 'patient_events'):
            module = self.positive if item['patient_class'] == 'positive' else self.negative
            criteria = item['this_space'] + '\nNo history of:\n' + item['trial_boilerplate_text']
            if stage == 'seeds':
                messages = module.build_seed_messages(criteria, rng=random.Random(self.args.seed + int(item['space_index'])))
                max_tokens = self.args.seed_max_tokens
            else:
                messages = module.build_patient_messages(criteria, item['seed_text'])
                max_tokens = self.args.event_max_tokens
            prompt = self.tokenizer.apply_chat_template(
                conversation=messages, add_generation_prompt=True, tokenize=False, enable_thinking=True)
        else:
            # Preserve the exact step-2 prompt. The larger limit prevents its
            # legacy head/tail truncation; reject an overlong context below.
            prompt = self.notes.build_prompts([item['masked_text']], self.tokenizer,
                                             max_ctx_tokens=self.args.max_model_len,
                                             model_name=self.args.tokenizer_model or self.args.model)[0]
            max_tokens = self.args.note_max_tokens
        length = len(self.tokenizer.encode(prompt, add_special_tokens=False))
        if length + max_tokens + 256 > self.args.max_model_len:
            raise ValueError(f'Full {stage} prompt and output budget exceed context for {item["key"]}')
        return dict(prompt=prompt, max_tokens=max_tokens, enable_thinking=True,
                    include_completion_metadata=True, include_raw_text=True, stage=stage,
                    key=item['key'], source_messages=messages, seed_text=item.get('seed_text'))

    def completion_work(self, directory):
        parser = resolve_parser_name(self.args.tokenizer_model or self.args.model, self.args.reasoning_parser)
        base_work = make_completion_work_fn(self.sampling, parser, self.tokenizer)

        async def work(client, payload):
            reasoning, answer, metadata = await base_work(client, payload)
            try:
                await asyncio.to_thread(validate_completion, answer, metadata,
                                        payload['stage'], payload.get('seed_text'))
            except OutputValidationError as exc:
                failure = dict(at=now(), stage=payload['stage'], key=payload['key'],
                               endpoint=str(client.base_url), error=str(exc),
                               prompt_sha256=hashlib.sha256(payload['prompt'].encode()).hexdigest(),
                               reasoning=reasoning, answer=answer, **metadata)
                await asyncio.to_thread(save_json, directory / ('invalid_' + str(uuid.uuid4()) + '.json'), failure)
                if payload.get('source_messages'):
                    # Pool retries the same payload. Supply the precise validation
                    # failure without accumulating prior patients or invalid drafts.
                    messages = payload['source_messages'] + [{'role': 'user', 'content':
                        'The previous response failed validation: ' + str(exc) + '\n'
                        'Regenerate the complete requested output from the original instructions. '
                        'Keep the exact assigned seed lines if a seed was supplied.'}]
                    payload['prompt'] = await asyncio.to_thread(
                        self.tokenizer.apply_chat_template, conversation=messages,
                        add_generation_prompt=True, tokenize=False, enable_thinking=True)
                raise
            if payload['stage'] == 'seeds':
                answer = '\n<new_patient>\n'.join(parse_seeds(answer))
            elif payload['stage'] == 'patient_events':
                answer = '\n'.join(f'<{kind}> {text}' for kind, text in
                                   parse_patient_history(answer, payload['seed_text']))
            return dict(reasoning=reasoning, answer=answer, raw_response=metadata['raw_text'],
                        finish_reason=metadata['finish_reason'], model=self.args.model,
                        endpoint=str(client.base_url), generated_at=now(),
                        prompt_sha256=hashlib.sha256(payload['prompt'].encode()).hexdigest())
        return work

    async def generate(self, items, stage, directory):
        directory.mkdir(parents=True, exist_ok=True)
        done = set()
        files = checkpoint_paths(directory)
        for path in files:
            keys = pq.read_table(path, columns=['key']).column('key').to_pylist()
            if done.intersection(keys) or len(set(keys)) != len(keys):
                raise ValueError('Duplicate generation checkpoint keys')
            done.update(keys)
        next_shard = max((int(path.stem.split('_')[-1]) for path in files), default=-1) + 1
        work = self.completion_work(directory)

        buffer, lock = [], asyncio.Lock()
        completed, expected = len(done), 0

        async def commit_buffer():
            nonlocal next_shard
            if buffer:
                frame = pd.DataFrame(buffer)
                await asyncio.to_thread(save_frame, directory / f'{next_shard // 1024:04d}' / f'shard_{next_shard:08d}.parquet', frame)
                next_shard += 1
                buffer.clear()
                self.write_status(stage=stage, stage_completed=completed, stage_expected=expected)

        async def flush():
            await finish_checkpoint(commit_buffer())

        registry = build_registry_from_args(self.args)
        iterator = iter(items)
        async with RequestSession(registry, work, max_attempts=self.args.max_attempts) as session:
            async def worker():
                nonlocal completed, expected
                while True:
                    # No await while advancing the shared iterator. Preparation
                    # is bounded by --pending-requests, not the full note count.
                    item = next(iterator, None)
                    if item is None:
                        return
                    expected += 1
                    if item['key'] in done:
                        continue
                    payload = await asyncio.to_thread(self.build_payload, item, stage)
                    result = await session.submit(payload)
                    if not isinstance(result, dict):
                        raise RuntimeError(f"Generation failed for {item['key']}: {result}")
                    async with lock:
                        buffer.append({**item, **result})
                        completed += 1
                        if len(buffer) >= self.args.results_per_shard:
                            await flush()

            tasks = [asyncio.create_task(worker()) for _ in range(self.args.pending_requests)]
            try:
                await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                await flush()
        if completed != expected:
            raise ValueError(f'Checkpoint coverage mismatch: {completed} != {expected}')
        return expected

    def seed_path(self, item):
        return self.out / 'patient_seeds' / f'{int(item["space_index"]) // 256:04d}' / (
            f'{item["patient_class"]}_{int(item["space_index"]):08d}.json')

    async def generate_patient_events(self, spaces):
        """Persist five seeds, then immediately expand their patients in one pool.

        Groups overlap: an unfinished seed batch never holds up another group's
        expansions. The pending-call semaphore also bounds prompt preparation.
        """
        directory = self.out / 'patient_events'
        directory.mkdir(parents=True, exist_ok=True)
        files = checkpoint_paths(directory)
        done = {}
        for path in files:
            for row in pq.read_table(path, columns=['key', 'seed_sha256']).to_pylist():
                if row['key'] in done:
                    raise ValueError('Duplicate patient generation checkpoint key')
                done[row['key']] = row['seed_sha256']
        expected_keys = {f'{kind}:{int(index)}:{patient}' for index in spaces.space_index
                         for kind in ('positive', 'negative') for patient in range(PATIENTS_PER_CLASS)}
        if set(done) - expected_keys:
            raise ValueError('Patient checkpoint contains keys outside the frozen input cohort')
        next_shard = max((int(path.stem.split('_')[-1]) for path in files), default=-1) + 1
        completed, seeds_completed = len(done), 0
        buffer, lock = [], asyncio.Lock()
        pending = asyncio.Semaphore(self.args.pending_requests)
        iterator = iter((dict(row, patient_class=kind, key=f'{kind}:{int(row["space_index"])}')
                         for row in spaces.to_dict('records') for kind in ('positive', 'negative')))

        def progress():
            self.write_status(stage='events', event_phase='seeds_and_individual_histories',
                              seeds_completed=seeds_completed, expected_seed_batches=2 * len(spaces),
                              stage_completed=completed - len(buffer), stage_expected=len(expected_keys),
                              completed_patient_histories=completed,
                              buffered_patient_histories=len(buffer))

        async def commit_buffer():
            nonlocal next_shard
            if buffer:
                await asyncio.to_thread(save_frame,
                    directory / f'{next_shard // 1024:04d}' / f'shard_{next_shard:08d}.parquet',
                    pd.DataFrame(buffer))
                next_shard += 1
                buffer.clear()
            progress()

        async def flush():
            await finish_checkpoint(commit_buffer())

        registry = build_registry_from_args(self.args)
        async with RequestSession(registry, self.completion_work(directory),
                                  max_attempts=self.args.max_attempts) as session:
            async def request(item, stage):
                async with pending:
                    payload = await asyncio.to_thread(self.build_payload, item, stage)
                    result = await session.submit(payload)
                if not isinstance(result, dict):
                    raise RuntimeError(f'Generation failed for {item["key"]}: {result}')
                return result

            async def expand(item, seed, patient):
                nonlocal completed
                key = f'{item["key"]}:{patient}'
                seed_hash = hashlib.sha256(seed.encode()).hexdigest()
                if key in done:
                    if done[key] != seed_hash:
                        raise ValueError(f'Saved patient seed changed: {key}')
                    return
                child = dict(item, key=key, seed_key=item['key'], patient_number=patient,
                             seed_text=seed, seed_sha256=seed_hash)
                result = await request(child, 'patient_events')
                # Avoid copying the full trial text into every patient checkpoint.
                record = {key: child[key] for key in ('key', 'seed_key', 'patient_number', 'seed_text',
                          'seed_sha256', 'space_index', 'patient_class', 'nct_id')}
                async with lock:
                    buffer.append({**record, **result})
                    completed += 1
                    if len(buffer) >= self.args.results_per_shard:
                        await flush()

            async def worker():
                nonlocal seeds_completed
                while (item := next(iterator, None)) is not None:
                    path = self.seed_path(item)
                    if path.exists():
                        saved = await asyncio.to_thread(lambda: json.loads(path.read_text()))
                        if any(saved[k] != item[k] for k in ('key', 'space_index', 'patient_class', 'nct_id')):
                            raise ValueError(f'Seed checkpoint identity mismatch: {path}')
                    else:
                        if any(f'{item["key"]}:{i}' in done for i in range(PATIENTS_PER_CLASS)):
                            raise ValueError(f'Missing parent seed checkpoint: {path}')
                        result = await request(item, 'seeds')
                        saved = {**{k: item[k] for k in ('key', 'space_index', 'patient_class', 'nct_id')}, **result}
                        path.parent.mkdir(parents=True, exist_ok=True)
                        await asyncio.to_thread(save_json, path, saved)
                    seeds = parse_seeds(saved['answer'])
                    seeds_completed += 1
                    if seeds_completed % 32 == 0:
                        progress()
                    # TaskGroup cancels siblings on interruption/failure before
                    # the shared pool closes; completed patients are flushed below.
                    async with asyncio.TaskGroup() as group:
                        for patient, seed in enumerate(seeds):
                            group.create_task(expand(item, seed, patient))

            tasks = [asyncio.create_task(worker()) for _ in range(self.args.pending_requests)]
            try:
                await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                await flush()
        if completed != len(expected_keys) or seeds_completed != 2 * len(spaces):
            raise ValueError('Seed/patient checkpoint coverage mismatch')

    def assembled_event_records(self, spaces):
        """Join five independently saved histories without mixing patient order."""
        locations = {}
        for path in checkpoint_paths(self.out / 'patient_events'):
            for key in pq.read_table(path, columns=['key']).column('key').to_pylist():
                if key in locations:
                    raise ValueError('Duplicate patient checkpoint key')
                locations[key] = path

        @lru_cache(maxsize=16)
        def read_shard(path):
            return {row['key']: row for row in pd.read_parquet(path).to_dict('records')}

        for row in spaces.to_dict('records'):
            for kind in ('positive', 'negative'):
                key = f'{kind}:{int(row["space_index"])}'
                patients = [read_shard(locations[f'{key}:{i}'])[f'{key}:{i}'] for i in range(PATIENTS_PER_CLASS)]
                provenance = [{k: patient[k] for k in ('key', 'patient_number', 'endpoint', 'model',
                              'prompt_sha256', 'seed_sha256', 'generated_at')} for patient in patients]
                yield dict(key=key, space_index=row['space_index'], patient_class=kind,
                           answer='\n<new_patient>\n'.join(p['answer'] for p in patients),
                           raw_response='\n<new_patient>\n'.join(p['raw_response'] for p in patients),
                           reasoning='\n<new_patient>\n'.join(p['reasoning'] for p in patients),
                           model=self.args.model, endpoint=json.dumps([p['endpoint'] for p in patients]),
                           generated_at=max(p['generated_at'] for p in patients),
                           prompt_sha256=hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest(),
                           patient_provenance_json=json.dumps(provenance),
                           seed_checkpoint=str(self.seed_path(dict(row, patient_class=kind))))

    def export_events(self, spaces, directory):
        frame = pd.DataFrame(self.assembled_event_records(spaces))
        expected = {f'{kind}:{int(index)}' for kind in ('positive', 'negative') for index in spaces.space_index}
        if len(frame) != len(expected) or set(frame.key) != expected:
            raise ValueError('Event coverage does not match every sampled space and both classes')
        for kind in ('positive', 'negative'):
            subset = frame.loc[frame.patient_class.eq(kind)].sort_values('space_index')
            result = spaces.merge(subset[['space_index', 'answer', 'raw_response', 'reasoning',
                                         'model', 'endpoint', 'generated_at', 'prompt_sha256',
                                         'patient_provenance_json', 'seed_checkpoint']],
                                  on='space_index', validate='one_to_one')
            result['criteria_index'] = result.space_index
            result = result.rename(columns={'answer': 'synthetic_patient_prompts',
                                             'raw_response': 'synthetic_patient_prompt_generation_reasoning'})
            path = self.out / f'trial_spaces_with_{kind}_prompts.csv'
            temporary = path.with_suffix('.csv.tmp')
            result.to_csv(temporary, index=False)
            temporary.replace(path)
        frame = spaces[['space_index', 'nct_id']].merge(frame, on='space_index', validate='one_to_many')
        return frame.sort_values(['space_index', 'patient_class']).reset_index(drop=True)

    def prepare_notes(self, spaces, events, identity):
        directory = self.out / 'note_inputs'
        directory.mkdir(exist_ok=True)
        records, row_offset = [], 0
        for start in range(0, len(spaces), self.args.spaces_per_note_chunk):
            selected = set(spaces.iloc[start:start + self.args.spaces_per_note_chunk].space_index)
            index = start // self.args.spaces_per_note_chunk
            path = directory / f'promptframe_{index:06d}.parquet'
            event_path = directory / f'events_{index:06d}.parquet'
            if path.exists() and event_path.exists():
                nrows = pq.ParquetFile(path).metadata.num_rows
            else:
                patients = event_rows(events.loc[events.space_index.isin(selected)].to_dict('records'), identity)
                save_frame(event_path, patients)
                promptframe = self.notes.create_masked_dataset(patients).reset_index(drop=True)
                promptframe['row_id'] = range(row_offset, row_offset + len(promptframe))
                promptframe['key'] = promptframe.row_id.map(lambda value: f'note:{value}')
                save_frame(path, promptframe)
                nrows = len(promptframe)
            records.append(dict(path=str(path), events=str(event_path), notes=nrows,
                                first_row_id=row_offset, last_row_id=row_offset + nrows - 1))
            row_offset += nrows
        save_json(self.out / 'note_input_manifest.json', dict(chunks=records, notes=row_offset))
        return records

    async def run(self):
        identity = self.manifest()
        spaces = pd.read_csv(self.args.input_csv)
        if spaces.space_index.tolist() != list(range(len(spaces))):
            raise ValueError('Expected contiguous 0c space_index values')
        spaces['trial_boilerplate_text'] = spaces.trial_boilerplate_text.fillna('')
        self.write_status(stage='events', sampled_trials=spaces.nct_id.nunique(), sampled_spaces=len(spaces),
                          expected_seed_batches=2 * len(spaces), expected_patients=10 * len(spaces))
        await self.generate_patient_events(spaces)
        events = await asyncio.to_thread(self.export_events, spaces, self.out / 'patient_events')
        if self.args.stop_after_events:
            self.write_status(stage='events_completed', all_outputs_local=True)
            return
        self.write_status(stage='preparing_notes')
        chunks = await asyncio.to_thread(self.prepare_notes, spaces, events, identity)
        del events
        events_path = self.out / 'all_synthetic_events.parquet'
        event_count = await asyncio.to_thread(compose_parquet, [chunk['events'] for chunk in chunks], events_path)
        note_count = sum(chunk['notes'] for chunk in chunks)
        self.write_status(stage='notes', expected_notes=note_count)
        # All chunks share one persistent pool. Reading at most 128 rows at a
        # time bounds memory even for millions of notes.
        def note_items():
            for chunk in chunks:
                for batch in pq.ParquetFile(chunk['path']).iter_batches(batch_size=128):
                    yield from batch.to_pylist()
        await self.generate(note_items(), 'notes', self.out / 'note_shards')
        self.write_status(stage='assembling')
        # Schema names consumed by patient summarization and training.
        paths = []
        for source in checkpoint_paths(self.out / 'note_shards'):
            target_dir = self.out / 'notes_for_training'
            target_dir.mkdir(exist_ok=True)
            target = target_dir / source.relative_to(self.out / 'note_shards')
            if not target.exists():
                frame = await asyncio.to_thread(pd.read_parquet, source)
                frame = frame.rename(columns={'answer': 'synthetic_note', 'raw_response': 'synth_note_reasoning_and_note'})
                await asyncio.to_thread(save_frame, target, frame)
            paths.append(target)
        notes_path = self.out / 'all_synthetic_notes.parquet'
        await asyncio.to_thread(compose_parquet, paths, notes_path, note_count)
        self.write_status(stage='completed', completed_at=now(), notes=note_count, events=event_count,
                          outputs=[str(notes_path), str(events_path)], all_outputs_local=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-csv', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--tokenizer-model')
    parser.add_argument('--download-dir')
    parser.add_argument('--reasoning-parser', default='auto')
    parser.add_argument('--seed-max-tokens', type=int, default=16000)
    parser.add_argument('--event-max-tokens', type=int, default=20000)
    parser.add_argument('--note-max-tokens', type=int, default=15000)
    parser.add_argument('--max-model-len', type=int, default=250000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--spaces-per-note-chunk', type=int, default=64)
    parser.add_argument('--pending-requests', type=int, default=1536)
    parser.add_argument('--stop-after-events', action='store_true', help='Save seeds and patient histories without starting notes')
    add_remote_cli_args(parser)
    parser.set_defaults(results_per_shard=128, max_attempts=16, request_timeout=1800)
    return parser.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'generation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model or args.model,
                                                  cache_dir=args.download_dir)
        generator = Generator(args, tokenizer)
        try:
            asyncio.run(generator.run())
        except Exception as exc:
            generator.write_status(stage='failed', error=repr(exc))
            raise


if __name__ == '__main__':
    main()
