#!/usr/bin/env python3
"""One-off, patient-held-out ColBERT smoke test from live labels.

Run directly with Python; the parent freezes a read-only label snapshot and
launches two DDP workers. No LLM calls, server management or production writes.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import datetime, timezone
import fcntl
import hashlib
from itertools import islice
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SIBLING = ROOT.parent / 'matchminer-ai-inference-kehl' / 'src'
if SIBLING.exists():
    sys.path.insert(0, str(SIBLING))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from colbert_experiments.complete_pass_batches import CompletePassBatchSampler
from colbert_experiments.checkpointing import PauseRequest, save_checkpoint
from colbert_experiments.contracts import DIRECTIONS, contract_hash, judge_messages, prompt_sources
from colbert_experiments.data import digest, load_corpus, read_jsonl, write_json, write_jsonl
from colbert_experiments.model import BASE_MODEL, ColBERT, maxsim
from colbert_experiments.ordinal_ranking import cached_ranking_backward, ranking_loss
from colbert_experiments import regression as regression
from colbert_experiments.export_completed_scores import validate_score

DEFAULT_RUN = ROOT.parent / 'data/no_phi/synthetic_cohorts_5000_gemma4_31b_20261024/post_notes_colbert_20261025'
VERSION = 'completed-pairs-patient-holdout-smoke-v1'


def sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def source_split(patient):
    # Production's effective split is train for everybody. Never use it when
    # original provenance exists, and never infer a split from the label score.
    split = patient.get('source_split', patient['split'])
    if split not in ('train', 'val', 'test'):
        raise ValueError('Patient has an unknown source split')
    return split


def candidate_sample(path, patients, capacities, seed):
    """Bounded reservoirs from the full file, not the first processed patients."""
    reservoirs = {s: [] for s in capacities}
    seen = Counter()
    rng = {s: random.Random(f'{seed}:{s}:candidates') for s in capacities}
    hasher = hashlib.sha256()
    before = path.stat()
    with path.open('rb') as stream:
        for number, line in enumerate(stream, 1):
            hasher.update(line)
            row = json.loads(line)
            split = source_split(patients[row['patient_id']])
            if split not in ('train', 'val'):
                continue
            seen[split] += 1
            if split not in capacities:
                continue
            pair = {k: row[k] for k in ('patient_id', 'cohort_id', 'split')}
            reservoir = reservoirs[split]
            if len(reservoir) < capacities[split]:
                reservoir.append(pair)
            else:
                index = rng[split].randrange(seen[split])
                if index < capacities[split]:
                    reservoir[index] = pair
            if number % 250000 == 0:
                print(f'[snapshot] scanned {number:,} candidate rows', flush=True)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Candidate file changed during snapshot selection; retry')
    for split in reservoirs:
        rng[split].shuffle(reservoirs[split])
    return reservoirs, dict(candidates_sha256=hasher.hexdigest(), available_candidates=dict(seen))


def all_candidates(path, patients, split):
    """Stream every candidate in one original patient partition without a cap."""
    before = path.stat()
    with path.open('rb') as stream:
        for line in stream:
            row = json.loads(line)
            if source_split(patients[row['patient_id']]) == split:
                yield {k: row[k] for k in ('patient_id', 'cohort_id', 'split')}
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Candidate file changed during the complete partition scan; retry')


def response_keys(directory):
    """Index published filenames to avoid millions of missing-file NFS lookups.

    No response content is loaded here. completed_score still checks the fixed
    cutoff and complete question/prompt identity before accepting each pair.
    """
    keys = set()
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.name.endswith('.json'):
                keys.add(entry.name[:-5])
            elif len(entry.name) == 2 and entry.is_dir():
                with os.scandir(entry.path) as children:
                    keys.update(child.name[:-5] for child in children if child.name.endswith('.json'))
    return keys


def completed_score(pair, *, responses, patients, cohorts, cutoff_ns, question_contract, sources):
    """Only a negative gate or a positive gate with all four answers is complete."""
    provenance = {}

    def read_question(task):
        key = digest([pair['patient_id'], pair['cohort_id'], task])
        path = responses / key[:2] / (key + '.json')
        legacy = responses / (key + '.json')
        try:
            stream = path.open('rb')
        except FileNotFoundError:
            try:
                path, stream = legacy, legacy.open('rb')
            except FileNotFoundError:
                return None
        with stream:
            if os.fstat(stream.fileno()).st_mtime_ns > cutoff_ns:
                return None
            raw = stream.read()
        row = json.loads(raw)
        messages = judge_messages(task, patients[pair['patient_id']]['text'],
                                  cohorts[pair['cohort_id']]['text'], sources=sources)
        if (any(row[k] != pair[k] for k in ('patient_id', 'cohort_id', 'split'))
                or row.get('task') != task or row.get('key') != key
                or row.get('contract') != question_contract
                or row.get('prompt_sha256') != digest(messages)
                or row.get('finish_reason') != 'stop'):
            raise ValueError(f'Checkpoint identity, prompt or completion mismatch: {path}')
        provenance[task] = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest())
        return row

    gate = read_question(regression.GATE)
    if gate is None:
        return None
    judgments = [gate]
    if gate['answer'] == 'yes':
        for task in regression.SPECIFICITY:
            row = read_question(task)
            if row is None:
                return None
            judgments.append(row)
    score = regression.aggregate_pair(pair, judgments, question_contract=question_contract)
    return dict(**score, patient_source_split=source_split(patients[pair['patient_id']]),
                source_checkpoints=provenance)


def compact_snapshot(args, patients, cohorts, teacher):
    """Read one sequential aggregate file, with no question-checkpoint inspection."""
    archive = getattr(args, 'score_archive', None) or args.source_run/'labels/completed_scores'
    pointer = archive/'manifest.json'
    final_scores = args.source_run/'labels/scores.jsonl'
    exported = None
    if not getattr(args, 'score_archive', None) and final_scores.exists():
        path = final_scores
    elif pointer.exists():
        exported = json.loads(pointer.read_text())
        if (exported.get('format') != 'completed-gated-scores-v1'
                or exported.get('teacher_manifest') != teacher
                or Path(exported['scores_file']).name != exported['scores_file']):
            raise ValueError('Compact score archive has a different teacher/corpus contract')
        path = archive/exported['scores_file']
    elif getattr(args, 'score_archive', None):
        raise ValueError('Requested compact archive has not been published: ' + str(pointer))
    else:
        return None
    print(f'[snapshot] reading compact scores sequentially: {path}', flush=True)
    pmap, cmap = {p['id']: p for p in patients}, {c['id']: c for c in cohorts}
    targets = dict(train=args.train_pairs, val=args.val_pairs)
    selected = {s: [] for s in targets}
    counts = Counter()
    seen = set()
    rng = {s: random.Random(f'{args.seed}:{s}:completed_scores') for s in targets}
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        source_mtime = os.fstat(stream.fileno()).st_mtime
        for number, line in enumerate(stream, 1):
            hasher.update(line)
            row = json.loads(line)
            validate_score(row, pmap, cmap, teacher['contract'], teacher['aggregation_contract'])
            key = row['patient_id'], row['cohort_id']
            if key in seen:
                raise ValueError('Compact score archive repeats a patient/cohort pair')
            seen.add(key)
            split = source_split(pmap[row['patient_id']])
            counts[split] += 1
            if split in targets:
                row['patient_source_split'] = split
                target, reservoir = targets[split], selected[split]
                if not target or len(reservoir) < target:
                    reservoir.append(row)
                else:
                    index = rng[split].randrange(counts[split])
                    if index < target:
                        reservoir[index] = row
            if number % 100000 == 0:
                print(f'[snapshot] {number:,} compact completed-score rows loaded', flush=True)
    if exported and (hasher.hexdigest() != exported['scores_sha256']
                     or len(seen) != exported['counts']['completed_pairs']):
        raise ValueError('Compact score archive checksum or row count differs from its manifest')
    report = {}
    for split, rows in selected.items():
        if len({row['score'] > 0 for row in rows}) < 2:
            raise ValueError(f'{split} needs complete pairs of both binary classes for meaningful AUROC')
        report[split] = dict(pairs=len(rows), requested=targets[split] or 'all',
            selection='all_completed_candidate_pairs' if not targets[split] else 'completed_pair_reservoir',
            available_completed=counts[split], patients=len({row['patient_id'] for row in rows}),
            score_distribution=dict(sorted(Counter(str(row['score']) for row in rows).items())))
        print('[snapshot] ' + split + ' ' + json.dumps(report[split]), flush=True)
    cutoff = exported['cutoff_utc'] if exported else datetime.fromtimestamp(source_mtime, timezone.utc).isoformat()
    print(f'[snapshot] completed-score cutoff={cutoff}; individual response files inspected=0', flush=True)
    return dict(scores=selected['train'] + selected['val'], counts=report, cutoff_utc=cutoff,
        report=dict(source_score_file=str(path), source_scores_sha256=hasher.hexdigest(),
                    available_completed_by_patient_split=dict(counts), checkpoint_files_inspected=0),
        sampling='All completed original-training pairs; uniform reservoir for capped partitions. '
                 'Compact aggregate contracts verified without reopening LLM response files.')


def freeze_snapshot(snapshot, settings, teacher, patients, cohorts, scores, counts, report, cutoff_utc, sampling):
    patient_ids, cohort_ids = {s['patient_id'] for s in scores}, {s['cohort_id'] for s in scores}
    write_jsonl(snapshot / 'patients.jsonl', [p for p in patients if p['id'] in patient_ids])
    write_jsonl(snapshot / 'cohorts.jsonl', [c for c in cohorts if c['id'] in cohort_ids])
    write_jsonl(snapshot / 'scores.jsonl', scores)
    manifest = dict(settings=settings, cutoff_utc=cutoff_utc,
        teacher_manifest=teacher, counts=counts, **report,
        split_policy='Optimizer: original train patients only; validation: original val patients; test patients excluded. '
                     'Cohorts may be shared across patient partitions.', sampling=sampling,
        files={name: sha256(snapshot / name) for name in ('patients.jsonl', 'cohorts.jsonl', 'scores.jsonl')})
    write_json(snapshot / 'manifest.json', manifest)
    return manifest


def prepare_snapshot(args):
    if args.train_pairs < 0 or args.val_pairs < 1:
        raise ValueError('train-pairs must be zero (all) or positive; val-pairs must be positive')
    snapshot = args.output / 'snapshot'
    manifest_path = snapshot / 'manifest.json'
    settings = dict(version=VERSION, source_run=str(args.source_run.resolve()),
                    train_pairs=args.train_pairs, val_pairs=args.val_pairs,
                    candidate_multiplier=args.candidate_multiplier, seed=args.seed,
                    question_contract=contract_hash())
    if getattr(args, 'score_archive', None):
        settings['score_archive'] = str(args.score_archive.resolve())
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['settings'] != settings:
            raise ValueError('Snapshot settings changed; use a new --output directory')
        if any(sha256(snapshot / name) != expected for name, expected in manifest['files'].items()):
            raise ValueError('Frozen smoke-test snapshot changed')
        print('[snapshot] reusing fixed completed-pair snapshot', flush=True)
        return manifest
    corpus = args.source_run / 'corpus_all'
    labels = args.source_run / 'labels'
    cutoff_ns = time.time_ns()
    patients, cohorts = load_corpus(corpus)
    teacher = json.loads((labels / 'labeling_manifest.json').read_text())
    if any(teacher.get(k) != expected for k, expected in dict(contract=settings['question_contract'],
            aggregation_contract=regression.score_contract(), patients=digest(patients), cohorts=digest(cohorts)).items()):
        raise ValueError('Teacher contract or complete corpus text differs from this experiment')
    compact = compact_snapshot(args, patients, cohorts, teacher)
    if compact is not None:
        return freeze_snapshot(snapshot, settings, teacher, patients, cohorts, **compact)
    if not getattr(args, 'inspect_checkpoints', False):
        raise ValueError('No compact completed-score archive is available yet. Run '
            'colbert_experiments/export_completed_scores.py --source-run SOURCE_RUN on the producing host, '
            'then rerun this command. --inspect-checkpoints explicitly enables the slower per-response scan.')
    pmap, cmap = {p['id']: p for p in patients}, {c['id']: c for c in cohorts}
    targets = dict(train=args.train_pairs, val=args.val_pairs)
    candidates = args.source_run / 'mining_all/candidates.jsonl'
    samples, report = candidate_sample(candidates, pmap,
        {split: count * args.candidate_multiplier for split, count in targets.items() if count}, args.seed)
    sources = prompt_sources()
    published = None
    if args.train_pairs == 0:
        print('[snapshot] indexing published responses for an uncapped training-partition scan', flush=True)
        published = response_keys(labels / 'responses')

    def load(pair):
        if published is not None and digest([pair['patient_id'], pair['cohort_id'], regression.GATE]) not in published:
            return None
        return completed_score(pair, responses=labels / 'responses', patients=pmap, cohorts=cmap,
            cutoff_ns=cutoff_ns, question_contract=settings['question_contract'], sources=sources)

    scores, counts = [], {}
    with ThreadPoolExecutor(max_workers=args.read_workers) as pool:
        for split, target in targets.items():
            checked, missing, selected, seen = 0, 0, [], set()
            pairs = iter(samples[split]) if target else all_candidates(candidates, pmap, split)
            progress_at = time.monotonic()
            while batch := list(islice(pairs, 64)):
                for pair, score in zip(batch, pool.map(load, batch)):
                    checked += 1
                    if score is None:
                        missing += 1
                        continue
                    key = (pair['patient_id'], pair['cohort_id'])
                    if key not in seen and (not target or len(selected) < target):
                        selected.append(score)
                        seen.add(key)
                if target and len(selected) >= target:
                    break
                if time.monotonic() - progress_at >= 10:
                    limit = f'{target:,}' if target else 'all'
                    print(f'[snapshot] {split}: {len(selected):,}/{limit} complete pairs; {checked:,} inspected', flush=True)
                    progress_at = time.monotonic()
            if not selected:
                raise ValueError(f'No complete {split} pairs; allow more labels or increase --candidate-multiplier')
            if len({row['score'] > 0 for row in selected}) < 2:
                raise ValueError(f'{split} snapshot has only one binary class; request more pairs/labels for meaningful AUROC')
            counts[split] = dict(pairs=len(selected), requested=target if target else 'all', inspected=checked,
                selection='capped_completed_candidate_sample' if target else 'all_completed_candidate_pairs',
                incomplete_or_after_cutoff=missing, patients=len({s['patient_id'] for s in selected}),
                score_distribution=dict(sorted(Counter(str(s['score']) for s in selected).items())))
            print('[snapshot] ' + split + ' ' + json.dumps(counts[split]), flush=True)
            scores.extend(selected)
    return freeze_snapshot(snapshot, settings, teacher, patients, cohorts, scores, counts, report,
        datetime.fromtimestamp(cutoff_ns / 1e9, timezone.utc).isoformat(),
        'Uncapped partitions scan every candidate; capped partitions use uniform candidate reservoirs. '
        'Only pairs complete at the fixed cutoff are included; missing specificity answers are never zero labels.')


def split_scores(patients, scores):
    pmap = {p['id']: p for p in patients}
    result = {s: [row for row in scores if source_split(pmap[row['patient_id']]) == s] for s in ('train', 'val')}
    if any(source_split(pmap[row['patient_id']]) == 'test' for row in scores):
        raise ValueError('Test patients must not enter this smoke-test snapshot')
    for split, rows in result.items():
        if not rows or any(row['patient_source_split'] != split for row in rows):
            raise ValueError('Missing or inconsistent original patient split')
    if {r['patient_id'] for r in result['train']} & {r['patient_id'] for r in result['val']}:
        raise ValueError('Train/validation patient leakage')
    return result


def fixed_eval_pairs(rows, count, seed):
    # Approximately prevalence-preserving binary stratification guarantees both
    # classes for AUROC without selecting pairs on the student's predictions.
    rng = random.Random(seed)
    buckets = {b: [row for row in rows if (row['score'] > 0) == b] for b in (False, True)}
    count = min(count, len(rows))
    if count < 2 or not all(buckets.values()):
        raise ValueError('Evaluation needs at least one zero and one nonzero pair')
    positive = max(1, min(count - 1, round(count * len(buckets[True]) / len(rows)), len(buckets[True])))
    negative = min(count - positive, len(buckets[False]))
    positive = count - negative
    result = rng.sample(buckets[False], negative) + rng.sample(buckets[True], positive)
    rng.shuffle(result)
    return result


def prediction_metrics(rows):
    from sklearn.metrics import roc_auc_score
    if not rows or any(not math.isfinite(row['maxsim']) for row in rows):
        raise ValueError('Evaluation requires finite continuous predictions')
    labels = [row['score'] > 0 for row in rows]
    return dict(pairs=len(rows), positives=sum(labels), negatives=len(labels) - sum(labels),
        loss=sum((row['maxsim'] - row['score'] / 5) ** 2 for row in rows) / len(rows),
        auroc=float(roc_auc_score(labels, [row['maxsim'] for row in rows])) if len(set(labels)) == 2 else None)


def objective_metrics(rows, *, objective='mse', scale=20., contrastive_weight=1.):
    result = prediction_metrics(rows)
    if objective != 'mse':
        loss, components = ranking_loss(torch.tensor([r['maxsim'] for r in rows]),
            [r['score'] for r in rows], [r['direction'] for r in rows],
            [r['patient_id'] if r['direction'] == 'patient_to_cohort' else r['cohort_id'] for r in rows],
            objective=objective, scale=scale, contrastive_weight=contrastive_weight)
        result.update(mse_diagnostic=result['loss'], loss=float(loss), objective=objective,
                      **{k: float(v) if isinstance(v, torch.Tensor) else v for k, v in components.items()})
    return result


@torch.inference_mode()
def evaluate(model, pairs, patients, cohorts, directions, *, rank, world, batch_size,
             objective='mse', scale=20., contrastive_weight=1.):
    was_training = model.training
    model.eval()
    rows = []
    try:
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=model.device.type == 'cuda'):
            local = pairs[rank::world]
            for direction in directions:
                for start in range(0, len(local), batch_size):
                    batch = local[start:start + batch_size]
                    texts, roles = [], []
                    for pair in batch:
                        patient, cohort = patients[pair['patient_id']]['text'], cohorts[pair['cohort_id']]['text']
                        query, document = (patient, cohort) if direction == 'patient_to_cohort' else (cohort, patient)
                        texts.extend([regression.regression_query(direction, query), document])
                        roles.extend([True, False])
                    encoded = model.encode_texts(texts, is_queries=roles)
                    scores = torch.stack([maxsim(encoded[i], encoded[i+1]) for i in range(0, len(encoded), 2)]).float().cpu().tolist()
                    rows.extend(dict(patient_id=p['patient_id'], cohort_id=p['cohort_id'], direction=direction,
                                     score=p['score'], maxsim=s) for p, s in zip(batch, scores))
        if world > 1:
            gathered = [None] * world
            dist.all_gather_object(gathered, rows)
            rows = [row for part in gathered for row in part]
        settings = dict(objective=objective, scale=scale, contrastive_weight=contrastive_weight)
        return dict(**objective_metrics(rows, **settings), by_direction={direction: objective_metrics(
            [row for row in rows if row['direction'] == direction], **settings) for direction in directions}), rows
    finally:
        model.train(was_training)


def train_worker(args):
    rank, world, local_rank = (int(os.environ.get(k, default)) for k, default in
                              [('RANK', '0'), ('WORLD_SIZE', '1'), ('LOCAL_RANK', '0')])
    device = torch.device('cpu' if args.device == 'cpu' else f'cuda:{local_rank}')
    if device.type == 'cuda':
        torch.cuda.set_device(local_rank)
    if world > 1:
        dist.init_process_group('gloo' if device.type == 'cpu' else 'nccl',
                                **(dict(device_id=device) if device.type == 'cuda' else {}))
    try:
        with PauseRequest(stop_at=args.stop_at, pause_file=args.output / 'PAUSE') as pause:
            return fit(args, device, rank, world, pause)
    finally:
        if world > 1:
            dist.destroy_process_group()


def fit(args, device, rank, world, pause):
    snapshot, output = args.output / 'snapshot', args.output / 'fit'
    output.mkdir(exist_ok=True)
    patients, cohorts = load_corpus(snapshot)
    scores = read_jsonl(snapshot / 'scores.jsonl')
    partitions = split_scores(patients, scores)
    directions = list(DIRECTIONS) if args.direction == 'both' else [args.direction]
    # Validate the production score contract, then consume complete epochs of
    # original-train patient pairs rather than sampling a fixed update budget.
    groups, _ = regression.build_groups(patients, cohorts, partitions['train'], split=['train', 'val', 'test'])
    groups = [g for g in groups if g['direction'] in directions]
    sampler = CompletePassBatchSampler(groups, documents_per_query=args.documents_per_query,
        query_groups_per_update=world * args.accumulation, seed=args.seed)
    total_steps = args.steps if args.steps is not None else args.epochs * sampler.updates_per_epoch
    if rank == 0:
        print(f'[plan] {len(partitions["train"]):,} training pairs; {sampler.pairs:,} directional pairs/epoch; '
              f'{sampler.updates_per_epoch:,} updates/epoch; {total_steps:,} total updates '
              f'({str(args.epochs) + " complete epochs" if args.steps is None else "explicit --steps override"})', flush=True)
    probes = {s: fixed_eval_pairs(rows, args.eval_pairs, f'{args.seed}:{s}') for s, rows in partitions.items()}
    pmap, cmap = {p['id']: p for p in patients}, {c['id']: c for c in cohorts}
    torch.manual_seed(args.seed)
    model = ColBERT(args.model, query_length=args.context, document_length=args.context,
                    cache_dir=args.cache_dir, checkpointing=True).to(device)
    identity = dict(version=VERSION, snapshot=sha256(snapshot / 'manifest.json'), base=model.fingerprint(),
        steps=total_steps, epochs=args.epochs if args.steps is None else None,
        world_size=world, accumulation=args.accumulation, documents_per_query=args.documents_per_query,
        learning_rate=args.learning_rate, seed=args.seed, directions=directions, eval_every=args.eval_every,
        eval_pairs=digest(probes), eval_batch_size=args.eval_batch_size,
        objective='MSE(mean_MaxSim, score/5)', auroc='raw mean_MaxSim predicts score > 0; zero is negative',
        batch_sampling=sampler.report(), loss_weighting='equal labeled pairs within each update', device_type=device.type)
    # Keep the original default identity exactly resumable for existing MSE runs.
    if args.loss != 'mse':
        identity.update(objective=args.loss, ranking_scale=args.ranking_scale,
            contrastive_weight=args.contrastive_weight,
            loss_weighting='mean CoSENT over directions with unequal labels; mean contrast over queries with known positives and zeros',
            effective_batch_comparisons='two-pass scalar-score gradient cache with seeded forward replay')
    if args.lr_scheduler != 'constant':
        identity.update(lr_scheduler=args.lr_scheduler, warmup_ratio=args.warmup_ratio)
    manifest_path = output / 'training_manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != identity:
        raise ValueError('Training settings changed; use a new output directory')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.01)
    scheduler = None
    if args.lr_scheduler == 'linear':
        from transformers import get_linear_schedule_with_warmup
        scheduler = get_linear_schedule_with_warmup(optimizer,
            num_warmup_steps=math.ceil(total_steps * args.warmup_ratio), num_training_steps=total_steps)
    latest = output / 'latest.json'
    step, history, evaluations = 0, [], []
    if latest.exists():
        state = torch.load(output / json.loads(latest.read_text())['path'], map_location='cpu', weights_only=True)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(state['scheduler'])
        step, history, evaluations = state['step'], state['history'], state['evaluations']
        del state
    if rank == 0:
        write_json(manifest_path, identity)
        write_json(output / 'eval_pairs.json', probes)
    wrapped = DistributedDataParallel(model, device_ids=[device.index] if device.type == 'cuda' else None) if world > 1 else model
    wrapped.train()
    started = time.monotonic()

    def checkpoint(current):
        if rank == 0:
            save_checkpoint(output, dict(step=current, model=model.state_dict(), optimizer=optimizer.state_dict(),
                history=history, evaluations=evaluations,
                **(dict(scheduler=scheduler.state_dict()) if scheduler is not None else {})), limit=2)
        if world > 1:
            dist.barrier()

    def should_pause(current):
        if not pause.requested(device, world):
            return False
        optimizer.zero_grad(set_to_none=True)
        checkpoint(current)
        if rank == 0:
            write_json(output / 'status.json', dict(state='paused', step=current, resumable=True))
        return True

    def report(current):
        entry = dict(step=current)
        for split, pairs in probes.items():
            metrics, predictions = evaluate(model, pairs, pmap, cmap, directions, rank=rank, world=world,
                batch_size=args.eval_batch_size, objective=args.loss, scale=args.ranking_scale,
                contrastive_weight=args.contrastive_weight)
            entry[split] = metrics
            if rank == 0:
                write_json(output / 'predictions' / f'{split}_{current:06d}.json', predictions)
        evaluations.append(entry)
        if rank == 0:
            aucs = ' '.join(f'val_auroc_{d}={entry["val"]["by_direction"][d]["auroc"]:.4f}' for d in directions)
            print(f'[eval {current:05d}] train_loss={entry["train"]["loss"]:.6f} '
                  f'val_loss={entry["val"]["loss"]:.6f} {aucs}'
                  + (f' objective={args.loss} val_mse_diagnostic={entry["val"]["mse_diagnostic"]:.6f}'
                     if args.loss != 'mse' else ''), flush=True)
            write_json(output / 'evaluation_history.json', evaluations)

    if should_pause(step):
        return 'paused'
    if not evaluations or evaluations[-1]['step'] != step:
        report(step)
    for current in range(step, total_steps):
        optimizer.zero_grad(set_to_none=True)
        batch = sampler.batch(current)
        pair_count = sum(len(indices) for _, indices in batch)
        microbatches = math.ceil(len(batch) / world)
        epoch, epoch_update = divmod(current, sampler.updates_per_epoch)
        loss_sum = torch.zeros((), device=device)
        ranking_stats = {}
        if args.loss != 'mse':
            result = cached_ranking_backward(model, wrapped, batch, device=device, rank=rank, world=world,
                seed_for_micro=lambda micro: args.seed + (current * args.accumulation + micro) * world + rank,
                objective=args.loss, scale=args.ranking_scale, contrastive_weight=args.contrastive_weight,
                should_pause=lambda: should_pause(current))
            if result is None:
                return 'paused'
            loss_sum = result.pop('loss')
            ranking_stats = {k: float(v) if isinstance(v, torch.Tensor) else v for k, v in result.items()}
        for micro in range(microbatches if args.loss == 'mse' else 0):
            if should_pause(current):
                return 'paused'
            index = micro * world + rank
            # All ranks execute the same number of collectives. A missing tail
            # slot uses a zero-weight forward, contributing no duplicate label.
            real = index < len(batch)
            group, indices = batch[index] if real else batch[0]
            weight = len(indices) if real else 0
            torch.manual_seed(args.seed + (current * args.accumulation + micro) * world + rank)
            sync = wrapped.no_sync() if world > 1 and micro + 1 < microbatches else nullcontext()
            with sync, torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                prediction = wrapped(group['query'], [group['documents'][i]['text'] for i in indices])
                loss = regression.regression_loss(prediction, [group['labels'][i] for i in indices])
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite smoke-test training loss')
                # DDP averages gradients across ranks; compensate so unequal
                # document counts and zero-weight padding preserve the global
                # mean squared error over actual pairs in this update.
                (loss * (world * weight / pair_count)).backward()
            loss_sum += loss.detach() * weight
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        learning_rate = optimizer.param_groups[0]['lr']
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if args.loss == 'mse':
            if world > 1:
                dist.all_reduce(loss_sum)
            loss_sum /= pair_count
        if rank == 0:
            entry = dict(step=current+1, train_batch_loss=float(loss_sum), gradient_norm=float(norm),
                epoch=epoch+1, epoch_update=epoch_update+1, directional_pairs=pair_count,
                score_counts=dict(sorted(Counter(str(g['labels'][i]) for g, indices in batch for i in indices).items())),
                learning_rate=learning_rate, **ranking_stats, elapsed_seconds=time.monotonic()-started)
            history.append(entry)
            write_json(output / 'status.json', dict(state='training', **entry))
            print(f'[train {current+1:05d}/{total_steps} epoch={epoch+1}] '
                  f'train_batch_loss={entry["train_batch_loss"]:.6f} pairs={pair_count}'
                  + (f' objective={args.loss} lr={learning_rate:.3g}' if args.loss != 'mse' else ''), flush=True)
        if should_pause(current+1):
            return 'paused'
        epoch_end = epoch_update + 1 == sampler.updates_per_epoch
        if (current+1) % args.eval_every == 0 or epoch_end or current+1 == total_steps:
            report(current+1)
        if (current+1) % args.save_steps == 0 or epoch_end or current+1 == total_steps:
            checkpoint(current+1)
    if rank == 0:
        model.save(args.output / 'model')
        scoring = dict(contract=regression.score_contract(), score='mean_maxsim', prediction_link='identity',
            output_scale=5, query_instruction=regression.INSTRUCTION, semantics='gated score; smoke-test artifact')
        if args.loss != 'mse':
            scoring.pop('output_scale')
            scoring.update(objective=args.loss, calibrated=False,
                semantics='Ordinal retrieval score learned from gated labels; not a probability or predicted 0-5 score')
        write_json(args.output / 'model/regression_scoring.json', scoring)
        write_json(output / 'metrics.json', dict(history=history, evaluations=evaluations))
        write_json(output / 'status.json', dict(state='complete', steps=total_steps,
            completed_epochs=total_steps // sampler.updates_per_epoch,
            directional_pairs_seen=sum(row['directional_pairs'] for row in history)))
    if world > 1:
        dist.barrier()
    return 'complete'


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', type=Path, default=DEFAULT_RUN)
    p.add_argument('--output', type=Path, required=True, help='New directory separate from the production run; repeat to resume')
    p.add_argument('--gpus', default='0,1', help='Physical CUDA devices on this machine, default two A6000s')
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda', help='CPU is for tiny mechanical tests only')
    p.add_argument('--prepare-only', action='store_true', help='Freeze a snapshot without loading a model or using GPUs')
    p.add_argument('--score-archive', type=Path, help='Compact export directory; default SOURCE_RUN/labels/completed_scores')
    p.add_argument('--inspect-checkpoints', action='store_true',
                   help='Allow slow per-response reconstruction only when no compact score file is available')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--train-pairs', type=int, default=0,
                   help='Maximum completed train-patient pairs; 0 (default) includes all at the snapshot cutoff')
    p.add_argument('--val-pairs', type=int, default=2000)
    p.add_argument('--candidate-multiplier', type=int, default=12,
                   help='Candidates inspected per requested capped pair; applies to validation and explicit training caps')
    p.add_argument('--read-workers', type=int, default=4)
    p.add_argument('--model', default=BASE_MODEL)
    p.add_argument('--cache-dir', help='Local model cache on this machine; otherwise use Hugging Face defaults')
    p.add_argument('--context', type=int, default=8192)
    p.add_argument('--epochs', type=int, default=3,
                   help='Complete passes over all training pairs in each selected direction; default 3')
    p.add_argument('--steps', type=int,
                   help='Optional explicit update-count override; by default train all --epochs with no fixed step cap')
    p.add_argument('--accumulation', type=int, default=32)
    p.add_argument('--documents-per-query', type=int, default=2)
    p.add_argument('--learning-rate', type=float, default=1e-5)
    p.add_argument('--loss', choices=['mse', 'cosent', 'legacy_ranking'], default='mse',
                   help='legacy_ranking adapts CoSENT + contrastive learning, using only explicitly labeled negatives')
    p.add_argument('--ranking-scale', type=float, default=20., help='Inverse temperature for ranking objectives')
    p.add_argument('--contrastive-weight', type=float, default=1., help='Known-negative contrast weight in legacy_ranking')
    p.add_argument('--lr-scheduler', choices=['constant', 'linear'], default='constant')
    p.add_argument('--warmup-ratio', type=float, default=.01, help='Warmup fraction for linear scheduler')
    p.add_argument('--eval-every', type=int, default=5)
    p.add_argument('--eval-pairs', type=int, default=256, help='Fixed unique pairs per split, evaluated in each direction')
    p.add_argument('--eval-batch-size', type=int, default=2, help='Pairs encoded together; two texts per pair')
    p.add_argument('--direction', choices=['both', *DIRECTIONS], default='both')
    p.add_argument('--save-steps', type=int, default=10)
    p.add_argument('--stop-at', help='Optional ISO timestamp with timezone offset for a checkpointed pause')
    p.add_argument('--seed', type=int, default=42)
    return p


def main():
    args = parser().parse_args()
    args.source_run, args.output = args.source_run.resolve(), args.output.resolve()
    if args.score_archive:
        args.score_archive = args.score_archive.resolve()
    if args.output.is_relative_to(args.source_run) or args.source_run.is_relative_to(args.output):
        raise ValueError('--output must be separate from the production run')
    if args.train_pairs < 0:
        raise ValueError('train-pairs must be zero (all) or positive')
    if args.steps is not None and args.steps < 1:
        raise ValueError('steps must be positive when explicitly supplied')
    for key in ('val_pairs', 'candidate_multiplier', 'read_workers', 'epochs', 'accumulation',
                'documents_per_query', 'eval_every', 'eval_batch_size', 'save_steps'):
        if getattr(args, key) < 1:
            raise ValueError(key + ' must be positive')
    if args.eval_pairs < 2:
        raise ValueError('eval-pairs must be at least two for binary AUROC')
    if (not all(math.isfinite(v) for v in (args.learning_rate, args.ranking_scale,
                                         args.contrastive_weight, args.warmup_ratio))
            or min(args.learning_rate, args.ranking_scale) <= 0 or args.contrastive_weight < 0
            or not 0 <= args.warmup_ratio < 1):
        raise ValueError('Use positive LR/scale, nonnegative contrast weight and warmup ratio in [0, 1)')
    if args.worker:
        if train_worker(args) == 'paused':
            raise SystemExit(75)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'smoke.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare_snapshot(args)
        if args.prepare_only:
            return
        gpus = args.gpus.split(',')
        if any(not g.isdigit() for g in gpus) or len(set(gpus)) != len(gpus):
            raise ValueError('--gpus must be a comma-separated list of distinct GPU indices')
        # The same script becomes the worker entrypoint; no shell, hostname or
        # Camus-specific executable path is used on the second machine.
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                   '--nproc_per_node=' + str(len(gpus)), str(Path(__file__).resolve()), *sys.argv[1:],
                   '--source-run', str(args.source_run), '--output', str(args.output), '--worker']
        if args.cache_dir:
            command += ['--cache-dir', str(Path(args.cache_dir).expanduser().resolve())]
        if args.score_archive:
            command += ['--score-archive', str(args.score_archive)]
        if Path(args.model).is_dir():
            command += ['--model', str(Path(args.model).resolve())]
        environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': args.gpus if args.device == 'cuda' else '',
            'PYTHONUNBUFFERED': '1', 'TOKENIZERS_PARALLELISM': 'false',
            'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS', '4')}
        print(f'[launch] {len(gpus)} DDP ranks; GPUs={args.gpus}; complete texts up to {args.context} tokens; '
              f'{len(gpus)*args.accumulation} query groups/update; up to '
              f'{len(gpus)*args.accumulation*args.documents_per_query} labeled pairs/update', flush=True)
        subprocess.run(command, cwd=ROOT, env=environment, check=True)


if __name__ == '__main__':
    main()
