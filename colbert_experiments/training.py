"""Explicit per-query positives/negatives; optional two-GPU DistributedDataParallel."""
from collections import defaultdict
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .contracts import DIRECTIONS, TASKS, contract_hash, query_text
from .data import digest, load_corpus, read_jsonl, write_json
from .model import ColBERT, grouped_loss
from .learning import make_probes, evaluate_probes


def build_groups(patients, cohorts, labels, *, split='train'):
    patients, cohorts = {r['id']: r for r in patients}, {r['id']: r for r in cohorts}
    groups, seen = {}, {}
    expected_contract = contract_hash()
    for label in labels:
        if label.get('contract') != expected_contract:
            raise ValueError('Label prompt/schema contract differs from the current experiment')
        p, c, task = patients[label['patient_id']], cohorts[label['cohort_id']], label['task']
        if task not in TASKS or label['answer'] not in ('yes', 'no'):
            raise ValueError('Invalid binary task label')
        if p['split'] != c['split'] or label['split'] != p['split']:
            raise ValueError('Label crosses corpus splits')
        if p['split'] != split:
            continue
        key = (p['id'], c['id'], task)
        if key in seen:
            if seen[key] != label['answer']:
                raise ValueError('Conflicting duplicate labels; explicitly adjudicate first')
            continue
        seen[key] = label['answer']
        for direction in DIRECTIONS:
            query, document = (p, c) if direction == 'patient_to_cohort' else (c, p)
            group = groups.setdefault((direction, task, query['id']), dict(direction=direction,
                task=task, query_id=query['id'], query=query_text(task, direction, query['text']),
                positive=[], negative=[]))
            group['positive' if label['answer'] == 'yes' else 'negative'].append(document)
    usable = [g for g in groups.values() if g['positive'] and g['negative']]
    report = dict(total_groups=len(groups), usable_groups=len(usable),
                  skipped_without_both_classes=len(groups) - len(usable), strata={})
    for direction in DIRECTIONS:
        for task in TASKS:
            report['strata'][direction + '/' + task] = sum(g['direction'] == direction and g['task'] == task for g in usable)
    return usable, report


def train(args):
    world, rank, local_rank = (int(os.environ.get(k, default)) for k, default in
                              [('WORLD_SIZE', '1'), ('RANK', '0'), ('LOCAL_RANK', '0')])
    cpu = getattr(args, 'device', 'cuda') == 'cpu'
    if not cpu and not torch.cuda.is_available():
        raise RuntimeError('Training requires an explicitly selected CUDA device')
    if not cpu:
        torch.cuda.set_device(local_rank)
    if world > 1:
        if cpu:
            dist.init_process_group('gloo')
        else:
            dist.init_process_group('nccl', device_id=torch.device('cuda', local_rank))
    try:
        _train(args, world, rank, local_rank)
    finally:
        if world > 1:
            dist.destroy_process_group()


def _train(args, world, rank, local_rank):
    device = torch.device('cpu' if getattr(args, 'device', 'cuda') == 'cpu' else f'cuda:{local_rank}')
    if (args.steps is not None and args.steps < 1) or args.accumulation < 1 or args.documents_per_class < 1 or args.save_steps < 1:
        raise ValueError('Training step, accumulation and group sizes must be positive')
    patients, cohorts = load_corpus(args.corpus)
    labels = [row for path in args.labels for row in read_jsonl(path)]
    regression = getattr(args, 'objective', 'binary') == 'gated_regression'
    group_builder, probe_builder, evaluator = build_groups, make_probes, evaluate_probes
    if regression:
        from . import regression as continuous
        group_builder, probe_builder, evaluator = continuous.build_groups, continuous.make_probes, continuous.evaluate_probes
        if args.documents_per_query < 1:
            raise ValueError('documents-per-query must be positive')
    groups, group_report = group_builder(patients, cohorts, labels)
    val_groups, val_report = group_builder(patients, cohorts, labels, split='val')
    if not groups:
        raise ValueError('No usable labeled training queries')
    batch_sampler = None
    if regression:
        from .batching import RegressionBatchSampler
        batch_sampler = RegressionBatchSampler(groups)
        group_report['batch_sampling'] = batch_sampler.report()
    if args.steps is None:
        if not math.isfinite(args.epochs) or args.epochs <= 0:
            raise ValueError('Expected query sampling passes must be positive')
        args.steps = max(1, math.ceil(len(groups) * args.epochs / (world * args.accumulation)))
    probes = {split: probe_builder(rows, per_stratum=args.eval_groups_per_stratum,
                                documents_per_class=args.documents_per_query if regression else 2, seed=args.seed)
              for split, rows in [('train', groups), ('val', val_groups)]} if args.eval_every else {}
    buckets = defaultdict(list)
    for group in groups:
        buckets[(group['direction'], group['task'])].append(group)
    strata = sorted(buckets)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    model = ColBERT(args.model, revision=args.revision, query_length=args.query_length,
                   document_length=args.document_length, cache_dir=args.cache_dir, checkpointing=True).to(device)
    identity = dict(contract=contract_hash(), patients=digest(patients), cohorts=digest(cohorts), labels=digest(labels),
        base=model.fingerprint(), world_size=world, steps=args.steps, accumulation=args.accumulation,
        documents_per_class=args.documents_per_class, learning_rate=args.learning_rate,
        temperature=args.temperature, seed=args.seed, score='mean_maxsim', loss='explicit_pairwise_logistic',
        eval_every=args.eval_every, probes_sha256=digest(probes))
    if regression:
        identity.update(contract=continuous.score_contract(), loss='mse_normalized_gated_score',
                        documents_per_query=args.documents_per_query, prediction_link=args.prediction_link,
                        normalization=5, query_instruction=continuous.INSTRUCTION,
                        batch_sampling=batch_sampler.report(),
                        query_groups_per_update=world * args.accumulation,
                        max_pairs_per_update=world * args.accumulation * args.documents_per_query)
    if device.type == 'cpu':
        identity['execution_device'] = 'cpu'
    manifest = output / 'training_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError('Training identity changed; use a new round directory')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.01)
    step, history, evaluations = 0, [], []
    latest = output / 'latest.json'
    if latest.exists():
        checkpoint = output / json.loads(latest.read_text())['path']
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        step, history = state['step'], state['history']
        evaluations = state.get('evaluations', [])
    if rank == 0:
        write_json(manifest, identity)
        write_json(output / 'group_report.json', group_report)
        write_json(output / 'validation_group_report.json', val_report)
        if probes:
            write_json(output / 'fixed_probes.json', probes)
            if regression:
                write_json(output / 'baseline_metrics.json', continuous.constant_baselines(groups, probes))
    wrapped = DistributedDataParallel(model, device_ids=[local_rank] if device.type == 'cuda' else None) if world > 1 else model
    wrapped.train()
    start = time.monotonic()

    def checkpoint(current):
        if rank == 0:
            path = output / 'checkpoint.pt'
            temporary = path.with_suffix('.tmp')
            torch.save(dict(step=current, history=history, evaluations=evaluations,
                            model=model.state_dict(), optimizer=optimizer.state_dict()), temporary)
            temporary.replace(path)
            write_json(latest, dict(path=path.name, step=current))
        if world > 1:
            dist.barrier()

    def evaluate(current):
        entry = dict(step=current)
        for split, items in probes.items():
            options = dict(link=args.prediction_link) if regression else dict(temperature=args.temperature)
            summary, scores = evaluator(model, items, rank=rank, world=world, **options)
            entry[split] = summary
            if rank == 0:
                write_json(output / 'probe_scores' / f'{split}_{current:06d}.json', scores)
        evaluations.append(entry)
        if rank == 0:
            write_json(output / 'evaluation_history.json', evaluations)
            print('[evaluate] ' + json.dumps(entry), flush=True)

    if probes and (not evaluations or evaluations[-1]['step'] != step):
        evaluate(step)

    for current in range(step, args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), device=device)
        batch_score_counts = [0] * 6
        if regression:
            global_batch = batch_sampler.batch(query_groups=world * args.accumulation,
                documents_per_query=args.documents_per_query, seed=args.seed, step=current)
            update_batch = global_batch[rank::world]
        for micro in range(args.accumulation):
            index = (current * args.accumulation + micro) * world + rank
            rng = random.Random(args.seed + index)
            torch.manual_seed(args.seed + index)
            if regression:
                group, indices = update_batch[micro]
                texts = [group['documents'][i]['text'] for i in indices]
                targets = [group['labels'][i] for i in indices]
                for target in targets:
                    batch_score_counts[target] += 1
            else:
                group = rng.choice(buckets[strata[index % len(strata)]])
                positives = rng.sample(group['positive'], min(args.documents_per_class, len(group['positive'])))
                negatives = rng.sample(group['negative'], min(args.documents_per_class, len(group['negative'])))
                texts = [d['text'] for d in positives + negatives]
                targets = [1] * len(positives) + [0] * len(negatives)
            sync = wrapped.no_sync() if world > 1 and micro + 1 < args.accumulation else nullcontext()
            with sync, torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                scores = wrapped(group['query'], texts)
                loss = (continuous.regression_loss(scores, targets, link=args.prediction_link) if regression
                        else grouped_loss(scores, targets, args.temperature))
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                (loss / args.accumulation).backward()
            loss_sum += loss.detach() / args.accumulation
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if world > 1:
            dist.all_reduce(loss_sum)
            loss_sum /= world
        batch_metrics = {}
        if regression:
            counts = torch.tensor(batch_score_counts, device=device, dtype=torch.long)
            if world > 1:
                dist.all_reduce(counts)
            observed = counts.cpu().tolist()
            present = [i for i, count in enumerate(observed) if count]
            batch_metrics = dict(batch_pairs=sum(observed), batch_score_counts=dict(enumerate(observed)),
                batch_min_score=min(present), batch_max_score=max(present),
                batch_query_groups=world * args.accumulation,
                batch_unique_query_groups=len({(g['direction'], g['query_id']) for g, _ in global_batch}),
                batch_unique_directional_pairs=len({(g['direction'], g['query_id'], g['documents'][i]['id'])
                    for g, indices in global_batch for i in indices}))
        if rank == 0:
            entry = dict(step=current + 1, loss=float(loss_sum), gradient_norm=float(norm),
                         elapsed_seconds=round(time.monotonic() - start, 2), **batch_metrics)
            history.append(entry)
            write_json(output / 'status.json', dict(state='training', **entry))
            print(json.dumps(entry), flush=True)
        if probes and ((current + 1) % args.eval_every == 0 or current + 1 == args.steps):
            evaluate(current + 1)
        if (current + 1) % args.save_steps == 0 or current + 1 == args.steps:
            checkpoint(current + 1)
    if rank == 0:
        model_output = Path(getattr(args, 'output_model', None) or output / 'model')
        model.save(model_output)
        if regression:
            write_json(model_output / 'regression_scoring.json', dict(contract=continuous.score_contract(),
                score='mean_maxsim', prediction_link=args.prediction_link,
                sigmoid_scale=20, sigmoid_center=.5, output_scale=5, query_instruction=continuous.INSTRUCTION,
                semantics='gated potential-eligibility plus specificity score; not a probability'))
        write_json(output / 'metrics.json', dict(history=history, evaluations=evaluations, groups=group_report,
            gpu_count=world if device.type == 'cuda' else 0,
            gpu_name=torch.cuda.get_device_name(local_rank) if device.type == 'cuda' else 'cpu',
            max_memory_allocated=torch.cuda.max_memory_allocated(local_rank) if device.type == 'cuda' else 0))
        write_json(output / 'status.json', dict(state='complete', steps=args.steps, model=str(model_output)))
    if world > 1:
        dist.barrier()
