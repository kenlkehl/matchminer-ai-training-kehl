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
from .data import digest, load_corpus, read_jsonl, write_json, selected_splits
from .model import ColBERT, grouped_loss
from .learning import make_probes, evaluate_probes
from .checkpointing import PauseRequest, save_checkpoint
from .complete_pass_batches import CompletePassBatchSampler
from .ordinal_ranking import cached_ranking_backward


def build_groups(patients, cohorts, labels, *, split='train'):
    included = selected_splits(split)
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
        if p['split'] not in included:
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
        with PauseRequest(stop_at=getattr(args, 'stop_at', None),
                          pause_file=getattr(args, 'pause_file', None)) as pause:
            return _train(args, world, rank, local_rank, pause)
    finally:
        if world > 1:
            dist.destroy_process_group()


def _train(args, world, rank, local_rank, pause):
    device = torch.device('cpu' if getattr(args, 'device', 'cuda') == 'cpu' else f'cuda:{local_rank}')
    if (args.steps is not None and args.steps < 1) or args.accumulation < 1 or args.documents_per_class < 1 or args.save_steps < 1:
        raise ValueError('Training step, accumulation and group sizes must be positive')
    if getattr(args, 'save_total_limit', 2) < 2:
        raise ValueError('Checkpoint limit must be at least two for atomic rotation')
    patients, cohorts = load_corpus(args.corpus)
    labels = [row for path in args.labels for row in read_jsonl(path)]
    regression = getattr(args, 'objective', 'binary') == 'gated_regression'
    loss_name = getattr(args, 'loss', 'mse')
    ranking = regression and loss_name != 'mse'
    groups_per_forward = getattr(args, 'query_groups_per_forward', 1)
    encoder_token_budget = getattr(args, 'encoder_token_budget', 32768)
    token_cache_tokens = getattr(args, 'token_cache_tokens', 8000000)
    if min(groups_per_forward, encoder_token_budget) < 1 or token_cache_tokens < 0:
        raise ValueError('Use positive encoder batch limits and nonnegative token cache capacity')
    schedule = getattr(args, 'lr_scheduler', 'constant')
    warmup = getattr(args, 'warmup_ratio', .01)
    scale, contrastive_weight = getattr(args, 'ranking_scale', 20.), getattr(args, 'contrastive_weight', 1.)
    if (not all(math.isfinite(v) for v in (args.learning_rate, scale, contrastive_weight, warmup))
            or min(args.learning_rate, scale) <= 0 or contrastive_weight < 0 or not 0 <= warmup < 1):
        raise ValueError('Use positive LR/scale, nonnegative contrast weight and warmup ratio in [0, 1)')
    if ranking and args.prediction_link != 'identity':
        raise ValueError('Ranking objectives use raw MaxSim, without a sigmoid prediction link')
    group_builder, probe_builder, evaluator = build_groups, make_probes, evaluate_probes
    if regression:
        from . import regression as continuous
        group_builder, probe_builder, evaluator = continuous.build_groups, continuous.make_probes, continuous.evaluate_probes
        if args.documents_per_query < 1:
            raise ValueError('documents-per-query must be positive')
    training_splits = selected_splits(getattr(args, 'training_splits', ['train']))
    groups, group_report = group_builder(patients, cohorts, labels, split=training_splits)
    val_groups, val_report = group_builder(patients, cohorts, labels, split='val')
    validation_is_held_out = 'val' not in training_splits
    val_report['held_out'] = validation_is_held_out
    if not validation_is_held_out:
        val_groups = []
    group_report['training_splits'] = training_splits
    if not groups:
        raise ValueError('No usable labeled training queries')
    if getattr(args, 'require_complete_coverage', False):
        missing = {}
        for direction, records in [('patient_to_cohort', patients), ('cohort_to_patient', cohorts)]:
            expected = {r['id'] for r in records if r['split'] in training_splits}
            present = {g['query_id'] for g in groups if g['direction'] == direction}
            missing[direction] = len(expected - present)
        if any(missing.values()):
            raise ValueError('Missing teacher labels for selected training splits: '
                             f'{missing["patient_to_cohort"]} patients, {missing["cohort_to_patient"]} cohorts')
    batch_sampler = None
    if regression:
        if ranking:
            batch_sampler = CompletePassBatchSampler(groups, documents_per_query=args.documents_per_query,
                query_groups_per_update=world * args.accumulation, seed=args.seed)
        else:
            from .batching import RegressionBatchSampler
            batch_sampler = RegressionBatchSampler(groups)
        group_report['batch_sampling'] = batch_sampler.report()
    if args.steps is None:
        if not math.isfinite(args.epochs) or args.epochs <= 0:
            raise ValueError('Expected query sampling passes must be positive')
        args.steps = max(1, math.ceil(args.epochs * (batch_sampler.updates_per_epoch if ranking
                                                  else len(groups) / (world * args.accumulation))))
    if rank == 0 and ranking:
        print(f'[plan] {batch_sampler.pairs:,} directional pairs/epoch; '
              f'{batch_sampler.updates_per_epoch:,} updates/epoch; {args.steps:,} total updates; '
              f'loss={loss_name}; LR={args.learning_rate:g} ({schedule})', flush=True)
    probes = {split: probe_builder(rows, per_stratum=args.eval_groups_per_stratum,
                                documents_per_class=args.documents_per_query if regression else 2, seed=args.seed)
              for split, rows in [('train', groups), ('val', val_groups)] if rows} if args.eval_every else {}
    buckets = defaultdict(list)
    for group in groups:
        buckets[(group['direction'], group['task'])].append(group)
    strata = sorted(buckets)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    model = ColBERT(args.model, revision=args.revision, query_length=args.query_length,
                   document_length=args.document_length, cache_dir=args.cache_dir,
                   checkpointing=getattr(args, 'gradient_checkpointing', True)).to(device)
    if hasattr(model, 'configure_token_cache'):
        model.configure_token_cache(token_cache_tokens)
    identity = dict(contract=contract_hash(), patients=digest(patients), cohorts=digest(cohorts), labels=digest(labels),
        base=model.fingerprint(), world_size=world, steps=args.steps, accumulation=args.accumulation,
        documents_per_class=args.documents_per_class, learning_rate=args.learning_rate,
        temperature=args.temperature, seed=args.seed, score='mean_maxsim', loss='explicit_pairwise_logistic',
        eval_every=args.eval_every, probes_sha256=digest(probes), training_splits=training_splits,
        validation_is_held_out=validation_is_held_out,
        require_complete_coverage=getattr(args, 'require_complete_coverage', False))
    if regression:
        identity.update(contract=continuous.score_contract(), loss='mse_normalized_gated_score',
                        documents_per_query=args.documents_per_query, prediction_link=args.prediction_link,
                        normalization=5, query_instruction=continuous.INSTRUCTION,
                        batch_sampling=batch_sampler.report(),
                        query_groups_per_update=world * args.accumulation,
                        max_pairs_per_update=world * args.accumulation * args.documents_per_query)
        if ranking:
            identity.pop('normalization')
            identity.update(loss=loss_name, ranking_scale=scale, contrastive_weight=contrastive_weight,
                effective_batch_comparisons='two-pass scalar-score gradient cache with seeded forward replay',
                loss_weighting='mean CoSENT over directions with unequal labels; mean contrast over queries with known positives and zeros')
    if schedule != 'constant':
        identity.update(lr_scheduler=schedule, warmup_ratio=warmup)
    if device.type == 'cpu':
        identity['execution_device'] = 'cpu'
    if not getattr(args, 'gradient_checkpointing', True):
        identity['gradient_checkpointing'] = False
    if hasattr(model, 'execution_settings'):
        identity['encoder_execution'] = model.execution_settings()
    if ranking and groups_per_forward > 1:
        identity['ranking_encoder_batches'] = dict(query_groups_per_forward=groups_per_forward,
                                                   padded_token_budget=encoder_token_budget)
    manifest = output / 'training_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError('Training identity changed; use a new round directory')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.01)
    scheduler = None
    if schedule == 'linear':
        from transformers import get_linear_schedule_with_warmup
        scheduler = get_linear_schedule_with_warmup(optimizer,
            num_warmup_steps=math.ceil(args.steps * warmup), num_training_steps=args.steps)
    step, history, evaluations = 0, [], []
    latest = output / 'latest.json'
    had_checkpoint = latest.exists()
    if had_checkpoint:
        checkpoint = output / json.loads(latest.read_text())['path']
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(state['scheduler'])
        step, history = state['step'], state['history']
        evaluations = state.get('evaluations', [])
    if rank == 0:
        write_json(manifest, identity)
        write_json(output / 'group_report.json', group_report)
        write_json(output / 'validation_group_report.json', val_report)
        if probes:
            write_json(output / 'fixed_probes.json', probes)
            if regression and not ranking:
                write_json(output / 'baseline_metrics.json', continuous.constant_baselines(groups, probes))
    wrapped = DistributedDataParallel(model, device_ids=[local_rank] if device.type == 'cuda' else None) if world > 1 else model
    wrapped.train()
    start = time.monotonic()

    def checkpoint(current):
        if rank == 0:
            save_checkpoint(output, dict(step=current, history=history, evaluations=evaluations,
                model=model.state_dict(), optimizer=optimizer.state_dict(),
                **(dict(scheduler=scheduler.state_dict()) if scheduler is not None else {})),
                limit=getattr(args, 'save_total_limit', 2))
        if world > 1:
            dist.barrier()

    def pause_if_requested(current):
        if not pause.requested(device, world):
            return False
        # A partial accumulation has not updated model or optimizer state.
        # Discard its gradients and resume the same deterministic update later.
        optimizer.zero_grad(set_to_none=True)
        checkpoint(current)
        if rank == 0:
            write_json(output / 'status.json', dict(state='paused', step=current,
                steps=args.steps, checkpoint=json.loads(latest.read_text()),
                reason='pause request or training deadline', resumable=True))
        return True

    def evaluate(current):
        entry = dict(step=current)
        for split, items in probes.items():
            options = dict(link=args.prediction_link, objective=loss_name, scale=scale,
                           contrastive_weight=contrastive_weight) if regression else dict(temperature=args.temperature)
            summary, scores = evaluator(model, items, rank=rank, world=world, **options)
            entry[split] = summary
            if rank == 0:
                write_json(output / 'probe_scores' / f'{split}_{current:06d}.json', scores)
        evaluations.append(entry)
        if rank == 0:
            write_json(output / 'evaluation_history.json', evaluations)
            print('[evaluate] ' + json.dumps(entry), flush=True)

    if pause_if_requested(step):
        return 'paused'
    if not had_checkpoint:
        checkpoint(step)
    if probes and (not evaluations or evaluations[-1]['step'] != step):
        evaluate(step)

    for current in range(step, args.steps):
        step_started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), device=device)
        batch_score_counts = [0] * 6
        ranking_metrics = {}
        if regression:
            global_batch = batch_sampler.batch(current) if ranking else batch_sampler.batch(
                query_groups=world * args.accumulation,
                documents_per_query=args.documents_per_query, seed=args.seed, step=current)
            update_batch = global_batch[rank::world]
        if ranking:
            result = cached_ranking_backward(model, wrapped, global_batch, device=device, rank=rank, world=world,
                seed_for_micro=lambda micro: args.seed + (current * args.accumulation + micro) * world + rank,
                objective=loss_name, scale=scale, contrastive_weight=contrastive_weight,
                groups_per_forward=groups_per_forward, encoder_token_budget=encoder_token_budget,
                should_pause=lambda: pause_if_requested(current))
            if result is None:
                return 'paused'
            loss_sum = result.pop('loss')
            ranking_metrics = {k: float(v) if isinstance(v, torch.Tensor) else v for k, v in result.items()}
            ranking_metrics['epoch'] = current // batch_sampler.updates_per_epoch + 1
            for group, indices in update_batch:
                for i in indices:
                    batch_score_counts[group['labels'][i]] += 1
        for micro in range(args.accumulation if not ranking else 0):
            if pause_if_requested(current):
                return 'paused'
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
        learning_rate = optimizer.param_groups[0]['lr']
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if world > 1 and not ranking:
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
                batch_query_groups=len(global_batch),
                batch_unique_query_groups=len({(g['direction'], g['query_id']) for g, _ in global_batch}),
                batch_unique_directional_pairs=len({(g['direction'], g['query_id'], g['documents'][i]['id'])
                    for g, indices in global_batch for i in indices}))
        if rank == 0:
            entry = dict(step=current + 1, steps=args.steps, loss=float(loss_sum), gradient_norm=float(norm),
                         learning_rate=learning_rate, elapsed_seconds=round(time.monotonic() - start, 2),
                         step_seconds=round(time.monotonic() - step_started, 3),
                         **(dict(gpu_peak_allocated_gb=round(torch.cuda.max_memory_allocated(device) / 2**30, 2))
                            if device.type == 'cuda' else {}),
                         **ranking_metrics, **batch_metrics)
            if hasattr(model, 'token_cache_info'):
                entry['token_cache'] = model.token_cache_info()
            history.append(entry)
            write_json(output / 'status.json', dict(state='training', **entry))
            print(json.dumps(entry), flush=True)
        if pause_if_requested(current + 1):
            return 'paused'
        epoch_end = ranking and (current + 1) % batch_sampler.updates_per_epoch == 0
        if probes and ((current + 1) % args.eval_every == 0 or epoch_end or current + 1 == args.steps):
            evaluate(current + 1)
        if (current + 1) % args.save_steps == 0 or epoch_end or current + 1 == args.steps:
            checkpoint(current + 1)
    if rank == 0:
        model_output = Path(getattr(args, 'output_model', None) or output / 'model')
        model.save(model_output)
        if regression:
            scoring = dict(contract=continuous.score_contract(),
                score='mean_maxsim', prediction_link=args.prediction_link,
                sigmoid_scale=20, sigmoid_center=.5, output_scale=5, query_instruction=continuous.INSTRUCTION,
                semantics='gated potential-eligibility plus specificity score; not a probability')
            if ranking:
                for key in ('output_scale', 'sigmoid_scale', 'sigmoid_center'):
                    scoring.pop(key)
                scoring.update(objective=loss_name, calibrated=False,
                    semantics='Ordinal retrieval score learned from gated labels; not a probability or predicted 0-5 score')
            write_json(model_output / 'regression_scoring.json', scoring)
        write_json(output / 'metrics.json', dict(history=history, evaluations=evaluations, groups=group_report,
            gpu_count=world if device.type == 'cuda' else 0,
            gpu_name=torch.cuda.get_device_name(local_rank) if device.type == 'cuda' else 'cpu',
            max_memory_allocated=torch.cuda.max_memory_allocated(local_rank) if device.type == 'cuda' else 0))
        write_json(output / 'status.json', dict(state='complete', steps=args.steps, model=str(model_output)))
    if world > 1:
        dist.barrier()
    return 'complete'
