#!/usr/bin/env python3
"""Bounded forward/backward timing on fixed probes; never updates or saves weights."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / 'matchminer-ai-inference-kehl/src'))

import torch
from colbert_experiments.model import BASE_MODEL, ColBERT
from colbert_experiments.ordinal_ranking import cached_ranking_backward
from colbert_experiments.retained_ranking import retained_ranking_backward


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probes', type=Path, required=True, help='Synthetic/non-PHI fixed_probes.json')
    p.add_argument('--model', default=BASE_MODEL)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--groups', type=int, default=8)
    p.add_argument('--documents', type=int, default=8)
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--warmup', type=int, default=1, help='Untimed full-batch runs per configuration')
    p.add_argument('--check-gradients', action='store_true', help='Enable deterministic kernels and compare gradients')
    p.add_argument('--gradient-reference', choices=['replay', 'retained_gpu'], default='replay')
    p.add_argument('--deterministic', action='store_true', help='Use deterministic CUDA kernels for gradient checks')
    p.add_argument('--profile', type=Path, help='Export a separate final-configuration CUDA trace')
    p.add_argument('--encoder-token-budget', type=int, default=65536)
    p.add_argument('--group-counts', type=int, nargs='+', default=[1, 4])
    p.add_argument('--backward-modes', choices=['replay', 'retained'], nargs='+', default=['replay'])
    p.add_argument('--activation-storage', choices=['auto', 'gpu', 'cpu', 'direct'], default='auto')
    p.add_argument('--activation-gpu-budget-gb', type=float, default=0)
    args = p.parse_args()
    if min(args.groups, args.documents, args.repeats, args.encoder_token_budget, *args.group_counts) < 1 or args.warmup < 0:
        p.error('All batch limits must be positive')
    torch.set_num_threads(4)
    args.deterministic = args.deterministic or args.check_gradients
    if args.deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    strata = defaultdict(list)
    for group in json.loads(args.probes.read_text())['train']:
        strata[group['stratum']].append(group)
    chosen = []
    while len(chosen) < args.groups and any(strata.values()):
        for rows in strata.values():
            if rows and len(chosen) < args.groups:
                chosen.append(rows.pop(0))
    batch = []
    for group in chosen:
        row = {**group, 'direction': group['stratum'].split('/')[0]}
        batch.append((row, list(range(min(args.documents, len(group['documents']))))))
    if len(batch) != args.groups:
        p.error(f'Requested {args.groups} groups but only {len(batch)} probes are available')
    model = ColBERT(args.model).to(device).train()
    model.configure_token_cache(8000000)
    model.prepare_groups([g['query'] for g, _ in batch],
                         [[g['documents'][i]['text'] for i in indices] for g, indices in batch])
    def run(groups_per_forward, rows, mode='replay', storage=None):
        model.zero_grad(set_to_none=True)
        backward = retained_ranking_backward if mode == 'retained' else cached_ranking_backward
        options = dict(activation_storage=storage or args.activation_storage,
                       activation_gpu_budget_gb=args.activation_gpu_budget_gb,
                       profile_memory=True) if mode == 'retained' else {}
        return backward(model, model, rows, device=device, rank=0, world=1,
            seed_for_micro=lambda micro: 42 + micro, groups_per_forward=groups_per_forward,
            encoder_token_budget=args.encoder_token_budget, **options)
    print(json.dumps(dict(device=torch.cuda.get_device_name(device), groups=len(batch),
        pairs=sum(len(i) for _, i in batch), python=platform.python_version(), torch=torch.__version__,
        cuda=torch.version.cuda, visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        warmup=args.warmup, repeats=args.repeats, deterministic=args.deterministic,
        encoder_token_budget=args.encoder_token_budget,
        activation_storage=args.activation_storage, activation_gpu_budget_gb=args.activation_gpu_budget_gb)), flush=True)
    for mode, group_count in [(m, g) for m in args.backward_modes for g in args.group_counts]:
        reference = None
        if args.check_gradients:
            reference_loss = run(group_count, batch,
                'retained' if args.gradient_reference == 'retained_gpu' else 'replay', storage='gpu')['loss'].cpu()
            reference = [p.grad.detach().cpu().clone() if p.grad is not None else None for p in model.parameters()]
        for _ in range(args.warmup):
            run(group_count, batch, mode)
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        samples, peaks, reserved_peaks = [], [], []
        for _ in range(args.repeats):
            started = time.perf_counter()
            metrics = run(group_count, batch, mode)
            torch.cuda.synchronize()
            samples.append(time.perf_counter() - started)
            peaks.append(max(torch.cuda.max_memory_allocated(device) / 2**30,
                             metrics.get('forward_peak_allocated_gb', 0)))
            reserved_peaks.append(torch.cuda.max_memory_reserved(device) / 2**30)
        elapsed = statistics.mean(samples)
        gradient_metrics = {}
        if reference is not None:
            torch.testing.assert_close(metrics['loss'].cpu(), reference_loss)
            squared_error = squared_reference = 0.
            max_error = 0.
            for parameter, expected in zip(model.parameters(), reference, strict=True):
                if expected is None:
                    assert parameter.grad is None
                    continue
                actual = parameter.grad.detach().cpu()
                difference = (actual - expected).double()
                squared_error += float(difference.square().sum())
                squared_reference += float(expected.double().square().sum())
                max_error = max(max_error, float(difference.abs().max()))
            relative_l2 = (squared_error / max(squared_reference, 1e-30)) ** .5
            gradient_metrics = dict(gradient_relative_l2_error=relative_l2, gradient_max_abs_error=max_error,
                                    gradient_reference=args.gradient_reference)
        print(json.dumps(dict(query_groups_per_forward=group_count, ranking_backward=mode, seconds=elapsed,
            seconds_per_repeat=samples,
            pairs_per_second=sum(len(i) for _, i in batch) / elapsed,
            peak_allocated_gb=max(peaks), peak_reserved_gb=max(reserved_peaks),
            encoder_batches_per_pass=metrics['encoder_batches_per_pass'],
            **{key: metrics[key] for key in ('replay_max_error', 'encoder_forward_passes',
                'model_backward_calls', 'activation_offloaded_gb',
                'activation_gpu_soft_limit_gb') if key in metrics},
            phase_seconds_last_repeat={k: v for k, v in metrics.items() if k.endswith('_seconds')},
            phase_memory_gb_last_repeat={k: v for k, v in metrics.items() if k.endswith('_allocated_gb')},
            pinned_allocator_gb={k: v / 2**30 for k, v in
                (torch.cuda.memory.host_memory_stats().items() if hasattr(torch.cuda.memory, 'host_memory_stats') else [])
                if k in ('allocated_bytes.peak', 'allocated_bytes.current', 'active_bytes.peak')},
            **gradient_metrics, token_cache=model.token_cache_info())), flush=True)
        if reference is not None and relative_l2 > 1e-3:
            raise AssertionError(f'Gradient relative L2 error {relative_l2} exceeds 0.001')
    if args.profile:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as profiler:
            run(group_count, batch, mode)
        profiler.export_chrome_trace(str(args.profile))


if __name__ == '__main__':
    main()
