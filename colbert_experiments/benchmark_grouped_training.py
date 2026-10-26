#!/usr/bin/env python3
"""Bounded forward/backward timing on fixed probes; never updates or saves weights."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
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
    p.add_argument('--encoder-token-budget', type=int, default=65536)
    p.add_argument('--group-counts', type=int, nargs='+', default=[1, 4])
    p.add_argument('--backward-modes', choices=['replay', 'retained'], nargs='+', default=['replay'])
    p.add_argument('--activation-storage', choices=['auto', 'gpu', 'cpu'], default='auto')
    p.add_argument('--activation-gpu-budget-gb', type=float, default=0)
    args = p.parse_args()
    if min(args.groups, args.documents, args.repeats, args.encoder_token_budget) < 1:
        p.error('All batch limits must be positive')
    torch.set_num_threads(4)
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
    model = ColBERT(args.model).to(device).train()
    model.configure_token_cache(8000000)
    model.prepare_groups([g['query'] for g, _ in batch],
                         [[g['documents'][i]['text'] for i in indices] for g, indices in batch])
    def run(groups_per_forward, rows, mode='replay'):
        model.zero_grad(set_to_none=True)
        backward = retained_ranking_backward if mode == 'retained' else cached_ranking_backward
        options = dict(activation_storage=args.activation_storage,
                       activation_gpu_budget_gb=args.activation_gpu_budget_gb) if mode == 'retained' else {}
        return backward(model, model, rows, device=device, rank=0, world=1,
            seed_for_micro=lambda micro: 42 + micro, groups_per_forward=groups_per_forward,
            encoder_token_budget=args.encoder_token_budget, **options)
    run(1, batch[:1])
    print(json.dumps(dict(device=torch.cuda.get_device_name(device), groups=len(batch),
                          pairs=sum(len(i) for _, i in batch))), flush=True)
    for mode, group_count in [(m, g) for m in args.backward_modes for g in args.group_counts]:
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        for _ in range(args.repeats):
            metrics = run(group_count, batch, mode)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - started) / args.repeats
        print(json.dumps(dict(query_groups_per_forward=group_count, ranking_backward=mode, seconds=elapsed,
            pairs_per_second=sum(len(i) for _, i in batch) / elapsed,
            peak_allocated_gb=torch.cuda.max_memory_allocated(device) / 2**30,
            encoder_batches_per_pass=metrics['encoder_batches_per_pass'],
            **{key: metrics[key] for key in ('replay_max_error', 'encoder_forward_passes',
                'model_backward_calls', 'activation_offloaded_gb', 'activation_gpu_soft_limit_gb') if key in metrics},
            token_cache=model.token_cache_info())), flush=True)


if __name__ == '__main__':
    main()
