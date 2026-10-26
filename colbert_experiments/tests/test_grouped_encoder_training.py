import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from colbert_experiments.ranking_microbatches import prepare_encoder_batches


class PackingModel:
    def prepare_groups(self, queries, documents):
        return [(q, d) for q, d in zip(queries, documents)]

    def prepared_batch_tokens(self, groups):
        return sum(int(q) for q, _ in groups)


def test_packing_preserves_every_pair_and_keeps_oversized_groups_complete():
    batch = [(dict(query=str(n), documents=[dict(text=str(j)) for j in range(i + 1)]), list(range(i + 1)))
             for i, n in enumerate([3, 4, 100, 2, 2])]
    offsets = [0, 1, 3, 6, 10, 15]
    packets, _ = prepare_encoder_batches(PackingModel(), batch, offsets, rank=0, world=1,
        device='cpu', groups_per_forward=3, token_budget=8)
    assert [len(p.kwargs['prepared_groups']) for p in packets] == [2, 1, 2]
    assert torch.cat([p.positions for p in packets]).tolist() == list(range(15))
    assert [p.seed_micro for p in packets] == [0, 2, 3]
    empty, fallback = prepare_encoder_batches(PackingModel(), batch[:1], offsets[:2], rank=1, world=2,
        device='cpu', groups_per_forward=3, token_budget=8)
    assert not empty and fallback['prepared_groups']


WORKER = r'''
import copy, os, sys, torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from colbert_experiments.ordinal_ranking import cached_ranking_backward, ranking_loss
from colbert_experiments.ranking_microbatches import prepare_encoder_batches
rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
dist.init_process_group('gloo')
class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4)
        self.dropout = torch.nn.Dropout(.3)
        self.output = torch.nn.Linear(4, 1)
    def prepare_groups(self, queries, documents):
        return list(zip(queries, documents))
    def prepared_batch_tokens(self, groups):
        return sum(int(q) * (len(d) + 1) for q, d in groups)
    def forward(self, query=None, documents=None, *, prepared_groups=None):
        groups = prepared_groups if prepared_groups is not None else [(query, documents)]
        x = torch.tensor([[float(q), float(d), float(q) * float(d)] for q, docs in groups for d in docs])
        return self.output(self.dropout(self.linear(x).tanh())).squeeze(1)
torch.manual_seed(17)
model = Toy()
reference = copy.deepcopy(model)
wrapped = DDP(model)
batch = []
for i, (q, direction, labels) in enumerate([
    ('1', 'forward', [0, 5]), ('1', 'forward', [2]), ('2', 'reverse', [3, 0]),
    ('3', 'reverse', [0]), ('4', 'forward', [5]), ('1', 'reverse', [0, 4]), ('2', 'forward', [1])]):
    batch.append((dict(query=q, query_id=q, direction=direction, labels=labels,
                       documents=[dict(text=str(i + j + 1)) for j in range(len(labels))]), list(range(len(labels)))))
for objective in ('cosent', 'legacy_ranking'):
    for size in (1, 3, 7):
        for budget in (7, 100):
            model.zero_grad(set_to_none=True)
            reference.zero_grad(set_to_none=True)
            result = cached_ranking_backward(model, wrapped, batch[:size], device=torch.device('cpu'),
                rank=rank, world=world, seed_for_micro=lambda micro: 123 + micro * world + rank,
                objective=objective, groups_per_forward=3, encoder_token_budget=budget)
            labels, directions, queries, offsets = [], [], [], [0]
            for group, indices in batch[:size]:
                labels.extend(group['labels'])
                directions.extend([group['direction']] * len(indices))
                queries.extend([group['query_id']] * len(indices))
                offsets.append(len(labels))
            scores = [None] * len(labels)
            for owner in range(world):
                packets, _ = prepare_encoder_batches(reference, batch[:size], offsets,
                    rank=owner, world=world, device='cpu', groups_per_forward=3, token_budget=budget)
                for packet in packets:
                    torch.manual_seed(123 + packet.seed_micro * world + owner)
                    values = reference(**packet.kwargs)
                    for position, score in zip(packet.positions.tolist(), values):
                        scores[position] = score
            expected, _ = ranking_loss(torch.stack(scores), labels, directions, queries, objective=objective)
            expected.backward()
            torch.testing.assert_close(result['loss'], expected)
            for p, ref in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(p.grad, ref.grad, atol=3e-6, rtol=3e-5)
            assert result['replay_max_error'] == 0
if rank == 0:
    open(sys.argv[1], 'w').write('passed')
dist.destroy_process_group()
'''


def test_grouped_ddp_matches_full_batch_with_dropout_uneven_packing_and_empty_rank(tmp_path):
    worker = tmp_path / 'worker.py'
    worker.write_text(WORKER)
    root = Path(__file__).resolve().parents[2]
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1',
           'PYTHONPATH': str(root) + os.pathsep + os.environ.get('PYTHONPATH', '')}
    result = subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone',
        '--nproc_per_node=2', str(worker), str(tmp_path / 'result')], env=env,
        text=True, capture_output=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / 'result').read_text() == 'passed'
