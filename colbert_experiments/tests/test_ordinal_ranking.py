import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from colbert_experiments.ordinal_ranking import cosent, ranking_loss


def test_cosent_matches_sentence_transformers_value_and_gradient():
    from sentence_transformers import losses
    scores = torch.tensor([.3, .8, -.2, .1, .5], requires_grad=True)
    labels = torch.tensor([0, 5, 0, 2, 2])
    # Substitute an already-computed MaxSim vector for the default pairwise
    # cosine function, retaining the actual installed CoSENT implementation.
    official = losses.CoSENTLoss(None, similarity_fct=lambda x, y: x)
    expected = official.compute_loss_from_embeddings([scores, scores], labels)
    actual = cosent(scores, labels)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, scores, retain_graph=True)[0],
                               torch.autograd.grad(expected, scores)[0])
    # The old [-1, 1] label normalization does not change an ordinal objective.
    torch.testing.assert_close(cosent(scores, labels / 5 * 2 - 1), actual)
    torch.testing.assert_close(cosent(scores, labels ** 3), actual)


def test_ties_and_cross_direction_scores_never_create_ordering_constraints():
    scores = torch.tensor([.9, -.9, .2], requires_grad=True)
    loss, stats = ranking_loss(scores, [0, 5, 5], ['forward', 'reverse', 'reverse'],
                              ['q', 'q', 'q'], objective='legacy_ranking')
    assert loss.item() == 0 and stats['ordinal_comparisons'] == stats['contrastive_queries'] == 0
    assert torch.equal(torch.autograd.grad(loss, scores)[0], torch.zeros_like(scores))


def test_contrastive_term_uses_only_known_zeros_for_the_same_query():
    scores = torch.tensor([.2, .8, .4, .99, -.9, .99], requires_grad=True)
    labels = [0, 5, 2, 0, 0, 0]
    directions = ['forward'] * 5 + ['reverse']
    queries = ['q', 'q', 'q', 'other', 'other', 'q']
    loss, stats = ranking_loss(scores, labels, directions, queries)
    expected = torch.nn.functional.softplus(20 * (scores[0] - scores[1:3])).mean()
    torch.testing.assert_close(stats['known_negative_contrastive'], expected)
    assert stats['contrastive_queries'] == 1
    gradients, = torch.autograd.grad(stats['known_negative_contrastive'], scores)
    assert gradients[0] > 0 and (gradients[1:3] < 0).all()
    assert torch.equal(gradients[3:], torch.zeros(3))
    # Multiple positives are not implicitly negatives for one another.
    ordinal, components = ranking_loss(scores, labels, directions, queries, objective='cosent')
    torch.testing.assert_close(loss, ordinal + expected)
    assert components['known_negative_contrastive'] == 0


WORKER = r'''
import copy, json, os, sys, torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from colbert_experiments.ordinal_ranking import cached_ranking_backward, ranking_loss
rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
dist.init_process_group('gloo')
class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4)
        self.dropout = torch.nn.Dropout(.3)
        self.output = torch.nn.Linear(4, 1)
    def forward(self, query, documents):
        x = torch.tensor([[float(query), float(d), float(query) * float(d)] for d in documents])
        return self.output(self.dropout(self.linear(x).tanh())).squeeze(1)
torch.manual_seed(17)
model = Toy()
reference = copy.deepcopy(model)
wrapped = DDP(model)
batch = []
for i, (q, direction, labels) in enumerate([
    ('1', 'forward', [0, 5]), ('1', 'forward', [2]),
    ('2', 'reverse', [3, 0]), ('3', 'reverse', [0]), ('4', 'forward', [5])]):
    group = dict(query=q, query_id=q, direction=direction, labels=labels,
                 documents=[dict(text=str(i + j + 1)) for j in range(len(labels))])
    batch.append((group, list(range(len(labels)))))
results = []
for objective in ('cosent', 'legacy_ranking'):
    for size in (1, 3, 5):
        model.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
        result = cached_ranking_backward(model, wrapped, batch[:size], device=torch.device('cpu'),
            rank=rank, world=world, seed_for_micro=lambda micro: 123 + micro * world + rank,
            objective=objective)
        scores, labels, directions, queries = [], [], [], []
        for i, (group, indices) in enumerate(batch[:size]):
            torch.manual_seed(123 + i)
            scores.append(reference(group['query'], [group['documents'][j]['text'] for j in indices]))
            labels.extend(group['labels'])
            directions.extend([group['direction']] * len(indices))
            queries.extend([group['query_id']] * len(indices))
        expected, _ = ranking_loss(torch.cat(scores), labels, directions, queries, objective=objective)
        expected.backward()
        torch.testing.assert_close(result['loss'], expected)
        for p, ref in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(p.grad, ref.grad, atol=3e-6, rtol=3e-5)
        assert result['replay_max_error'] == 0
        results.append(dict(objective=objective, chunks=size, loss=float(expected.detach())))
# The same full batch must retain its gradient when collective pause polling
# is coarsened. Both ranks finish the same number of polls, including padding.
large_batch = batch * 7
gradients = []
for interval in (1, 8):
    model.zero_grad(set_to_none=True)
    polls = [0]
    def pause_probe():
        polls[0] += 1
        flag = torch.zeros((), dtype=torch.long)
        dist.all_reduce(flag)
        return False
    result = cached_ranking_backward(model, wrapped, large_batch, device=torch.device('cpu'),
        rank=rank, world=world, seed_for_micro=lambda micro: 123 + micro * world + rank,
        should_pause=pause_probe, pause_poll_microbatches=interval)
    assert polls[0] == (38 if interval == 1 else 8)
    gradients.append([p.grad.clone() for p in model.parameters()])
for before, after in zip(*gradients):
    torch.testing.assert_close(before, after)
if rank == 0:
    open(sys.argv[1], 'w').write(json.dumps(results))
dist.destroy_process_group()
'''


def test_two_rank_cached_gradients_equal_full_batch_with_dropout_and_uneven_tail(tmp_path):
    worker = tmp_path/'worker.py'
    worker.write_text(WORKER)
    root = Path(__file__).resolve().parents[2]
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1',
           'PYTHONPATH': str(root) + os.pathsep + os.environ.get('PYTHONPATH', '')}
    result = subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone',
        '--nproc_per_node=2', str(worker), str(tmp_path/'results.json')],
        env=env, text=True, capture_output=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    rows = json.loads((tmp_path/'results.json').read_text())
    assert len(rows) == 6 and all(r['loss'] > 0 for r in rows)
