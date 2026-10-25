from collections import Counter

import pytest

from colbert_experiments.batching import RegressionBatchSampler
from colbert_experiments.contracts import DIRECTIONS


def group(direction, key, labels):
    return dict(direction=direction, query_id=key, query='Complete synthetic query',
                documents=[dict(id=f'{key}-{i}', text=f'Full synthetic document {i}')
                           for i in range(len(labels))], labels=labels)


def test_large_update_retains_all_available_scores_despite_severe_imbalance():
    groups = [group(direction, f'zero-{i}', [0] * 20)
              for direction in DIRECTIONS for i in range(1000)]
    groups += [group(direction, 'mixed', [0] * 100 + list(range(1, 6))) for direction in DIRECTIONS]
    sampler = RegressionBatchSampler(groups)
    batch = sampler.batch(query_groups=32 * 4, documents_per_query=8, seed=42, step=0)
    for direction in DIRECTIONS:
        observed = {g['labels'][i] for g, indices in batch if g['direction'] == direction for i in indices}
        assert observed == set(range(6))
    assert sum(len(indices) for _, indices in batch) == 1024
    for g, indices in batch:
        assert len(indices) == len(set(indices)) == 8
        if g['query_id'] == 'mixed':
            assert {g['labels'][i] for i in indices} == set(range(6))
    # Each DDP rank reconstructs the same update, then processes its own slice.
    shards = [batch[rank::4] for rank in range(4)]
    assert all(len(shard) == 32 for shard in shards)
    assert Counter((g['direction'], g['query_id'], tuple(indices)) for shard in shards for g, indices in shard) == Counter(
        (g['direction'], g['query_id'], tuple(indices)) for g, indices in batch)
    assert batch == sampler.batch(query_groups=128, documents_per_query=8, seed=42, step=0)
    assert batch != sampler.batch(query_groups=128, documents_per_query=8, seed=42, step=1)


def test_sparse_score_support_and_singletons_are_kept_without_invented_labels():
    groups = [group(direction, str(score), [score]) for direction in DIRECTIONS for score in (0, 3)]
    sampler = RegressionBatchSampler(groups)
    batch = sampler.batch(query_groups=32, documents_per_query=8, seed=1, step=5)
    assert len(batch) == 32
    assert Counter(g['labels'][i] for g, indices in batch for i in indices) == {0: 16, 3: 16}
    assert all(indices == [0] for _, indices in batch)
    with pytest.raises(ValueError, match='positive'):
        sampler.batch(query_groups=0, documents_per_query=8, seed=1, step=5)


def test_small_document_budget_still_contains_anchor_and_query_extremes():
    sampler = RegressionBatchSampler([group('patient_to_cohort', 'mixed', [0] * 99 + list(range(1, 6)))])
    batch = sampler.batch(query_groups=6, documents_per_query=3, seed=2, step=1)
    assert {g['labels'][i] for g, indices in batch for i in indices} == set(range(6))
    assert all({0, 5}.issubset({g['labels'][i] for i in indices}) for g, indices in batch)


def test_queries_are_not_repeated_when_unused_members_of_the_stratum_remain():
    sampler = RegressionBatchSampler([group('patient_to_cohort', str(i), [0, 5]) for i in range(32)])
    batch = sampler.batch(query_groups=32, documents_per_query=8, seed=2, step=1)
    assert len({g['query_id'] for g, _ in batch}) == 32


def test_accumulated_group_mse_matches_one_large_update_gradient():
    import torch
    from colbert_experiments.regression import regression_loss
    sampler = RegressionBatchSampler([group(direction, 'mixed', [0, 0, 1, 2, 3, 4, 5, 5])
                                      for direction in DIRECTIONS])
    batch = sampler.batch(query_groups=32, documents_per_query=8, seed=3, step=4)
    targets = torch.tensor([[g['labels'][i] for i in indices] for g, indices in batch])
    features = torch.arange(targets.numel(), dtype=torch.float32).reshape(targets.shape) / targets.numel()
    large = torch.tensor(.7, requires_grad=True)
    regression_loss((features * large).flatten(), targets.flatten()).backward()
    accumulated = torch.tensor(.7, requires_grad=True)
    for x, labels in zip(features, targets):
        (regression_loss(x * accumulated, labels) / len(batch)).backward()
    torch.testing.assert_close(accumulated.grad, large.grad)
