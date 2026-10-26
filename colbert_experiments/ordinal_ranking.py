"""CoSENT on MaxSim, with optional contrastive loss on known negatives only.

The two-pass backward compares the full effective batch without retaining every
encoder graph. It caches scalar scores, differentiates the ranking loss, then
replays each microbatch with the same random seed to apply the score gradients.
"""
from collections import defaultdict
from contextlib import nullcontext
import math
import time

import torch
import torch.distributed as dist


PAUSE_POLL_MICROBATCHES = 8


def cosent(scores, labels, *, scale=20.):
    """log(1 + sum exp(scale * (lower-label score - higher-label score)))."""
    labels = torch.as_tensor(labels, device=scores.device)
    differences = scale * (scores[:, None] - scores[None, :])
    ordered = differences[labels[:, None] < labels[None, :]]
    # Keep a differentiable zero even when every label is tied.
    return torch.logsumexp(torch.cat((scores.sum().reshape(1) * 0, ordered)), dim=0)


def ranking_loss(scores, labels, directions, query_ids, *, objective='legacy_ranking',
                 scale=20., contrastive_weight=1.):
    """Ordinal comparisons per direction; known-zero contrasts per actual query.

    This adapts the former CoSENT + MNRL recipe. It does NOT create unjudged
    cross-query negatives. Each positive competes only with explicitly scored
    zeros for its own query; other positives never compete with one another.
    """
    if objective not in ('cosent', 'legacy_ranking'):
        raise ValueError('Expected cosent or legacy_ranking')
    if scale <= 0 or contrastive_weight < 0 or not all(map(math.isfinite, (scale, contrastive_weight))):
        raise ValueError('Use finite positive scale and nonnegative contrastive weight')
    labels = torch.as_tensor(labels, device=scores.device)
    if scores.ndim != 1 or labels.shape != scores.shape or not len(scores):
        raise ValueError('Expected aligned nonempty vectors of scores and labels')
    if len(directions) != len(scores) or len(query_ids) != len(scores):
        raise ValueError('Each score needs a direction and query identity')
    zero = scores.sum() * 0
    direction_indices, queries = defaultdict(list), defaultdict(list)
    for i, (direction, query_id) in enumerate(zip(directions, query_ids)):
        direction_indices[direction].append(i)
        queries[(direction, query_id)].append(i)
    ordinal, comparisons = [], 0
    for indices in direction_indices.values():
        y = labels[indices]
        count = int((y[:, None] < y[None, :]).sum())
        comparisons += count
        if count:
            ordinal.append(cosent(scores[indices], y, scale=scale))
    ordinal = torch.stack(ordinal).mean() if ordinal else zero
    contrasts, query_count = [], 0
    if objective == 'legacy_ranking':
        for indices in queries.values():
            y, s = labels[indices], scores[indices]
            positives, negatives = s[y > 0], s[y == 0]
            if len(positives) and len(negatives):
                # -log softmax(p) over this positive and known negatives only.
                differences = scale * (negatives[None, :] - positives[:, None])
                contrasts.append(torch.logsumexp(torch.cat(
                    (torch.zeros_like(positives[:, None]), differences), dim=1), dim=1).mean())
                query_count += 1
    contrastive = torch.stack(contrasts).mean() if contrasts else zero
    return ordinal + contrastive_weight * contrastive, dict(
        cosent=ordinal, known_negative_contrastive=contrastive,
        ordinal_comparisons=comparisons, contrastive_queries=query_count)


def cached_ranking_backward(model, wrapped, batch, *, device, rank, world,
                            seed_for_micro, objective='legacy_ranking', scale=20.,
                            contrastive_weight=1., should_pause=lambda: False,
                            pause_poll_microbatches=PAUSE_POLL_MICROBATCHES):
    """Exact full-batch loss gradient, subject to deterministic forward replay.

    All ranks receive the same batch. The cache pass uses the unwrapped encoder
    (no DDP reducer); replay uses DDP with no_sync until the final microbatch.
    Dummy tail forwards carry zero gradient. Pause checks are collective, so
    poll every few microbatches rather than forcing the shorter-text rank to
    wait after every encoding. Both phase boundaries are checked as well.
    Callers own zero_grad/step/clipping.
    Returns None on a coordinated pause, otherwise detached loss diagnostics.
    """
    if pause_poll_microbatches < 1:
        raise ValueError('Pause polling interval must be positive')
    labels, directions, queries, offsets = [], [], [], [0]
    for group, indices in batch:
        labels.extend(group['labels'][i] for i in indices)
        directions.extend([group['direction']] * len(indices))
        queries.extend([group['query_id']] * len(indices))
        offsets.append(len(labels))
    cached = torch.zeros(len(labels), device=device, dtype=torch.float32)
    cache_started = time.monotonic()
    microbatches = math.ceil(len(batch) / world)
    for micro in range(microbatches):
        if micro % pause_poll_microbatches == 0 and should_pause():
            return None
        index = micro * world + rank
        if index < len(batch):
            group, indices = batch[index]
            torch.manual_seed(seed_for_micro(micro))
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                scores = model(group['query'], [group['documents'][i]['text'] for i in indices])
            cached[offsets[index]:offsets[index + 1]] = scores.float()
    if should_pause():
        return None
    if world > 1:
        dist.all_reduce(cached)  # Disjoint slots; no averaging or duplicate pairs.
    # This is a tiny vector of scalar scores, not token embeddings. Computing
    # its loss on CPU avoids many small GPU kernels and synchronizing boolean
    # index operations for each query. Only the scalar-score gradients return
    # to the encoder GPU; the full-batch objective is unchanged.
    loss_scores = cached.detach().cpu().requires_grad_()
    cache_seconds = time.monotonic() - cache_started
    loss_started = time.monotonic()
    if not torch.isfinite(loss_scores).all():
        raise ValueError('Nonfinite cached MaxSim score')
    loss, components = ranking_loss(loss_scores, labels, directions, queries,
        objective=objective, scale=scale, contrastive_weight=contrastive_weight)
    score_grad, = torch.autograd.grad(loss, loss_scores)
    score_grad = score_grad.to(device)
    loss_seconds = time.monotonic() - loss_started
    replay_started = time.monotonic()
    replay_error = torch.zeros((), device=device)
    for micro in range(microbatches):
        if micro % pause_poll_microbatches == 0 and should_pause():
            return None
        index = micro * world + rank
        real = index < len(batch)
        group, indices = batch[index] if real else batch[0]
        torch.manual_seed(seed_for_micro(micro))
        sync = wrapped.no_sync() if world > 1 and micro + 1 < microbatches else nullcontext()
        with sync, torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            prediction = wrapped(group['query'], [group['documents'][i]['text'] for i in indices]).float()
            if real:
                section = slice(offsets[index], offsets[index + 1])
                replay_error = torch.maximum(replay_error, (prediction.detach() - cached.detach()[section]).abs().max())
                # DDP averages parameter gradients, so undo its division.
                surrogate = (prediction * score_grad[section]).sum() * world
            else:
                surrogate = prediction.sum() * 0
            surrogate.backward()
    # A request arriving during the last interval still discards this update
    # before optimizer.step, and all ranks checkpoint the same completed step.
    if should_pause():
        return None
    if world > 1:
        dist.all_reduce(replay_error, op=dist.ReduceOp.MAX)
    # Do not silently use gradients of a different dropout/encoder realization.
    if not torch.isfinite(replay_error) or replay_error > 1e-5:
        raise ValueError(f'Cached and replayed scores differ: {float(replay_error):.6g}')
    return dict(loss=loss.detach(), **{k: v.detach() if isinstance(v, torch.Tensor) else v
                                     for k, v in components.items()}, replay_max_error=float(replay_error),
                cache_pass_seconds=cache_seconds, ranking_loss_seconds=loss_seconds,
                replay_backward_seconds=time.monotonic() - replay_started)
