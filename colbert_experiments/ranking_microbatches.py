"""Pack several complete query groups without changing effective-batch membership."""
from dataclasses import dataclass

import torch


@dataclass
class EncoderBatch:
    kwargs: dict
    positions: torch.Tensor
    seed_micro: int


def prepare_encoder_batches(model, batch, offsets, *, rank, world, device,
                            groups_per_forward=1, token_budget=32768):
    if groups_per_forward < 1 or token_budget < 1:
        raise ValueError('Encoder group count and token budget must be positive')
    indices = list(range(rank, len(batch), world))

    def raw(index):
        group, selected = batch[index]
        return dict(query=group['query'], documents=[group['documents'][i]['text'] for i in selected])

    if groups_per_forward == 1:
        packed = [([index], raw(index), micro) for micro, index in enumerate(indices)]
        fallback = raw(0)
    else:
        if not hasattr(model, 'prepare_groups'):
            raise ValueError('Multi-query batches require the shared native ColBERT encoder')
        raw_groups = [raw(index) for index in indices or [0]]
        prepared = model.prepare_groups([g['query'] for g in raw_groups], [g['documents'] for g in raw_groups])
        fallback = dict(prepared_groups=prepared[:1])
        packed, pending, rows, first = [], [], [], 0
        for micro, (index, group) in enumerate(zip(indices, prepared)):
            if pending and (len(pending) >= groups_per_forward or
                            model.prepared_batch_tokens(rows + [group]) > token_budget):
                packed.append((pending, dict(prepared_groups=rows), first))
                pending, rows = [], []
            if not pending:
                first = micro
            pending.append(index)
            rows.append(group)
        if pending:
            packed.append((pending, dict(prepared_groups=rows), first))
    result = []
    for indices, kwargs, micro in packed:
        positions = [p for index in indices for p in range(offsets[index], offsets[index + 1])]
        result.append(EncoderBatch(kwargs, torch.tensor(positions, device=device, dtype=torch.long), micro))
    return result, fallback
