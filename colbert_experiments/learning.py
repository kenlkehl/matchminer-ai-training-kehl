"""Fixed, explicitly labeled ranking probes for measuring learning over time."""
from collections import defaultdict
import random

import torch
import torch.distributed as dist

from .model import grouped_loss


def make_probes(groups, *, per_stratum=2, documents_per_class=2, seed=42):
    if per_stratum < 1 or documents_per_class < 1:
        raise ValueError('Probe sizes must be positive')
    buckets = defaultdict(list)
    for group in groups:
        buckets[group['direction'] + '/' + group['task']].append(group)
    probes = []
    for stratum, members in sorted(buckets.items()):
        rng = random.Random(f'{seed}:{stratum}')
        for group in rng.sample(sorted(members, key=lambda g: g['query_id']), min(per_stratum, len(members))):
            positives = rng.sample(group['positive'], min(documents_per_class, len(group['positive'])))
            negatives = rng.sample(group['negative'], min(documents_per_class, len(group['negative'])))
            probes.append(dict(stratum=stratum, query_id=group['query_id'], query=group['query'],
                documents=positives + negatives, labels=[1] * len(positives) + [0] * len(negatives)))
    return probes


def score_statistics(scores, labels, temperature):
    labels = torch.as_tensor(labels, device=scores.device)
    positive, negative = scores[labels == 1], scores[labels == 0]
    differences = positive[:, None] - negative[None, :]
    top = scores == scores.max()
    return dict(loss=float(grouped_loss(scores, labels, temperature)),
                pairwise_accuracy=float(((differences > 0).float() + .5 * (differences == 0).float()).mean()),
                margin=float(differences.mean()), precision_at_1=float(labels[top].float().mean()))


def summarize_scores(rows):
    buckets = defaultdict(list)
    for row in rows:
        buckets[row['stratum']].append(row)
    metrics = ('loss', 'pairwise_accuracy', 'margin', 'precision_at_1')
    by_stratum = {name: dict(groups=len(values), **{m: sum(r[m] for r in values) / len(values) for m in metrics})
                  for name, values in sorted(buckets.items())}
    return dict(groups=len(rows), strata=by_stratum,
                **{m: sum(r[m] for r in by_stratum.values()) / len(by_stratum) if by_stratum else None for m in metrics})


@torch.inference_mode()
def evaluate_probes(model, probes, *, temperature=.05, rank=0, world=1):
    was_training = model.training
    model.eval()
    rows = []
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=model.device.type == 'cuda'):
        for probe in probes[rank::world]:
            scores = model(probe['query'], [d['text'] for d in probe['documents']])
            rows.append(dict(stratum=probe['stratum'], query_id=probe['query_id'],
                document_ids=[d['id'] for d in probe['documents']], labels=probe['labels'],
                scores=scores.cpu().tolist(), **score_statistics(scores, probe['labels'], temperature)))
    model.train(was_training)
    if world > 1:
        gathered = [None] * world
        dist.all_gather_object(gathered, rows)
        rows = [r for part in gathered for r in part]
    return summarize_scores(rows), rows
