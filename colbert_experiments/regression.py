"""Gated five-question scores and a separate ColBERT regression contract."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F
from matchminer_ai.embedding.colbert_contract import INSTRUCTION, regression_query

from .contracts import TASKS, DIRECTIONS, contract_hash, judge_messages, parse_judgment, prompt_sources
from .data import digest, load_corpus, read_jsonl, write_json, write_jsonl, selected_splits

VERSION = 'cohort-colbert-gated-score-v1'
GATE = 'potentially_eligible'
SPECIFICITY = tuple(task for task in TASKS if task != GATE)


def score_contract(question_contract=None):
    return digest(dict(version=VERSION, questions=question_contract or contract_hash(), gate=GATE,
                       specificity=SPECIFICITY, score='number_of_yes_answers', normalization=5,
                       query_instruction=INSTRUCTION))


def aggregate_pair(pair, judgments, *, question_contract=None):
    """Skipped answers remain null, never manufactured negative judgments."""
    by_task = {}
    expected = question_contract or contract_hash()
    for row in judgments:
        if any(row[k] != pair[k] for k in ('patient_id', 'cohort_id', 'split')):
            raise ValueError('Judgments must belong to the same pair and split')
        if row['task'] not in TASKS or row.get('contract') != expected:
            raise ValueError('Question contract differs from the current rubric')
        parse_judgment(json.dumps({k: row[k] for k in ('answer', 'evidence_status', 'reason')}))
        if row['task'] in by_task and row['answer'] != by_task[row['task']]['answer']:
            raise ValueError('Conflicting duplicate judgments')
        by_task[row['task']] = row
    if GATE not in by_task:
        raise ValueError('Missing potential eligibility judgment; do not impute zero')
    asked = [GATE] + (list(SPECIFICITY) if by_task[GATE]['answer'] == 'yes' else [])
    if any(task not in by_task for task in asked):
        raise ValueError('Missing required specificity judgment; do not impute zero')
    answers = {task: by_task[task]['answer'] if task in asked else None for task in TASKS}
    score = sum(answer == 'yes' for answer in answers.values())
    return dict(patient_id=pair['patient_id'], cohort_id=pair['cohort_id'], split=pair['split'],
                score=score, normalized_score=score / 5, answers=answers, asked_tasks=asked,
                skipped_tasks=[task for task in TASKS if task not in asked],
                question_keys={task: by_task[task]['key'] for task in asked},
                question_contract=expected, contract=score_contract(expected))


async def gated_judgments(items, ask):
    """One question per call, with the gate completed before specificity calls."""
    items = {item['task']: item for item in items}
    gate = await ask(items[GATE])
    if gate is None:
        return None
    rows = [gate]
    if gate['answer'] == 'yes':
        for task in SPECIFICITY:
            row = await ask(items[task])
            if row is None:
                return None
            rows.append(row)
    return rows


def import_verified_judgments(corpus, candidates, teacher_dir, output):
    """Controlled objective comparison using unchanged, individually judged pairs."""
    patients, cohorts = load_corpus(corpus)
    pairs = read_jsonl(candidates)
    teacher_dir, output = Path(teacher_dir), Path(output)
    manifest = json.loads((teacher_dir / 'labeling_manifest.json').read_text())
    sources = prompt_sources()
    expected_contract = contract_hash(sources)
    if any(manifest[k] != expected for k, expected in dict(contract=expected_contract,
            patients=digest(patients), cohorts=digest(cohorts), pairs=digest(pairs)).items()):
        raise ValueError('Cached judgments have different questions, corpus, or candidate pairs')
    labels = read_jsonl(teacher_dir / 'labels.jsonl')
    by_pair = defaultdict(list)
    pmap, cmap = {p['id']: p for p in patients}, {c['id']: c for c in cohorts}
    for row in labels:
        p, c = pmap[row['patient_id']], cmap[row['cohort_id']]
        if p['split'] != c['split'] or p['split'] != row['split']:
            raise ValueError('Cached judgment crosses corpus splits')
        if row['prompt_sha256'] != digest(judge_messages(row['task'], p['text'], c['text'], sources=sources)):
            raise ValueError('Cached question does not match current complete clinical text')
        by_pair[p['id'], c['id']].append(row)
    scores, used, seen = [], {}, set()
    for pair in pairs:
        key = (pair['patient_id'], pair['cohort_id'])
        if key in seen:
            continue
        seen.add(key)
        score = aggregate_pair(pair, by_pair[key], question_contract=expected_contract)
        scores.append(score)
        for row in by_pair[key]:
            if row['task'] in score['asked_tasks']:
                used[row['key']] = row
    provenance = dict(contract=score_contract(expected_contract), source_teacher=str(teacher_dir.resolve()),
                      source_manifest=manifest, source_labels_sha256=digest(labels),
                      reused_question_answers=len(used), ignored_specificity_answers=len(labels) - len(used),
                      new_llm_calls=0, scores_sha256=digest(scores))
    prior = output / 'import_manifest.json'
    if prior.exists() and json.loads(prior.read_text()) != provenance:
        raise ValueError('Imported experiment identity changed')
    write_json(prior, provenance)
    write_jsonl(output / 'labels.jsonl', [used[k] for k in sorted(used)])
    write_jsonl(output / 'scores.jsonl', scores)
    report = score_balance(scores)
    write_json(output / 'score_balance.json', report)
    return report


def score_balance(scores):
    return dict(pairs=len(scores), distribution=dict(sorted(Counter(row['score'] for row in scores).items())),
                by_split={split: dict(sorted(Counter(row['score'] for row in scores if row['split'] == split).items()))
                          for split in sorted({row['split'] for row in scores})},
                required_question_answers=sum(len(row['asked_tasks']) for row in scores),
                skipped_specificity_calls=sum(len(row['skipped_tasks']) for row in scores))


def build_groups(patients, cohorts, scores, *, split='train'):
    included = selected_splits(split)
    pmap, cmap = {p['id']: p for p in patients}, {c['id']: c for c in cohorts}
    groups, seen = {}, {}
    question_contract = contract_hash()
    expected_contract = score_contract(question_contract)
    for row in scores:
        if row.get('contract') != expected_contract:
            raise ValueError('Regression score contract differs from this experiment')
        p, c = pmap[row['patient_id']], cmap[row['cohort_id']]
        if p['split'] != c['split'] or row['split'] != p['split']:
            raise ValueError('Regression label crosses corpus splits')
        answers = row['answers']
        asked = [GATE] + (list(SPECIFICITY) if answers.get(GATE) == 'yes' else [])
        if (set(answers) != set(TASKS) or answers.get(GATE) not in ('yes', 'no')
                or any(answers[t] not in ('yes', 'no') if t in asked else answers[t] is not None for t in TASKS)
                or type(row['score']) is not int
                or row['score'] != sum(v == 'yes' for v in answers.values())
                or row['normalized_score'] != row['score'] / 5
                or row['asked_tasks'] != asked
                or row['skipped_tasks'] != [t for t in TASKS if t not in asked]
                or row['question_contract'] != question_contract):
            raise ValueError('Score must be the gated sum of binary answers')
        key = (p['id'], c['id'])
        if key in seen:
            if seen[key] != row['score']:
                raise ValueError('Conflicting duplicate regression labels')
            continue
        seen[key] = row['score']
        if p['split'] not in included:
            continue
        for direction in DIRECTIONS:
            query, document = (p, c) if direction == 'patient_to_cohort' else (c, p)
            group = groups.setdefault((direction, query['id']), dict(direction=direction, task='gated_score',
                query_id=query['id'], query=regression_query(direction, query['text']), documents=[], labels=[]))
            group['documents'].append(document)
            group['labels'].append(row['score'])
    values = list(groups.values())
    return values, dict(total_groups=len(values), usable_groups=len(values),
                        groups_with_score_variation=sum(len(set(g['labels'])) > 1 for g in values),
                        strata={d + '/gated_score': sum(g['direction'] == d for g in values) for d in DIRECTIONS})


def prediction(scores, link='identity'):
    if link == 'identity':
        return scores.float()
    if link == 'sigmoid':
        return torch.sigmoid(20 * (scores.float() - .5))
    raise ValueError('Unknown regression prediction link')


def regression_loss(scores, labels, *, link='identity'):
    target = torch.as_tensor(labels, device=scores.device, dtype=torch.float32)
    if scores.shape != target.shape or not torch.isfinite(target).all() or not ((target >= 0) & (target <= 5)).all():
        raise ValueError('Regression scores need aligned targets in [0, 5]')
    return F.mse_loss(prediction(scores, link), target / 5)


def make_probes(groups, *, per_stratum=16, documents_per_class=4, seed=42):
    probes = []
    for direction in DIRECTIONS:
        rng = random.Random(f'{seed}:{direction}')
        available = sorted([g for g in groups if g['direction'] == direction], key=lambda g: g['query_id'])
        # Include both varying-label and constant-label groups; regression needs
        # the latter too. Selection is fixed independently of model scores.
        for group in rng.sample(available, min(per_stratum, len(available))):
            count = min(documents_per_class, len(group['documents']))
            indices = []
            if count >= 2 and len(set(group['labels'])) > 1:
                # Keep a genuine ordering comparison when one is available.
                for value in (min(group['labels']), max(group['labels'])):
                    indices.append(rng.choice([i for i, label in enumerate(group['labels']) if label == value]))
            indices += rng.sample([i for i in range(len(group['documents'])) if i not in indices], count - len(indices))
            probes.append(dict(stratum=direction + '/gated_score', query_id=group['query_id'], query=group['query'],
                               documents=[group['documents'][i] for i in indices], labels=[group['labels'][i] for i in indices]))
    return probes


def score_statistics(scores, labels, link='identity'):
    predicted = prediction(scores, link).detach().cpu().numpy() * 5
    raw, target = scores.detach().float().cpu().numpy(), np.asarray(labels, dtype=float)
    differing = target[:, None] > target[None, :]
    margins = (raw[:, None] - raw[None, :])[differing]
    # Linear graded gains, with tie-aware NDCG from sklearn.
    from sklearn.metrics import ndcg_score
    ndcg = float(ndcg_score(target[None], raw[None], k=10)) if len(target) > 1 and target.max() > 0 else None
    return dict(loss=float(regression_loss(scores, labels, link=link)),
                mae=float(np.abs(predicted - target).mean()), mse_score=float(((predicted - target) ** 2).mean()),
                pairwise_accuracy=float(((margins > 0) + .5 * (margins == 0)).mean()) if len(margins) else None,
                ndcg_at_10=ndcg, comparisons=int(len(margins)))


def summarize_probe_rows(rows):
    metrics = ('loss', 'mae', 'mse_score', 'pairwise_accuracy', 'ndcg_at_10')
    def summarize(values):
        result = {key: float(np.mean([r[key] for r in values if r[key] is not None]))
                  if any(r[key] is not None for r in values) else None for key in metrics}
        result['rmse'] = result['mse_score'] ** .5 if result['mse_score'] is not None else None
        return result
    strata = {name: dict(groups=len(values), **summarize(values))
              for name in sorted({r['stratum'] for r in rows}) if (values := [r for r in rows if r['stratum'] == name])}
    return dict(groups=len(rows), strata=strata, **summarize(list(strata.values())),
                ordering_groups=sum(r['pairwise_accuracy'] is not None for r in rows),
                ordering_comparisons=sum(r['comparisons'] for r in rows),
                ndcg_groups=sum(r['ndcg_at_10'] is not None for r in rows))


def constant_baselines(training_groups, probes):
    """A train-only uniform-query reference, before score-stratified sampling."""
    mean = float(np.mean([np.mean([np.mean(g['labels']) for g in training_groups if g['direction'] == direction])
                          for direction in DIRECTIONS if any(g['direction'] == direction for g in training_groups)]))
    result = dict(train_mean_score=mean, mean_weighting='equal directions, then equal query groups; '
                  'train labels only, before score-stratified batch sampling')
    for name, value in [('zero', 0.), ('train_mean', mean)]:
        result[name] = {split: summarize_probe_rows([
            dict(stratum=p['stratum'], **score_statistics(torch.full((len(p['labels']),), value / 5), p['labels']))
            for p in items]) for split, items in probes.items()}
    return result


@torch.inference_mode()
def evaluate_probes(model, probes, *, link='identity', rank=0, world=1, **unused):
    import torch.distributed as dist
    was_training = model.training
    model.eval()
    rows = []
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=model.device.type == 'cuda'):
        for probe in probes[rank::world]:
            scores = model(probe['query'], [d['text'] for d in probe['documents']])
            rows.append(dict(stratum=probe['stratum'], query_id=probe['query_id'], labels=probe['labels'],
                document_ids=[d['id'] for d in probe['documents']], scores=scores.cpu().tolist(),
                predictions=(5 * prediction(scores, link)).cpu().tolist(), **score_statistics(scores, probe['labels'], link)))
    model.train(was_training)
    if world > 1:
        gathered = [None] * world
        dist.all_gather_object(gathered, rows)
        rows = [row for values in gathered for row in values]
    return summarize_probe_rows(rows), rows
