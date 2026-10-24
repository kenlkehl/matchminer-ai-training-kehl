"""Exact reference retrieval, independent candidate sources, and per-question union."""
from collections import defaultdict
import json
from pathlib import Path
import random

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

from .contracts import DIRECTIONS, TASKS, contract_hash, query_text
from .data import digest, load_corpus, read_jsonl, write_json, write_jsonl
from .model import maxsim


def top_indices(scores, count):
    return sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))[:count]


def proposal_indices(scores, limit, rng, strategy='random'):
    """Bound expensive MaxSim while retaining lexical and random coverage."""
    if strategy not in ('random', 'lexical_random'):
        raise ValueError('Unknown proposal strategy')
    indices = list(range(len(scores)))
    if limit is None or limit >= len(indices):
        return indices
    if limit < 1:
        raise ValueError('Proposal limit must be positive')
    if strategy == 'random':
        return sorted(rng.sample(indices, limit))
    selected = set(top_indices(scores, (limit + 1) // 2))
    selected.update(rng.sample([i for i in indices if i not in selected], limit - len(selected)))
    return sorted(selected)


def merge_candidates(patients, cohorts, hits):
    """One pair across every question/source/direction; preserve every provenance hit."""
    patients, cohorts = {p['id']: p for p in patients}, {c['id']: c for c in cohorts}
    pairs = {}
    for hit in hits:
        pid, cid = hit['patient_id'], hit['cohort_id']
        if patients[pid]['split'] != cohorts[cid]['split']:
            raise ValueError('Cross-split candidate')
        key = (pid, cid)
        if key not in pairs:
            pairs[key] = dict(patient_id=pid, cohort_id=cid, trial_id=cohorts[cid]['trial_id'],
                              split=patients[pid]['split'], sources=[], tasks=list(TASKS))
        pairs[key]['sources'].append({k: v for k, v in hit.items() if k not in ('patient_id', 'cohort_id')})
    return list(pairs.values())


@torch.inference_mode()
def mine(model, corpus, output, *, top_k=10, lexical_k=5, random_k=5, seed=42,
         splits=('train', 'val'), candidate_limit=None, extra_pairs=None,
         proposal_strategy='random', cohort_query_limit=None, document_batch_size=16, device=None,
         objective='binary'):
    """Exact MaxSim over the selected pool; no hidden ANN recall approximation.

candidate_limit explicitly samples a proposal pool for expensive experiments.
Independent lexical/random proposals are still selected from the whole split.
Default None performs an exact full-corpus reference search.
"""
    patients, cohorts = load_corpus(corpus)
    if objective not in ('binary', 'gated_regression'):
        raise ValueError('Unknown retrieval objective')
    regression = objective == 'gated_regression'
    tasks = ['gated_score'] if regression else list(TASKS)
    if regression:
        from .regression import regression_query, score_contract
    folder = getattr(model, 'folder', None)
    if folder is not None and (Path(folder) / 'regression_scoring.json').exists():
        saved = json.loads((Path(folder) / 'regression_scoring.json').read_text())
        if not regression or saved['contract'] != score_contract():
            raise ValueError('Saved regression model requires its matching gated_regression mining contract')
    if min(top_k, lexical_k, random_k) < 0 or (candidate_limit is not None and candidate_limit < 1):
        raise ValueError('Candidate counts must be nonnegative; a proposal limit must be positive')
    if document_batch_size < 1:
        raise ValueError('Document batch size must be positive')
    output = Path(output)
    signature = dict(contract=contract_hash(), patients=digest(patients), cohorts=digest(cohorts),
                     model=model.fingerprint(), top_k=top_k, lexical_k=lexical_k, random_k=random_k,
                     seed=seed, splits=list(splits), candidate_limit=candidate_limit,
                     proposal_strategy=proposal_strategy, cohort_query_limit=cohort_query_limit,
                     document_batch_size=document_batch_size,
                     extra_pairs=digest(read_jsonl(extra_pairs)) if extra_pairs else None)
    if regression:
        signature.update(contract=score_contract(), objective=objective)
    manifest_path = output / 'mining_manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != signature:
            raise ValueError('Mining identity changed; use a new output directory')
        if (output / 'status.json').exists() and json.loads((output / 'status.json').read_text()).get('state') == 'complete':
            return read_jsonl(output / 'candidates.jsonl')
    write_json(manifest_path, signature)
    if device is not None:
        model.to(device)
    vectors = {}
    for name, records in [('patients', patients), ('cohorts', cohorts)]:
        path = output / f'{name}_documents.pt'
        if path.exists():
            vectors[name] = torch.load(path, map_location='cpu', weights_only=True)
        else:
            vectors[name] = {}
            selected = sorted((r for r in records if r['split'] in splits), key=lambda r: len(r['text']))
            for index in range(0, len(selected), document_batch_size):
                batch = selected[index:index + document_batch_size]
                embeddings = model.encode_batch([r['text'] for r in batch], is_query=False)
                vectors[name].update((row['id'], embedding) for row, embedding in zip(batch, embeddings))
                if index % (10 * document_batch_size) == 0 or index + len(batch) == len(selected):
                    print(f'[encode] {name} {index + len(batch)}/{len(selected)}', flush=True)
            temporary = path.with_suffix('.tmp')
            torch.save(vectors[name], temporary)
            temporary.replace(path)
    hits, rankings = [], []
    for direction in DIRECTIONS:
        queries, documents = (patients, cohorts) if direction == 'patient_to_cohort' else (cohorts, patients)
        doc_vectors = vectors['cohorts' if direction == 'patient_to_cohort' else 'patients']
        for split in splits:
            targets = [d for d in documents if d['split'] == split]
            if not targets:
                continue
            lexical = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
            matrix = lexical.fit_transform(d['text'] for d in targets)
            selected_queries = [r for r in queries if r['split'] == split]
            if direction == 'cohort_to_patient' and cohort_query_limit is not None:
                if cohort_query_limit < 1:
                    raise ValueError('Cohort query limit must be positive')
                rng = random.Random(f'{seed}:{split}:cohort_queries')
                selected_queries = rng.sample(selected_queries, min(cohort_query_limit, len(selected_queries)))
            for row in selected_queries:
                rng = random.Random(f'{seed}:{direction}:{row["id"]}')
                lexical_scores = (lexical.transform([row['text']]) @ matrix.T).toarray()[0]
                proposals = proposal_indices(lexical_scores, candidate_limit, rng, proposal_strategy)

                def add(index, source, task=None, rank=None, score=None):
                    pid, cid = (row['id'], targets[index]['id']) if direction == 'patient_to_cohort' else (targets[index]['id'], row['id'])
                    hit = dict(patient_id=pid, cohort_id=cid, direction=direction, source=source,
                               task=task, rank=rank, score=score)
                    hits.append(hit)
                    if source == 'colbert':
                        rankings.append(hit)

                texts = ([regression_query(direction, row['text'])] if regression else
                         [query_text(task, direction, row['text']) for task in tasks])
                query_vectors = model.encode_batch(texts, is_query=True)
                for task, query_vector in zip(tasks, query_vectors):
                    q = query_vector.to(model.device)
                    scores = [float(maxsim(q, doc_vectors[targets[i]['id']].to(model.device))) for i in proposals]
                    for rank, offset in enumerate(top_indices(scores, top_k), 1):
                        add(proposals[offset], 'colbert', task, rank, scores[offset])
                for rank, index in enumerate(top_indices(lexical_scores, lexical_k), 1):
                    add(index, 'tfidf_baseline', rank=rank, score=float(lexical_scores[index]))
                for index in rng.sample(range(len(targets)), min(random_k, len(targets))):
                    add(index, 'random_exploration')
                print(f'[mine] {direction} {split} {row["id"]}', flush=True)
    if extra_pairs:
        for pair in read_jsonl(extra_pairs):
            if pair['split'] in splits:
                hits.append(dict(patient_id=pair['patient_id'], cohort_id=pair['cohort_id'],
                                 direction='both', source='provided_challenge_or_baseline', task=None))
    pairs = merge_candidates(patients, cohorts, hits)
    write_jsonl(output / 'candidates.jsonl', pairs)
    write_jsonl(output / 'rankings.jsonl', rankings)
    budget = (dict(teacher_questions_min=len(pairs), teacher_questions_max=len(pairs) * len(TASKS))
              if regression else dict(teacher_questions=len(pairs) * len(TASKS)))
    write_json(output / 'status.json', dict(state='complete', pairs=len(pairs), **budget,
                                          model=model.fingerprint(), exact_pool=candidate_limit is None))
    return pairs


def evaluate_rankings(rankings, labels, *, k=10):
    """Recall denominator uses all supplied judgments, not just retrieved positives."""
    truth = {}
    expected_contract = contract_hash()
    for row in labels:
        if row['task'] not in TASKS or row.get('contract', expected_contract) != expected_contract:
            raise ValueError('Evaluation labels use a different task/prompt contract; regenerate labels')
        key = (row['patient_id'], row['cohort_id'], row['task'])
        answer = row['answer'] == 'yes'
        if key in truth and truth[key] != answer:
            raise ValueError('Conflicting evaluation labels')
        truth[key] = answer
    reports = {}
    for direction in DIRECTIONS:
        query_key = 'patient_id' if direction == 'patient_to_cohort' else 'cohort_id'
        document_key = 'cohort_id' if direction == 'patient_to_cohort' else 'patient_id'
        for task in TASKS:
            relevant, retrieved = defaultdict(set), defaultdict(list)
            judged_queries = {row[query_key] for row in labels if row['task'] == task}
            for row in labels:
                if row['task'] == task and row['answer'] == 'yes':
                    relevant[row[query_key]].add(row[document_key])
            for row in rankings:
                if (row['direction'] == direction and row['task'] == task and row['rank'] <= k
                        and row[query_key] in judged_queries):
                    retrieved[row[query_key]].append(row)
            recalls, precisions, unjudged = [], [], 0
            for query in judged_queries:
                rows = retrieved[query]
                positives = relevant[query]
                hit = sum(truth.get((r['patient_id'], r['cohort_id'], task)) is True for r in rows)
                unjudged += sum((r['patient_id'], r['cohort_id'], task) not in truth for r in rows)
                if positives:
                    recalls.append(hit / len(positives))
                precisions.append(hit / len(rows) if rows else 0.)
            reports[f'{direction}/{task}'] = dict(recall_at_k=float(np.mean(recalls)) if recalls else None,
                precision_at_k=float(np.mean(precisions)) if precisions else None,
                queries=len(judged_queries), unjudged_results=unjudged, k=k,
                recall_scope='supplied judgments only; unjudged positives may exist')
    return reports
