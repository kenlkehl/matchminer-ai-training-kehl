"""Exhaustive single-vector retrieval and shared-label comparison with ColBERT."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

from colbert_experiments.contracts import DIRECTIONS, TASKS, contract_hash
from colbert_experiments.data import digest, load_corpus, read_jsonl, write_json, write_jsonl
from colbert_experiments.regression import score_contract
from colbert_experiments.retrieval import merge_candidates, top_indices
from colbert_experiments.score_publication import ready_scores
from finetune_embedder import load_text_model
from trial_cohort_format import file_stamp
from retrieval_workspace import add_retrieval_storage_args, run_retrieval
from trialspace_prompt_profiles import model_prompts

VERSION = 'embeddinggemma-colbert-exhaustive-comparison-v1'


def freeze(path, value):
    path = Path(path)
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f'Comparison inputs/settings changed: {path}')
    write_json(path, value)


def finish(output, names, **metadata):
    write_json(Path(output) / 'complete.json', dict(**metadata,
        outputs={name: file_stamp(Path(output) / name) for name in names}))


def completed(output):
    path = Path(output) / 'complete.json'
    if not path.exists():
        return False
    for name, expected in json.loads(path.read_text())['outputs'].items():
        if file_stamp(Path(output) / name) != expected:
            raise ValueError('Completed comparison output changed: ' + name)
    return True


def source_identity():
    # Source files are small; content hashes survive checkouts that only touch mtimes.
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),
            Path(__file__).with_name('finetune_embedder.py'),
            Path(__file__).with_name('trialspace_prompt_profiles.py'),
            Path(__file__).with_name('retrieval_workspace.py'),
            Path(__file__).with_name('synthetic_pool.py'),
            Path(__file__).with_name('retrieval_cache_import.py'))}


def encode_corpus(args, patients, cohorts):
    output = Path(args.output)
    model = None
    vectors = {}
    for name, rows, prompt in [('patients', patients, 'patient_summary'), ('cohorts', cohorts, 'this_space')]:
        path = output / (name + '.npy')
        if not path.exists():
            if model is None:
                model = load_text_model(args.model, args.device, prompt_profile='colbert_matching',
                    max_tokens=args.max_tokens, revision=args.revision, cache_dir=args.cache_dir)
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16,
                    enabled=str(args.device).startswith('cuda')):
                values = model.encode([row['text'] for row in rows], prompt_name=prompt,
                    batch_size=args.encode_batch_size, normalize_embeddings=True,
                    convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
            with path.with_suffix('.tmp').open('wb') as stream:
                np.save(stream, values, allow_pickle=False)
            path.with_suffix('.tmp').replace(path)
        values = np.load(path, mmap_mode='r', allow_pickle=False)
        if values.ndim != 2 or len(values) != len(rows) or not np.isfinite(values).all():
            raise ValueError('Invalid saved dense embeddings: ' + str(path))
        if not np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-3):
            raise ValueError('Dense embeddings must be normalized')
        vectors[name] = values
    return vectors


def rank_dense(patients, cohorts, vectors, *, device='cpu', score_batch_size=128,
               patient_k=20, cohort_k=40, lexical_k=5, random_k=5, seed=42):
    """Every query searches the complete synthetic denominator."""
    from synthetic_pool import require_full_pool
    require_full_pool(patients, cohorts)
    hits = []
    for direction in DIRECTIONS:
        qkind, dkind = (('patients', 'cohorts') if direction == 'patient_to_cohort'
                        else ('cohorts', 'patients'))
        queries, documents = (patients, cohorts) if qkind == 'patients' else (cohorts, patients)
        k = patient_k if qkind == 'patients' else cohort_k
        for split in ('train',):
            qi = [i for i, row in enumerate(queries) if row['split'] == split]
            di = [i for i, row in enumerate(documents) if row['split'] == split]
            if not qi or not di:
                continue
            targets = [documents[i] for i in di]
            refs = torch.tensor(np.asarray(vectors[dkind][di]), device=device)
            lexical = TfidfVectorizer(ngram_range=(1, 2), lowercase=True) if lexical_k else None
            matrix = lexical.fit_transform(row['text'] for row in targets) if lexical else None
            for start in range(0, len(qi), score_batch_size):
                chosen = qi[start:start + score_batch_size]
                q = torch.tensor(np.asarray(vectors[qkind][chosen]), device=device)
                scores = (q @ refs.T).cpu().numpy()
                lex = ((lexical.transform([queries[i]['text'] for i in chosen]) @ matrix.T).toarray()
                       if lexical else None)
                for index, (i, values) in enumerate(zip(chosen, scores, strict=True)):
                    row = queries[i]
                    def add(j, source, rank=None, score=None):
                        p, c = (row, targets[j]) if qkind == 'patients' else (targets[j], row)
                        hits.append(dict(patient_id=p['id'], cohort_id=c['id'], direction=direction,
                                         source=source, task='gated_score', rank=rank, score=score))
                    for rank, j in enumerate(top_indices(values, k), 1):
                        add(j, 'embeddinggemma2', rank, float(values[j]))
                    if lexical:
                        for rank, j in enumerate(top_indices(lex[index], lexical_k), 1):
                            add(j, 'tfidf_baseline', rank, float(lex[index, j]))
                    rng = random.Random(f'{seed}:{direction}:{row["id"]}')
                    for j in rng.sample(range(len(targets)), min(random_k, len(targets))):
                        add(j, 'random_exploration')
                print(f'[dense mine] {direction}/{split}: {start + len(chosen)}/{len(qi)}', flush=True)
    return hits


def dense_mine(args):
    from synthetic_pool import require_full_corpus
    require_full_corpus(args.corpus)
    return run_retrieval(args, _dense_mine)


def _dense_mine(args):
    from trialspace_training import export_candidates
    patients, cohorts = load_corpus(args.corpus)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    signature = dict(version=VERSION, sources=source_identity(),
        patients=digest(patients), cohorts=digest(cohorts), contract=score_contract(),
        model=args.model, revision=args.revision, prompts=model_prompts('colbert_matching'),
        max_tokens=args.max_tokens, dimension=768, scope='exhaustive full synthetic corpus',
        patient_k=args.patient_k, cohort_k=args.cohort_k, lexical_k=args.lexical_k,
        random_k=args.random_k, seed=args.seed)
    model_path = Path(args.model)
    if model_path.is_dir():
        # A mutable local model path must never make old encodings look reusable.
        signature['model_files'] = {str(p.relative_to(model_path)): file_stamp(p)
            for p in sorted(model_path.rglob('*')) if p.is_file()}
    from retrieval_cache_import import validate_encoding_import
    validate_encoding_import(output, signature)
    freeze(output / 'manifest.json', signature)
    if completed(output):
        return
    vectors = encode_corpus(args, patients, cohorts)
    hits = rank_dense(patients, cohorts, vectors, device=args.device,
                     score_batch_size=args.score_batch_size, patient_k=args.patient_k,
                     cohort_k=args.cohort_k, lexical_k=args.lexical_k, random_k=args.random_k, seed=args.seed)
    lineage = Path(args.corpus) / 'source_trial_pairs.jsonl'
    if lineage.exists():
        hits.extend(dict(patient_id=p['patient_id'], cohort_id=p['cohort_id'], direction='both',
                         source='source_trial_or_independent_proposal', task=None) for p in read_jsonl(lineage))
    pairs = merge_candidates(patients, cohorts, hits)
    write_jsonl(output / 'candidates.jsonl', pairs)
    write_jsonl(output / 'rankings.jsonl', [h for h in hits if h['source'] == 'embeddinggemma2'])
    export_candidates(args.corpus, output / 'candidates.jsonl', output / 'patient_to_cohort.parquet',
                      output / 'cohort_to_patient.parquet')
    finish(output, ['candidates.jsonl', 'rankings.jsonl', 'patient_to_cohort.parquet',
                   'cohort_to_patient.parquet', 'patients.npy', 'cohorts.npy'], pairs=len(pairs),
           candidate_scope='exhaustive cosine similarity over the full synthetic corpus')


def union_candidates(corpus, sources, output):
    patients, cohorts = load_corpus(corpus)
    output = Path(output)
    freeze(output / 'manifest.json', dict(version=VERSION, patients=digest(patients),
        cohorts=digest(cohorts), contract=score_contract(),
        sources={name: file_stamp(Path(path) / 'candidates.jsonl') for name, path in sources.items()}))
    if completed(output):
        return
    def hits():
        for name, path in sources.items():
            for pair in read_jsonl(Path(path) / 'candidates.jsonl'):
                for hit in pair['sources']:
                    yield dict(hit, patient_id=pair['patient_id'], cohort_id=pair['cohort_id'], retriever=name)
    pairs = merge_candidates(patients, cohorts, hits())
    write_jsonl(output / 'candidates.jsonl', pairs)
    finish(output, ['candidates.jsonl'], pairs=len(pairs))


def project_labels(corpus, union, labels, sources, output, *, reasoning_source=None, source_origins=None):
    """One teacher judgment; independent self-contained pair tables per retriever."""
    patients, cohorts = load_corpus(corpus)
    union, labels, output = Path(union), Path(labels), Path(output)
    reasoning_source = reasoning_source or str(labels / 'responses')
    source_origins = source_origins or {}
    count = json.loads((union / 'complete.json').read_text())['pairs']
    if ready_scores(labels, corpus=corpus, candidates=union / 'candidates.jsonl', expected_pairs=count) is None:
        raise ValueError('The candidate set must be completely labeled before projection or training')
    scores = {(s['patient_id'], s['cohort_id']): s for s in read_jsonl(labels / 'scores.jsonl')}
    pmap, cmap = {p['id']: p for p in patients}, {c['id']: c for c in cohorts}
    contract, question_contract = score_contract(), contract_hash()
    result = {}
    for name, source in sources.items():
        target = output / name
        pairs = read_jsonl(Path(source) / 'candidates.jsonl')
        freeze(target / 'manifest.json', dict(version=VERSION, source=str(source_origins.get(name, source)),
            candidates_sha256=digest(pairs), union_scores_stamp=file_stamp(labels / 'scores.jsonl'),
            contract=contract, shared_reasoning=reasoning_source))
        if completed(target):
            result[name] = json.loads((target / 'complete.json').read_text())
            continue
        projected, rows = [], []
        for pair in pairs:
            p, c = pmap[pair['patient_id']], cmap[pair['cohort_id']]
            score = scores[p['id'], c['id']]
            if (score['contract'] != contract or score['question_contract'] != question_contract
                    or score['split'] != p['split'] or p['split'] != c['split']
                    or pair['split'] != p['split'] or type(score['score']) is not int
                    or not 0 <= score['score'] <= 5):
                raise ValueError('Invalid/cross-split shared label')
            projected.append(score)
            rows.append(dict(patient_id=p['id'], cohort_id=c['id'], cohort_trial_id=c['id'],
                nct_id=c['trial_id'], patient_summary=p['text'], this_space=c['text'], split=p['split'],
                patient_source_split=p.get('source_split', p['split']),
                cohort_source_split=c.get('source_split', c['split']),
                eligibility_result=score['score'], normalized_score=score['normalized_score'],
                answers=json.dumps(score['answers']), question_keys=json.dumps(score['question_keys']),
                question_contract=score['question_contract'], score_contract=score['contract'],
                retrieval_sources=json.dumps(pair['sources']),
                patient_to_cohort=any(h['direction'] in ('patient_to_cohort', 'both') for h in pair['sources']),
                cohort_to_patient=any(h['direction'] in ('cohort_to_patient', 'both') for h in pair['sources'])))
        frame = pd.DataFrame(rows)
        for key in ['pairs', *DIRECTIONS]:
            chosen = frame if key == 'pairs' else frame.loc[frame[key]]
            path = target / (key + '.parquet')
            chosen.to_parquet(path.with_suffix('.tmp'), index=False, compression='zstd',
                              dictionary_pagesize_limit=256 * 1024**2)
            path.with_suffix('.tmp').replace(path)
        write_jsonl(target / 'scores.jsonl', projected)
        finish(target, ['pairs.parquet', 'patient_to_cohort.parquet', 'cohort_to_patient.parquet', 'scores.jsonl'],
               pairs=len(rows), by_split=frame.split.value_counts().to_dict())
        result[name] = json.loads((target / 'complete.json').read_text())
    write_json(output / 'complete.json', dict(models=result, overlapping_pairs_duplicated=len(sources) > 1,
        shared_label_contract=contract, shared_reasoning=reasoning_source))


def compare_rankings(corpus, sources, labels, output):
    """Describe each saved top-k list using the identical shared teacher labels."""
    score_rows = read_jsonl(Path(labels) / 'scores.jsonl')
    scores = {(s['patient_id'], s['cohort_id']): s for s in score_rows}
    reports, selected = {}, {}
    for name, path in sources.items():
        groups = defaultdict(list)
        for hit in read_jsonl(Path(path) / 'rankings.jsonl'):
            score = scores[hit['patient_id'], hit['cohort_id']]
            groups[hit['direction'], score['split']].append((hit, score))
        reports[name] = {}
        for (direction, split), rows in groups.items():
            key = direction + '/' + split
            values = np.array([s['score'] for h, s in rows])
            reports[name][key] = dict(hits=len(rows), mean_teacher_score=float(values.mean()),
                potentially_eligible_fraction=float((values >= 1).mean()),
                score_at_least_3_fraction=float((values >= 3).mean()))
            selected[name, key] = {(h['patient_id'], h['cohort_id']) for h, s in rows}
    names = list(sources)
    overlap = {}
    if len(names) == 2:
        for key in reports[names[0]]:
            a, b = selected[names[0], key], selected.get((names[1], key), set())
            overlap[key] = dict(shared=len(a & b), union=len(a | b), jaccard=len(a & b) / max(1, len(a | b)))
    write_json(output, dict(version=VERSION, contract=score_contract(), models=reports, overlap=overlap,
        interpretation='Teacher-assessed precision/specificity of exhaustive top-k; no claim of full-corpus recall.'))


def evaluate_dense(args):
    """Before/after in-sample diagnostics; no synthetic holdout exists."""
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    patients, cohorts = load_corpus(args.corpus)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model)
    freeze(output / 'manifest.json', dict(version=VERSION, patients=digest(patients), cohorts=digest(cohorts),
        scores=file_stamp(Path(args.labels) / 'scores.jsonl'), max_tokens=args.max_tokens,
        prompts=model_prompts('colbert_matching'),
        trained_model={str(p.relative_to(model_path)): file_stamp(p) for p in sorted(model_path.rglob('*'))
                       if p.is_file() and p.suffix in ('.json', '.safetensors')}))
    vectors = encode_corpus(args, patients, cohorts)
    base = {kind: np.load(Path(args.embeddinggemma) / (kind + '.npy'), mmap_mode='r')
            for kind in ('patients', 'cohorts')}
    pi, ci = {p['id']: i for i, p in enumerate(patients)}, {c['id']: i for i, c in enumerate(cohorts)}
    from synthetic_pool import require_full_pool
    require_full_pool(patients, cohorts)
    groups = {'all_synthetic': read_jsonl(Path(args.labels) / 'scores.jsonl')}
    results = {}
    for split, rows in groups.items():
        p = np.array([pi[row['patient_id']] for row in rows])
        c = np.array([ci[row['cohort_id']] for row in rows])
        y = np.array([row['score'] for row in rows])
        results[split] = dict(pairs=len(rows), models={})
        for name, values in [('base', base), ('trained', vectors)]:
            # Bound intermediate allocations for large candidate unions.
            prediction = np.concatenate([np.einsum('ij,ij->i', values['patients'][p[i:i+8192]],
                values['cohorts'][c[i:i+8192]]) for i in range(0, len(p), 8192)])
            rho = float(spearmanr(y, prediction).statistic)
            aucs = {str(t): (float(roc_auc_score(y >= t, prediction)) if len(np.unique(y >= t)) == 2 else None)
                    for t in (1, 3)}
            results[split]['models'][name] = dict(spearman=rho if np.isfinite(rho) else None,
                auroc_score_at_least=aucs)
    write_json(output / 'metrics.json', dict(results=results, contract=score_contract(),
        held_out=False, labels=str(args.labels), interpretation='In-sample synthetic labeled-pair diagnostics; selection-conditioned AUROC, not generalization or full-corpus recall.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['dense-mine', 'union', 'project-labels', 'project-dense-labels', 'report', 'evaluate-dense'])
    add_retrieval_storage_args(parser)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='google/embeddinggemma-2')
    parser.add_argument('--revision')
    parser.add_argument('--cache-dir')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-tokens', type=int, default=8192)
    parser.add_argument('--encode-batch-size', type=int, default=32)
    parser.add_argument('--score-batch-size', type=int, default=128)
    parser.add_argument('--patient-k', type=int, default=20)
    parser.add_argument('--cohort-k', type=int, default=40)
    parser.add_argument('--lexical-k', type=int, default=5)
    parser.add_argument('--random-k', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--embeddinggemma', type=Path)
    parser.add_argument('--colbert', type=Path)
    parser.add_argument('--union', type=Path)
    parser.add_argument('--labels', type=Path)
    args = parser.parse_args()
    if min(args.encode_batch_size, args.score_batch_size, args.patient_k, args.cohort_k) < 1:
        parser.error('Batch sizes and top-k counts must be positive')
    if args.command == 'dense-mine':
        dense_mine(args)
        return
    if args.command == 'evaluate-dense':
        evaluate_dense(args)
        return
    if args.command == 'project-dense-labels':
        if not args.embeddinggemma or not args.labels:
            parser.error('EmbeddingGemma candidates and their labels are required')
        project_labels(args.corpus, args.embeddinggemma, args.labels,
                       {'embeddinggemma2': args.embeddinggemma}, args.output)
        return
    if not args.embeddinggemma or not args.colbert:
        parser.error('Both retriever output directories are required')
    sources = dict(embeddinggemma2=args.embeddinggemma, colbert_modernbert=args.colbert)
    if args.command == 'union':
        union_candidates(args.corpus, sources, args.output)
    elif args.command == 'project-labels':
        project_labels(args.corpus, args.union, args.labels, dict(sources, union=args.union), args.output)
    else:
        compare_rankings(args.corpus, sources, args.labels, args.output)


if __name__ == '__main__':
    main()
