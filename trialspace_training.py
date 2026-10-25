#!/usr/bin/env python3
"""Production TrialSpace: complete cohorts, gated question labels, ColBERT MSE.

The experiment and production entrypoints share the tested native ColBERT,
question contracts, adaptive teacher, and regression trainer. This entrypoint
fixes the production objective and adds resumable, distributed disk-backed mining.
"""
import asyncio
from collections import OrderedDict, defaultdict
import json
import os
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from sklearn.feature_extraction.text import TfidfVectorizer

from colbert_experiments.__main__ import parser as shared_parser
from colbert_experiments.contracts import DIRECTIONS
from colbert_experiments.data import digest, load_corpus, prepare, read_jsonl, write_json, write_jsonl
from colbert_experiments.model import ColBERT
from colbert_experiments.regression import regression_query, score_contract
from colbert_experiments.retrieval import merge_candidates, proposal_indices, top_indices
from trial_cohort_format import file_sha256

VERSION = 'trialspace-colbert-regression-v1'


def identity_file(path, identity):
    path = Path(path)
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError(f'Changed TrialSpace inputs/settings at {path}; use a new run directory')
    write_json(path, identity)


def prepare_corpus(patients, cohorts, output):
    output = Path(output)
    identity = dict(version=VERSION, patients=str(Path(patients).resolve()), cohorts=str(Path(cohorts).resolve()),
                    patient_sha256=file_sha256(patients), cohort_sha256=file_sha256(cohorts), contract=score_contract())
    identity_file(output / 'input_manifest.json', identity)
    complete = output / 'prepared.json'
    if complete.exists():
        saved = json.loads(complete.read_text())
        for name, expected in saved['output_sha256'].items():
            if file_sha256(output / name) != expected:
                raise ValueError('Prepared TrialSpace corpus changed')
        return saved
    prepare(patients, cohorts, output)
    ps, cs = load_corpus(output)
    by_trial = defaultdict(list)
    for cohort in cs:
        by_trial[cohort['trial_id']].append(cohort)
    lineage = [dict(patient_id=p['id'], cohort_id=c['id'], split=p['split'])
               for p in ps for c in by_trial.get(p.get('source_trial_id'), [])
               if p['split'] == c['split'] and p['split'] != 'test']
    write_jsonl(output / 'source_trial_pairs.jsonl', lineage)
    result = dict(version=VERSION, patients=len(ps), cohorts=len(cs), trials=len(by_trial),
                  source_trial_pairs=len(lineage), contract=score_contract(),
                  output_sha256={name: file_sha256(output / name) for name in
                                 ('patients.jsonl', 'cohorts.jsonl', 'source_trial_pairs.jsonl')})
    write_json(complete, result)
    return result


@torch.inference_mode()
def score_documents(query, documents, *, query_block=128, document_block=2048):
    """Batched exact mean-MaxSim; padding can never win a token maximum."""
    if not documents or any(len(d) == 0 for d in documents) or len(query) == 0:
        raise ValueError('Nonempty query and document token matrices are required')
    lengths = torch.tensor([len(d) for d in documents], device=query.device)
    padded = torch.nn.utils.rnn.pad_sequence([d.to(query.device, dtype=query.dtype) for d in documents], batch_first=True)
    result = torch.zeros(len(documents), device=query.device, dtype=torch.float32)
    for q in query.split(query_block):
        best = torch.full((len(documents), len(q)), -torch.inf, device=query.device)
        for start in range(0, padded.shape[1], document_block):
            d = padded[:, start:start + document_block]
            scores = torch.einsum('qd,bkd->bqk', q, d).float()
            valid = torch.arange(start, start + d.shape[1], device=query.device)[None] < lengths[:, None]
            scores.masked_fill_(~valid[:, None, :], -torch.inf)
            best = torch.maximum(best, scores.max(dim=-1).values)
        result += best.sum(dim=-1)
    return result / len(query)


class TokenStore:
    """Atomic document batches on disk with a bounded CPU shard cache."""
    def __init__(self, root, patients, cohorts, splits, batch_size, *, cache_batches=32):
        self.root, self.batches, self.locations = Path(root), [], {}
        self.cache, self.cache_batches = OrderedDict(), cache_batches
        for kind, rows in [('patients', patients), ('cohorts', cohorts)]:
            ordered = sorted((r for r in rows if r['split'] in splits), key=lambda r: (len(r['text']), r['id']))
            for start in range(0, len(ordered), batch_size):
                path = self.root / f'{kind}_{start // batch_size:07d}.pt'
                batch = ordered[start:start + batch_size]
                self.batches.append((path, batch))
                self.locations.update(((kind, row['id']), path) for row in batch)

    def encode(self, model, *, rank=0, world=1):
        self.root.mkdir(parents=True, exist_ok=True)
        for index, (path, rows) in enumerate(self.batches):
            if index % world != rank or path.exists():
                continue
            vectors = model.encode_batch([r['text'] for r in rows], is_query=False)
            temporary = path.with_suffix('.tmp')
            torch.save({row['id']: vector for row, vector in zip(rows, vectors, strict=True)}, temporary)
            temporary.replace(path)
            if index % (world * 20) == rank:
                print(f'[encode rank={rank}] batch {index + 1}/{len(self.batches)}', flush=True)

    def get(self, kind, identifier):
        path = self.locations[kind, identifier]
        if path not in self.cache:
            self.cache[path] = torch.load(path, map_location='cpu', weights_only=True)
            while len(self.cache) > self.cache_batches:
                self.cache.popitem(last=False)
        self.cache.move_to_end(path)
        return self.cache[path][identifier]


def export_candidates(corpus, candidates, cohort_output, patient_output):
    patients, cohorts = load_corpus(corpus)
    pmap, cmap = {p['id']: p for p in patients}, {c['id']: c for c in cohorts}
    tables = {direction: [] for direction in DIRECTIONS}
    for pair in read_jsonl(candidates):
        p, c = pmap[pair['patient_id']], cmap[pair['cohort_id']]
        if p['split'] != c['split'] or pair['split'] != p['split'] or p['split'] == 'test':
            raise ValueError('Candidate export crosses a split or includes test data')
        sources = pair['sources']
        for direction in DIRECTIONS:
            if not any(s.get('direction') in (direction, 'both') for s in sources):
                continue
            tables[direction].append(dict(pseudo_mrn=p['id'], patient_id=p['id'],
                patient_summary=p['text'], combined_patient_summary=p['text'], patient_boilerplate_text='',
                nct_id=c['trial_id'], trial_id=c['trial_id'], cohort_trial_id=c['id'], space_trial_id=c['id'],
                this_space=c['text'], cohort_eligibility_criteria=c['text'], trial_boilerplate_text='',
                split=p['split'], candidate_id=digest([p['id'], c['id']]),
                retrieval_sources=json.dumps(sources), trialspace_contract=score_contract()))
    for direction, path in zip(DIRECTIONS, (cohort_output, patient_output), strict=True):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp')
        pd.DataFrame(tables[direction]).to_parquet(temporary, index=False)
        temporary.replace(path)


def mine(args, model=None):
    rank, world, local = (int(os.environ.get(k, default)) for k, default in
                          [('RANK', '0'), ('WORLD_SIZE', '1'), ('LOCAL_RANK', '0')])
    if world > 1:
        dist.init_process_group('gloo')
    try:
        _mine(args, model, rank=rank, world=world, local=local)
    finally:
        if world > 1:
            dist.destroy_process_group()


@torch.inference_mode()
def _mine(args, model, *, rank, world, local):
    if min(args.document_batch_size, args.query_batch_size, args.score_batch_size, args.cache_batches) < 1:
        raise ValueError('Batch and cache sizes must be positive')
    if args.candidate_limit is not None and args.candidate_limit < 1:
        raise ValueError('Candidate limit must be positive')
    directional_top_k = {direction: getattr(args, 'top_k_' + direction, None) for direction in DIRECTIONS}
    directional_top_k = {direction: args.top_k if value is None else value for direction, value in directional_top_k.items()}
    if min(*directional_top_k.values(), args.lexical_k, args.random_k) < 0 or 'test' in args.splits:
        raise ValueError('Production mining requires nonnegative candidate counts and excludes test')
    output = Path(args.output)
    patients, cohorts = load_corpus(args.corpus)
    model = model or ColBERT(args.model, revision=args.revision, cache_dir=args.cache_dir,
                            query_length=args.query_length, document_length=args.document_length)
    saved_contract = getattr(model, 'folder', Path('/nonexistent')) / 'regression_scoring.json'
    if saved_contract.exists() and json.loads(saved_contract.read_text())['contract'] != score_contract():
        raise ValueError('Saved model uses a different regression prompt contract')
    source_pairs = Path(args.corpus) / 'source_trial_pairs.jsonl'
    extra = (read_jsonl(source_pairs) if source_pairs.exists() else []) + (read_jsonl(args.extra_pairs) if args.extra_pairs else [])
    signature = dict(version=VERSION, contract=score_contract(), model=model.fingerprint(),
        encoding_device_kind='cpu' if args.device == 'cpu' else 'cuda', token_storage_dtype='float16',
        patients=digest(patients), cohorts=digest(cohorts), extra_pairs=digest(extra), directional_top_k=directional_top_k,
        **{k: getattr(args, k) for k in ('top_k', 'lexical_k', 'random_k', 'candidate_limit',
            'proposal_strategy', 'cohort_query_limit', 'splits', 'seed', 'document_batch_size',
            'query_batch_size', 'score_batch_size')})
    if rank == 0:
        identity_file(output / 'mining_manifest.json', signature)
    if world > 1:
        dist.barrier()
    completion = output / 'complete.json'
    if completion.exists():
        saved = json.loads(completion.read_text())
        if any(file_sha256(output / name) != value for name, value in saved['outputs'].items()):
            raise ValueError('Completed mining artifacts changed')
        return
    device = 'cpu' if args.device == 'cpu' else f'cuda:{local}'
    if device != 'cpu':
        torch.cuda.set_device(local)
    model.to(device)
    store = TokenStore(output / 'tokens', patients, cohorts, args.splits, args.document_batch_size,
                       cache_batches=args.cache_batches)
    store.encode(model, rank=rank, world=world)
    if world > 1:
        dist.barrier()
    query_paths, query_number = [], 0
    for direction in DIRECTIONS:
        queries, documents = (patients, cohorts) if direction == 'patient_to_cohort' else (cohorts, patients)
        kind = 'cohorts' if direction == 'patient_to_cohort' else 'patients'
        for split in args.splits:
            targets = [r for r in documents if r['split'] == split]
            selected = [r for r in queries if r['split'] == split]
            if not targets or not selected:
                continue
            if direction == 'cohort_to_patient' and args.cohort_query_limit is not None:
                if args.cohort_query_limit < 1:
                    raise ValueError('Cohort query limit must be positive')
                selected = random.Random(f'{args.seed}:{split}').sample(selected, min(args.cohort_query_limit, len(selected)))
            jobs = []
            for row in selected:
                key = digest([direction, split, row['id']])
                path = output / 'queries' / key[:2] / f'{key}.json'
                query_paths.append(path)
                if query_number % world == rank and not path.exists():
                    jobs.append((row, path))
                query_number += 1
            if not jobs:
                continue
            lexical = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
            matrix = lexical.fit_transform(t['text'] for t in targets)
            for offset in range(0, len(jobs), args.query_batch_size):
                batch = jobs[offset:offset + args.query_batch_size]
                query_vectors = model.encode_batch([regression_query(direction, row['text']) for row, _ in batch], is_query=True)
                for (row, path), vector in zip(batch, query_vectors, strict=True):
                    rng = random.Random(f'{args.seed}:{direction}:{row["id"]}')
                    lexical_scores = (lexical.transform([row['text']]) @ matrix.T).toarray()[0]
                    proposals = proposal_indices(lexical_scores, args.candidate_limit, rng, args.proposal_strategy)
                    values = []
                    for begin in range(0, len(proposals), args.score_batch_size):
                        chosen = proposals[begin:begin + args.score_batch_size]
                        vectors = [store.get(kind, targets[i]['id']) for i in chosen]
                        values.extend(score_documents(vector.to(model.device), vectors).cpu().tolist())
                    hits = []
                    def add(index, source, rank_value=None, value=None):
                        pid, cid = (row['id'], targets[index]['id']) if direction == 'patient_to_cohort' else (targets[index]['id'], row['id'])
                        hits.append(dict(patient_id=pid, cohort_id=cid, direction=direction, source=source,
                                         task='gated_score' if source == 'colbert' else None, rank=rank_value, score=value))
                    for place, index in enumerate(top_indices(values, directional_top_k[direction]), 1):
                        add(proposals[index], 'colbert', place, values[index])
                    for place, index in enumerate(top_indices(lexical_scores, args.lexical_k), 1):
                        add(index, 'tfidf_baseline', place, float(lexical_scores[index]))
                    for index in rng.sample(range(len(targets)), min(args.random_k, len(targets))):
                        add(index, 'random_exploration')
                    write_json(path, hits)
                print(f'[mine rank={rank}] {direction}/{split}: {min(offset + len(batch), len(jobs))}/{len(jobs)}', flush=True)
    if world > 1:
        dist.barrier()
    if rank == 0:
        hits = [hit for path in query_paths for hit in json.loads(path.read_text())]
        for pair in extra:
            if pair['split'] in args.splits:
                hits.append(dict(patient_id=pair['patient_id'], cohort_id=pair['cohort_id'],
                                 direction='both', source='source_trial_or_independent_proposal', task=None))
        pairs = merge_candidates(patients, cohorts, hits)
        write_jsonl(output / 'candidates.jsonl', pairs)
        write_jsonl(output / 'rankings.jsonl', [h for h in hits if h['source'] == 'colbert'])
        export_candidates(args.corpus, output / 'candidates.jsonl', output / 'patient_to_cohort.parquet', output / 'cohort_to_patient.parquet')
        write_json(completion, dict(pairs=len(pairs), queries=len(query_paths),
            candidate_scope='exact MaxSim over recorded lexical/random proposal pools',
            outputs={name: file_sha256(output / name) for name in
                     ('candidates.jsonl', 'rankings.jsonl', 'patient_to_cohort.parquet', 'cohort_to_patient.parquet')}))
    if world > 1:
        dist.barrier()


def parser():
    p = shared_parser()
    p.description = __doc__
    sub = next(action for action in p._actions if hasattr(action, 'choices') and isinstance(action.choices, dict))
    sub.choices['mine'].set_defaults(objective='gated_regression', candidate_limit=256, proposal_strategy='lexical_random')
    sub.choices['mine'].add_argument('--query-batch-size', type=int, default=16)
    sub.choices['mine'].add_argument('--score-batch-size', type=int, default=16)
    sub.choices['mine'].add_argument('--cache-batches', type=int, default=32)
    sub.choices['mine'].add_argument('--top-k-patient-to-cohort', type=int,
        help='ColBERT cohort picks per patient; overrides --top-k for this direction')
    sub.choices['mine'].add_argument('--top-k-cohort-to-patient', type=int,
        help='ColBERT patient picks per cohort; overrides --top-k for this direction')
    sub.choices['label'].set_defaults(gated_regression=True, return_records=False)
    sub.choices['train'].set_defaults(objective='gated_regression', eval_every=100, eval_groups_per_stratum=32, steps=None)
    sub.choices['train'].add_argument('--epochs', type=float, default=3,
        help='Expected query-group sampling passes; --steps overrides this')
    sub.choices['train'].add_argument('--output-model', type=Path, help='Separate final model directory; checkpoints stay in --output')
    return p


def main():
    args = parser().parse_args()
    if args.command == 'prepare':
        print(prepare_corpus(args.patients, args.cohorts, args.output))
    elif args.command == 'mine':
        if args.objective != 'gated_regression':
            raise ValueError('Production TrialSpace requires gated regression')
        mine(args)
    elif args.command == 'label':
        from colbert_experiments.labeling import label
        args.scheduler_metrics_file = args.scheduler_metrics_file or str(Path(args.output) / 'concurrency_status.json')
        asyncio.run(label(args))
    elif args.command == 'train':
        if args.objective != 'gated_regression':
            raise ValueError('Production TrialSpace requires gated regression')
        from colbert_experiments.training import train
        train(args)
    else:
        raise ValueError('Use colbert_experiments for experimental/diagnostic subcommands')


if __name__ == '__main__':
    main()
