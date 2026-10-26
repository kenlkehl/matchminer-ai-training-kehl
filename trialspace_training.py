#!/usr/bin/env python3
"""Production TrialSpace: complete cohorts, gated labels, ColBERT ordinal ranking.

The experiment and production entrypoints share the tested native ColBERT,
question contracts, adaptive teacher, and student trainer. This entrypoint
defaults to CoSENT plus known-negative contrast and adds disk-backed mining.
"""
import argparse
import asyncio
from collections import OrderedDict, defaultdict
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from sklearn.feature_extraction.text import TfidfVectorizer

from colbert_experiments.__main__ import parser as shared_parser
from colbert_experiments.contracts import DIRECTIONS
from colbert_experiments.data import digest, load_corpus, prepare, read_jsonl, write_json, write_jsonl
from colbert_experiments.model import ColBERT
from colbert_experiments.packed_scoring import PackedTokens, score_pairs
from colbert_experiments.regression import regression_query, score_contract
from colbert_experiments.retrieval import merge_candidates, proposal_indices, top_indices
from trial_cohort_format import file_sha256

VERSION = 'trialspace-colbert-regression-v1'


def identity_file(path, identity):
    path = Path(path)
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError(f'Changed TrialSpace inputs/settings at {path}; use a new run directory')
    write_json(path, identity)


def prepare_corpus(patients, cohorts, output, *, merge_source_splits=True):
    output = Path(output)
    identity = dict(version=VERSION, patients=str(Path(patients).resolve()), cohorts=str(Path(cohorts).resolve()),
                    patient_sha256=file_sha256(patients), cohort_sha256=file_sha256(cohorts), contract=score_contract(),
                    merge_source_splits=merge_source_splits)
    identity_file(output / 'input_manifest.json', identity)
    complete = output / 'prepared.json'
    if complete.exists():
        saved = json.loads(complete.read_text())
        for name, expected in saved['output_sha256'].items():
            if file_sha256(output / name) != expected:
                raise ValueError('Prepared TrialSpace corpus changed')
        return saved
    prepare(patients, cohorts, output, merge_source_splits=merge_source_splits)
    ps, cs = load_corpus(output)
    by_trial = defaultdict(list)
    for cohort in cs:
        by_trial[cohort['trial_id']].append(cohort)
    lineage = [dict(patient_id=p['id'], cohort_id=c['id'], split=p['split'])
               for p in ps for c in by_trial.get(p.get('source_trial_id'), [])
               if p['split'] == c['split']]
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

    def resident(self, kind, device, *, budget_gb=8, reserve_gb=8):
        """Read each shard once when this direction's corpus fits the budget."""
        if budget_gb < 0 or reserve_gb < 0:
            raise ValueError('Token cache budget and reserve must be nonnegative')
        paths = [path for path, _ in self.batches if path.name.startswith(kind + '_')]
        estimated_bytes = sum(path.stat().st_size for path in paths)
        budget = int(budget_gb * 1024 ** 3)
        if str(device).startswith('cuda'):
            # The previous retrieval direction's tensors have been released,
            # but their allocator cache can still obscure available GPU memory.
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
            free, _ = torch.cuda.mem_get_info(device)
            budget = min(budget, max(0, free - int(reserve_gb * 1024 ** 3)))
        if not paths or estimated_bytes > budget:
            return None
        vectors = {}
        from concurrent.futures import ThreadPoolExecutor
        # Bound network-filesystem reads while retaining shard/ID order exactly.
        with ThreadPoolExecutor(max_workers=min(8, len(paths))) as readers:
            for batch in readers.map(lambda path: torch.load(path, map_location='cpu', weights_only=True), paths):
                vectors.update(batch)
        # Discard the CPU LRU after the one-time transfer. The packed corpus is
        # replaced between directions, so patient and cohort caches do not add.
        packed = PackedTokens(list(vectors.values()), device)
        ids = {identifier: i for i, identifier in enumerate(vectors)}
        self.cache.clear()
        print(f'[resident] {kind}: {len(ids)} documents, {packed.values.numel() * packed.values.element_size() / 1024**3:.2f} GiB on {device}', flush=True)
        return packed, ids


def export_candidates(corpus, candidates, cohort_output, patient_output):
    patients, cohorts = load_corpus(corpus)
    pmap, cmap = {p['id']: p for p in patients}, {c['id']: c for c in cohorts}
    tables = {direction: [] for direction in DIRECTIONS}
    # Freeze the rubric once for this export. Re-reading six prompt files for
    # every pair causes millions of network-filesystem operations at scale.
    contract = score_contract()
    for pair in read_jsonl(candidates):
        p, c = pmap[pair['patient_id']], cmap[pair['cohort_id']]
        if p['split'] != c['split'] or pair['split'] != p['split']:
            raise ValueError('Candidate export crosses an effective split')
        sources = pair['sources']
        for direction in DIRECTIONS:
            if not any(s.get('direction') in (direction, 'both') for s in sources):
                continue
            tables[direction].append(dict(pseudo_mrn=p['id'], patient_id=p['id'],
                patient_summary=p['text'], combined_patient_summary=p['text'], patient_boilerplate_text='',
                nct_id=c['trial_id'], trial_id=c['trial_id'], cohort_trial_id=c['id'], space_trial_id=c['id'],
                this_space=c['text'], cohort_eligibility_criteria=c['text'], trial_boilerplate_text='',
                split=p['split'], candidate_id=digest([p['id'], c['id']]),
                patient_source_split=p.get('source_split', p['split']),
                cohort_source_split=c.get('source_split', c['split']),
                retrieval_sources=json.dumps(sources), trialspace_contract=contract))
    for direction, path in zip(DIRECTIONS, (cohort_output, patient_output), strict=True):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp')
        # Full clinical strings repeat across millions of pairs. The default
        # 1 MiB dictionary spills almost immediately and duplicates that text.
        pd.DataFrame(tables[direction]).to_parquet(temporary, index=False,
            compression='zstd', use_dictionary=True, dictionary_pagesize_limit=256 * 1024**2)
        temporary.replace(path)


def finalize_candidate_exports(corpus, output, *, pair_count, query_count):
    """Publish full-text tables and their checksums after query JSON is durable."""
    output = Path(output)
    export_candidates(corpus, output / 'candidates.jsonl', output / 'patient_to_cohort.parquet',
                      output / 'cohort_to_patient.parquet')
    write_json(output / 'complete.json', dict(pairs=pair_count, queries=query_count,
        candidate_scope='exact MaxSim over recorded lexical/random proposal pools',
        outputs={name: file_sha256(output / name) for name in
                 ('candidates.jsonl', 'rankings.jsonl', 'patient_to_cohort.parquet', 'cohort_to_patient.parquet')}))


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
    if min(args.pair_batch_size, args.score_query_batches) < 1 or args.device_cache_gb < 0:
        raise ValueError('Invalid resident scoring cache/batch settings')
    directional_top_k = {direction: getattr(args, 'top_k_' + direction, None) for direction in DIRECTIONS}
    directional_top_k = {direction: args.top_k if value is None else value for direction, value in directional_top_k.items()}
    if min(*directional_top_k.values(), args.lexical_k, args.random_k) < 0:
        raise ValueError('Production mining requires nonnegative candidate counts')
    output = Path(args.output)
    patients, cohorts = load_corpus(args.corpus)
    model = model or ColBERT(args.model, revision=args.revision, cache_dir=args.cache_dir,
                            query_length=args.query_length, document_length=args.document_length)
    saved_contract = getattr(model, 'folder', Path('/nonexistent')) / 'regression_scoring.json'
    if saved_contract.exists() and json.loads(saved_contract.read_text())['contract'] != score_contract():
        raise ValueError('Saved model uses a different regression prompt contract')
    source_pairs = Path(args.corpus) / 'source_trial_pairs.jsonl'
    extra = (read_jsonl(source_pairs) if source_pairs.exists() else []) + (read_jsonl(args.extra_pairs) if args.extra_pairs else [])
    # Older prepared corpora omitted test lineage proposals. Recover selected
    # missing source pairs from immutable corpus IDs without rewriting that corpus.
    known = {(r['patient_id'], r['cohort_id']) for r in extra}
    by_trial = defaultdict(list)
    for cohort in cohorts:
        by_trial[cohort['trial_id']].append(cohort)
    for patient in patients:
        if patient['split'] not in args.splits:
            continue
        for cohort in by_trial.get(patient.get('source_trial_id'), []):
            key = (patient['id'], cohort['id'])
            if patient['split'] == cohort['split'] and key not in known:
                extra.append(dict(patient_id=key[0], cohort_id=key[1], split=patient['split']))
                known.add(key)
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
    write_json(output / f'runtime_rank_{rank}.json', dict(world_size=world, device=device,
        pair_batch_size=args.pair_batch_size, score_query_batches=args.score_query_batches,
        device_token_cache=args.device_token_cache, device_cache_gb=args.device_cache_gb))
    from colbert_experiments.query_checkpoints import QueryCheckpoints
    checkpoints = QueryCheckpoints(output)
    query_paths, query_number = [], 0
    resident = None
    for direction in DIRECTIONS:
        queries, documents = (patients, cohorts) if direction == 'patient_to_cohort' else (cohorts, patients)
        kind = 'cohorts' if direction == 'patient_to_cohort' else 'patients'
        resident = None  # Release the previous direction before staging this one.
        if args.device_token_cache:
            resident = store.resident(kind, device, budget_gb=args.device_cache_gb,
                                     reserve_gb=getattr(args, 'device_cache_reserve_gb', 8))
            if resident is None:
                print(f'[mine rank={rank}] {kind}: resident budget unavailable; using bounded CPU cache', flush=True)
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
                if query_number % world == rank and key not in checkpoints.keys:
                    jobs.append((row, path))
                query_number += 1
            if not jobs:
                continue
            lexical = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
            matrix = lexical.fit_transform(t['text'] for t in targets)
            work_batch_size = args.query_batch_size * (args.score_query_batches if resident else 1)
            for offset in range(0, len(jobs), work_batch_size):
                started = time.perf_counter()
                batch = jobs[offset:offset + work_batch_size]
                query_vectors = []
                # Preserve the established encoder batch shape/precision while
                # combining several batches for lexical work and pair scoring.
                for start in range(0, len(batch), args.query_batch_size):
                    query_vectors.extend(model.encode_batch([regression_query(direction, row['text'])
                        for row, _ in batch[start:start + args.query_batch_size]], is_query=True))
                encoded_at = time.perf_counter()
                lexical_batch = (lexical.transform([row['text'] for row, _ in batch]) @ matrix.T).toarray()
                proposals_batch, random_states = [], []
                for (row, _), lexical_scores in zip(batch, lexical_batch, strict=True):
                    rng = random.Random(f'{args.seed}:{direction}:{row["id"]}')
                    proposals_batch.append(proposal_indices(lexical_scores, args.candidate_limit, rng, args.proposal_strategy))
                    random_states.append(rng)
                proposed_at = time.perf_counter()
                scores_batch = []
                if resident:
                    ids = resident[1]
                    query_ids = [qi for qi, proposals in enumerate(proposals_batch) for _ in proposals]
                    document_ids = [ids[targets[i]['id']] for proposals in proposals_batch for i in proposals]
                    values = score_pairs(PackedTokens(query_vectors, device), resident[0], query_ids, document_ids,
                                         pair_batch_size=args.pair_batch_size)
                    begin = 0
                    for proposals in proposals_batch:
                        scores_batch.append(values[begin:begin + len(proposals)])
                        begin += len(proposals)
                else:
                    for vector, proposals in zip(query_vectors, proposals_batch, strict=True):
                        values = []
                        for begin in range(0, len(proposals), args.score_batch_size):
                            chosen = proposals[begin:begin + args.score_batch_size]
                            vectors = [store.get(kind, targets[i]['id']) for i in chosen]
                            values.extend(score_documents(vector.to(model.device), vectors).cpu().tolist())
                        scores_batch.append(values)
                scored_at = time.perf_counter()
                checkpoint_rows = {}
                for (row, path), lexical_scores, proposals, values, rng in zip(
                        batch, lexical_batch, proposals_batch, scores_batch, random_states, strict=True):
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
                    checkpoint_rows[path.stem] = hits
                checkpoints.write_batch(checkpoint_rows)
                finished = time.perf_counter()
                print(f'[mine rank={rank}] {direction}/{split}: {min(offset + len(batch), len(jobs))}/{len(jobs)} '
                      f'encode={encoded_at-started:.3f}s proposals={proposed_at-encoded_at:.3f}s '
                      f'score={scored_at-proposed_at:.3f}s checkpoint={finished-scored_at:.3f}s', flush=True)
    if world > 1:
        dist.barrier()
    if rank == 0:
        hits = checkpoints.ordered_hits([path.stem for path in query_paths])
        for pair in extra:
            if pair['split'] in args.splits:
                hits.append(dict(patient_id=pair['patient_id'], cohort_id=pair['cohort_id'],
                                 direction='both', source='source_trial_or_independent_proposal', task=None))
        pairs = merge_candidates(patients, cohorts, hits)
        write_jsonl(output / 'candidates.jsonl', pairs)
        write_jsonl(output / 'rankings.jsonl', [h for h in hits if h['source'] == 'colbert'])
        finalize_candidate_exports(args.corpus, output, pair_count=len(pairs), query_count=len(query_paths))
    if world > 1:
        dist.barrier()


def parser():
    p = shared_parser()
    p.description = __doc__
    sub = next(action for action in p._actions if hasattr(action, 'choices') and isinstance(action.choices, dict))
    sub.choices['prepare'].add_argument('--merge-source-splits', action=argparse.BooleanOptionalAction, default=True,
        help='Pool all synthetic patients/cohorts for retrieval and fitting; retain source_split provenance')
    sub.choices['mine'].set_defaults(objective='gated_regression', candidate_limit=256,
                                    proposal_strategy='lexical_random', splits=['train', 'val', 'test'])
    sub.choices['mine'].add_argument('--query-batch-size', type=int, default=16)
    sub.choices['mine'].add_argument('--score-batch-size', type=int, default=16)
    sub.choices['mine'].add_argument('--cache-batches', type=int, default=32)
    sub.choices['mine'].add_argument('--device-token-cache', action=argparse.BooleanOptionalAction, default=True)
    sub.choices['mine'].add_argument('--device-cache-gb', type=float, default=8,
        help='Per-worker resident token budget; larger corpora use the bounded CPU cache')
    sub.choices['mine'].add_argument('--device-cache-reserve-gb', type=float, default=8,
        help='GPU memory reserved outside the resident token cache; reduce only for bounded repair jobs')
    sub.choices['mine'].add_argument('--pair-batch-size', type=int, default=512,
        help='Query/document pairs scored together across multiple queries')
    sub.choices['mine'].add_argument('--score-query-batches', type=int, default=4,
        help='Encoder batches grouped for lexical proposal generation and pair scoring')
    sub.choices['mine'].add_argument('--top-k-patient-to-cohort', type=int,
        help='ColBERT cohort picks per patient; overrides --top-k for this direction')
    sub.choices['mine'].add_argument('--top-k-cohort-to-patient', type=int, default=40,
        help='ColBERT patient picks per cohort; overrides --top-k for this direction')
    sub.choices['label'].set_defaults(gated_regression=True, return_records=False)
    sub.choices['train'].set_defaults(objective='gated_regression', loss='legacy_ranking',
                                     learning_rate=2e-5, lr_scheduler='linear', warmup_ratio=.01,
                                     eval_every=100, eval_groups_per_stratum=32,
                                     steps=None, accumulation=32, documents_per_query=8,
                                     query_groups_per_forward=2,
                                     training_splits=['train', 'val', 'test'], require_complete_coverage=True)
    sub.choices['train'].add_argument('--epochs', type=float, default=3,
        help='Complete directional-pair epochs for ranking; query sampling passes for MSE; --steps overrides')
    sub.choices['train'].add_argument('--output-model', type=Path, help='Separate final model directory; checkpoints stay in --output')
    return p


def main():
    args = parser().parse_args()
    if args.command == 'prepare':
        print(prepare_corpus(args.patients, args.cohorts, args.output, merge_source_splits=args.merge_source_splits))
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
        if train(args) == 'paused':
            raise SystemExit(75)
    else:
        raise ValueError('Use colbert_experiments for experimental/diagnostic subcommands')


if __name__ == '__main__':
    main()
