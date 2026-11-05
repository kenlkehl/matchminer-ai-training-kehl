#!/usr/bin/env python3
"""Production TrialSpace corpus and teacher labels.

``prepare`` builds the corpus every round shares: patients with their key summary section as
``text`` and boilerplate section as ``boilerplate``, cohorts with their key criteria and
boilerplate criteria (trialspace.data). ``label`` asks the teacher for the candidate pairs' labels:
``--task match`` (the default), the five match labels in one request per pair, scored 0-6 for
TrialSpace and the multi-head TrialChecker; ``--task boilerplate``, the BoilerplateChecker label.
Mining is trialspace_comparison.py dense-mine; fitting is finetune_embedder.py.
"""
import argparse
import asyncio
from collections import defaultdict
import json
from pathlib import Path

import pandas as pd

from trialspace.__main__ import parser as shared_parser
from trialspace.contracts import DIRECTIONS
from trialspace.data import digest, load_corpus, prepare, read_jsonl, write_json, write_jsonl
from trialspace.regression import score_contract
from trial_cohort_format import file_stamp

VERSION = 'trialspace-corpus-v2'


def identity_file(path, identity):
    path = Path(path)
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError(f'Changed TrialSpace inputs/settings at {path}; use a new run directory')
    write_json(path, identity)


def prepare_corpus(patients, cohorts, output, *, merge_source_splits=True):
    if not merge_source_splits:
        raise ValueError('Production cannot hold out synthetic source splits; use --merge-source-splits')
    output = Path(output)
    identity = dict(version=VERSION, patients=str(Path(patients).resolve()), cohorts=str(Path(cohorts).resolve()),
                    patient_stamp=file_stamp(patients), cohort_stamp=file_stamp(cohorts), contract=score_contract(),
                    merge_source_splits=merge_source_splits)
    identity_file(output / 'input_manifest.json', identity)
    complete = output / 'prepared.json'
    if complete.exists():
        saved = json.loads(complete.read_text())
        for name, expected in saved['output_stamp'].items():
            if file_stamp(output / name) != expected:
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
    versions = defaultdict(set)
    for cohort in cs:
        versions[cohort['trial_id']].add(cohort.get('source', 'registry'))
    result = dict(version=VERSION, patients=len(ps), cohorts=len(cs), trials=len(by_trial),
                  cohort_sources={name: sum(c.get('source', 'registry') == name for c in cs) for name in ('protocol', 'registry')},
                  trials_with_both_versions=sum(len(v) > 1 for v in versions.values()),
                  source_trial_pairs=len(lineage), contract=score_contract(),
                  output_stamp={name: file_stamp(output / name) for name in
                                 ('patients.jsonl', 'cohorts.jsonl', 'source_trial_pairs.jsonl')})
    write_json(complete, result)
    return result



def export_candidates(corpus, candidates, cohort_output, patient_output):
    """Candidate pairs per retrieval direction, with both key texts and both boilerplate texts."""
    patients, cohorts = load_corpus(corpus)
    pmap, cmap = {p['id']: p for p in patients}, {c['id']: c for c in cohorts}
    tables = {direction: [] for direction in DIRECTIONS}
    # Freeze the contract once for this export rather than per pair.
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
                patient_summary=p['text'], patient_boilerplate_text=p.get('boilerplate', ''),
                nct_id=c['trial_id'], trial_id=c['trial_id'], cohort_trial_id=c['id'], space_trial_id=c['id'],
                this_space=c['text'], cohort_key_criteria=c['text'], trial_boilerplate_text=c.get('boilerplate', ''),
                split=p['split'], candidate_id=digest([p['id'], c['id']]),
                patient_source_split=p.get('source_split', p['split']),
                cohort_source_split=c.get('source_split', c['split']),
                cohort_source=c.get('source', 'registry'),
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


def parser():
    p = shared_parser()
    p.description = __doc__
    sub = next(action for action in p._actions if hasattr(action, 'choices') and isinstance(action.choices, dict))
    sub.choices['prepare'].add_argument('--merge-source-splits', action=argparse.BooleanOptionalAction, default=True,
        help='Pool all synthetic patients/cohorts for retrieval and fitting; retain source_split provenance')
    sub.choices['label'].set_defaults(return_records=False)
    sub.choices['label'].add_argument('--reuse-labels', type=Path,
        help='Completed labels of an earlier round over the same corpus; its judgments of pairs in --candidates are imported')
    sub.choices['label'].add_argument('--reuse-candidates', type=Path, help='The candidates.jsonl --reuse-labels judged')
    sub.choices['label'].add_argument('--combine-candidates', type=Path, nargs='+', default=[],
        help='--task boilerplate: further candidate files (earlier rounds); each unique combination of patient '
             'boilerplate section and cohort boilerplate criteria across all of them is labeled once')
    return p


def unique_boilerplate_pairs(corpus, candidate_files, output):
    """One pair per unique (patient boilerplate section, cohort boilerplate criteria) combination.

    Reads every candidate file (TrialSpace rounds 1-3), keeps pairs whose cohort has boilerplate
    criteria, and takes the first pair in (patient_id, cohort_id) order for each combination of
    the two texts, as BoilerplateChecker training always deduplicated them. Saved to
    <output>/candidates.jsonl with a receipt; a rerun must find the same pairs.
    """
    patients, cohorts = load_corpus(corpus)
    pmap, cmap = {p['id']: p for p in patients}, {c['id']: c for c in cohorts}
    pairs, seen_pairs = [], set()
    for path in candidate_files:
        for pair in read_jsonl(path):
            key = (pair['patient_id'], pair['cohort_id'])
            if key not in seen_pairs and (cmap[pair['cohort_id']].get('boilerplate') or '').strip():
                seen_pairs.add(key)
                pairs.append(dict(patient_id=pair['patient_id'], cohort_id=pair['cohort_id'],
                                  trial_id=pair.get('trial_id', cmap[pair['cohort_id']]['trial_id']), split=pair['split']))
    chosen, combinations = [], {}
    for pair in sorted(pairs, key=lambda pair: (pair['patient_id'], pair['cohort_id'])):
        combination = digest([pmap[pair['patient_id']].get('boilerplate') or '', cmap[pair['cohort_id']]['boilerplate']])
        if combination not in combinations:
            combinations[combination] = 0
            chosen.append(dict(pair, boilerplate_combination=combination))
        combinations[combination] += 1
    path = Path(output) / 'candidates.jsonl'
    if path.exists():
        if read_jsonl(path) != chosen:
            raise ValueError('This label directory was built from different candidates; use a new directory')
    else:
        write_jsonl(path, chosen)
    write_json(Path(output) / 'combinations.json', dict(
        sources=[dict(path=str(Path(f).resolve()), stamp=file_stamp(f)) for f in candidate_files],
        candidate_pairs_with_boilerplate=len(pairs), unique_combinations=len(chosen),
        pairs_per_combination_max=max(combinations.values(), default=0)))
    return path


def write_reuse_plan(args):
    """Record the earlier round to import from (trialspace.full_pool_reuse verifies it)."""
    if not (args.reuse_labels and args.reuse_candidates):
        raise ValueError('--reuse-labels and --reuse-candidates go together')
    plan = dict(sources=[dict(labels=str(args.reuse_labels), candidates=str(args.reuse_candidates),
                              mode='candidate_intersection', corpus=str(args.corpus))])
    path = Path(args.output) / 'reuse_sources.json'
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise ValueError('This label directory already imports from a different source')
        return
    write_json(path, plan)


def main():
    args = parser().parse_args()
    if args.command == 'prepare':
        print(prepare_corpus(args.patients, args.cohorts, args.output, merge_source_splits=args.merge_source_splits))
    elif args.command == 'label':
        from synthetic_pool import require_full_corpus
        require_full_corpus(args.corpus)
        from trialspace.labeling import label
        args.scheduler_metrics_file = args.scheduler_metrics_file or str(Path(args.output) / 'concurrency_status.json')
        if args.reuse_labels or args.reuse_candidates:
            write_reuse_plan(args)
        if args.task == 'boilerplate':
            args.candidates = str(unique_boilerplate_pairs(args.corpus, [Path(args.candidates), *args.combine_candidates],
                                                           args.output))
        elif args.combine_candidates:
            raise ValueError('--combine-candidates is for boilerplate labels; match labels cover --candidates')
        asyncio.run(label(args))


if __name__ == '__main__':
    main()
