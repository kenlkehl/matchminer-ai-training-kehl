#!/usr/bin/env python3
"""Production TrialSpace corpus and teacher labels.

``prepare`` builds the corpus every round shares: patients with their key summary section as
``text`` and boilerplate section as ``boilerplate``, cohorts with their key criteria and
boilerplate criteria and tagged criteria lines (trialspace.data). ``label`` asks the teacher for the
candidate pairs' labels: ``--task match`` (the default), the five match labels in one request per
pair, scored 0-6 for TrialSpace and the decision model's match questions; ``--task boilerplate``,
the boilerplate label of each unique boilerplate combination; ``--task criteria``, one label per
criteria line of a seeded fraction of the pairs (the decision model's per-criterion question: yes =
the patient does not clearly fail the line). Each round labels only boilerplate combinations and
lines that earlier rounds' label directories (``--skip-labeled``) did not.
Mining is trialspace_comparison.py dense-mine; fitting is finetune_embedder.py.
"""
import argparse
import asyncio
from collections import defaultdict
import json
from pathlib import Path

import pandas as pd

from matchminer_ai.embedding.trialspace_contract import DIRECTIONS
from matchminer_ai.labeling import add_label_cli_args, score_contract
from trialspace.data import digest, load_corpus, prepare, read_jsonl, write_json, write_jsonl
from trial_cohort_format import file_stamp

VERSION = 'trialspace-corpus-v3'


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
        versions[cohort['trial_id']].add(cohort.get('source', 'ctgov'))
    result = dict(version=VERSION, patients=len(ps), cohorts=len(cs), trials=len(by_trial),
                  cohort_sources={name: sum(c.get('source', 'ctgov') == name for c in cs) for name in ('protocol', 'ctgov')},
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
                cohort_source=c.get('source', 'ctgov'),
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


def shared_parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare', help='Adapt summaries and the cohort table; key and boilerplate texts')
    prepare.add_argument('--patients', required=True)
    prepare.add_argument('--cohorts', required=True)
    prepare.add_argument('--output', required=True)
    # The labeler and its flags are the inference package's (matchminer_ai.labeling, moved there 2026-11-06).
    add_label_cli_args(commands.add_parser('label', help='Label candidate pairs: match, boilerplate or per-criterion'))
    return p


def parser():
    p = shared_parser()
    sub = next(action for action in p._actions if hasattr(action, 'choices') and isinstance(action.choices, dict))
    sub.choices['prepare'].add_argument('--merge-source-splits', action=argparse.BooleanOptionalAction, default=True,
        help='Pool all synthetic patients/cohorts for retrieval and fitting; retain source_split provenance')
    sub.choices['label'].set_defaults(return_records=False)
    sub.choices['label'].add_argument('--reuse-labels', type=Path,
        help='Completed labels of an earlier round over the same corpus; its judgments of pairs in --candidates are imported')
    sub.choices['label'].add_argument('--reuse-candidates', type=Path, help='The candidates.jsonl --reuse-labels judged')
    sub.choices['label'].add_argument('--combine-candidates', type=Path, nargs='+', default=[],
        help='--task boilerplate: further candidate files; each unique combination of patient '
             'boilerplate section and cohort boilerplate criteria across all of them is labeled once')
    sub.choices['label'].add_argument('--skip-labeled', type=Path, nargs='+', default=[],
        help='--task boilerplate or criteria: earlier rounds\' label directories; their boilerplate combinations or '
             'criteria lines (their candidates.jsonl) are not labeled again')
    sub.choices['label'].add_argument('--criteria-pair-fraction', type=float, default=0.01,
        help='--task criteria: the share of candidate pairs whose every tagged criteria line is labeled, chosen by a '
             'seeded hash of the pair so a pair is chosen in every round or none')
    sub.choices['label'].add_argument('--criteria-seed', type=int, default=42)
    return p


def skipped(directories, field):
    """The values of ``field`` in earlier label directories' candidates.jsonl."""
    return {row[field] for directory in directories for row in read_jsonl(Path(directory) / 'candidates.jsonl')}


def selected(seed, patient_id, cohort_id, fraction):
    """Whether a pair is among the seeded ``fraction`` whose criteria lines are labeled."""
    return int(digest([seed, patient_id, cohort_id])[:15], 16) / 16 ** 15 < fraction


def criteria_digest(items):
    """A cohort's tagged lines (heading, text, type): sibling cohorts with the same lines share labels."""
    return digest([[i.get('heading') or '', i['text'], i['criterion_type']] for i in items])


def criteria_line_candidates(corpus, candidate_files, output, *, fraction, seed, skip=()):
    """One candidate row per tagged criteria line of each selected pair (``selected``), skipping
    lines already labeled for the patient (the same patient and criteria in an earlier round's
    directory, or a sibling cohort here). Saved to <output>/candidates.jsonl with a receipt."""
    from matchminer_ai.matching.decision_model import checkable_items
    if not 0 < fraction <= 1:
        raise ValueError('--criteria-pair-fraction must be in (0, 1]')
    patients, cohorts = load_corpus(corpus)
    cmap = {c['id']: c for c in cohorts}
    done = skipped(skip, 'line_key')
    rows, pairs, seen_pairs = [], 0, set()
    for path in candidate_files:
        for pair in read_jsonl(path):
            key = (pair['patient_id'], pair['cohort_id'])
            if key in seen_pairs or not selected(seed, *key, fraction):
                continue
            seen_pairs.add(key)
            cohort = cmap[pair['cohort_id']]
            items = checkable_items(cohort.get('items') or [])
            if not items:
                continue
            pairs += 1
            digest_ = criteria_digest(cohort['items'])
            for item in items:
                line = digest([pair['patient_id'], digest_, item['number']])
                if line in done:
                    continue
                done.add(line)
                rows.append(dict(patient_id=pair['patient_id'], cohort_id=pair['cohort_id'], trial_id=cohort['trial_id'],
                                 split=pair['split'], criteria_digest=digest_, number=int(item['number']),
                                 criterion_type=item['criterion_type'], boilerplate=bool(item.get('boilerplate')),
                                 line_key=line))
    path = Path(output) / 'candidates.jsonl'
    if path.exists():
        if read_jsonl(path) != rows:
            raise ValueError('This label directory was built from different candidates; use a new directory')
    else:
        write_jsonl(path, rows)
    write_json(Path(output) / 'criteria_sample.json', dict(
        sources=[dict(path=str(Path(f).resolve()), stamp=file_stamp(f)) for f in candidate_files],
        skipped_directories=[str(Path(d).resolve()) for d in skip], fraction=fraction, seed=seed,
        selected_pairs_with_lines=pairs, lines=len(rows),
        exclusion_lines=sum(r['criterion_type'] == 'exclusion' for r in rows),
        boilerplate_lines=sum(r['boilerplate'] for r in rows)))
    return path


def unique_boilerplate_pairs(corpus, candidate_files, output, skip=()):
    """One pair per unique (patient boilerplate section, cohort boilerplate criteria) combination.

    Reads every candidate file (one TrialSpace round's), keeps pairs whose cohort has boilerplate
    criteria, and takes the first pair in (patient_id, cohort_id) order for each combination of
    the two texts not already in an earlier round's label directory (``skip``). Saved to
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
    chosen, combinations, done = [], {}, skipped(skip, 'boilerplate_combination')
    for pair in sorted(pairs, key=lambda pair: (pair['patient_id'], pair['cohort_id'])):
        combination = digest([pmap[pair['patient_id']].get('boilerplate') or '', cmap[pair['cohort_id']]['boilerplate']])
        if combination in done:
            continue
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
        skipped_directories=[str(Path(d).resolve()) for d in skip], skipped_combinations=len(done),
        candidate_pairs_with_boilerplate=len(pairs), unique_combinations=len(chosen),
        pairs_per_combination_max=max(combinations.values(), default=0)))
    return path


def write_reuse_plan(args):
    """Record the earlier round to import from (matchminer_ai.labeling.full_pool_reuse verifies it)."""
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
        from matchminer_ai.labeling import label
        args.scheduler_metrics_file = args.scheduler_metrics_file or str(Path(args.output) / 'concurrency_status.json')
        if args.reuse_labels or args.reuse_candidates:
            write_reuse_plan(args)
        if args.task == 'boilerplate':
            args.candidates = str(unique_boilerplate_pairs(args.corpus, [Path(args.candidates), *args.combine_candidates],
                                                           args.output, skip=args.skip_labeled))
        elif args.task == 'criteria':
            args.candidates = str(criteria_line_candidates(args.corpus, [Path(args.candidates), *args.combine_candidates],
                                                           args.output, fraction=args.criteria_pair_fraction,
                                                           seed=args.criteria_seed, skip=args.skip_labeled))
        elif args.combine_candidates or args.skip_labeled:
            raise ValueError('--combine-candidates and --skip-labeled are for boilerplate and criteria labels')
        asyncio.run(label(args))


if __name__ == '__main__':
    main()
