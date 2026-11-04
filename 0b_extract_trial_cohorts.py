#!/usr/bin/env python3
"""Extract cancer-treatment cohorts through the sibling inference package API.

summarize_trials runs the inference package's three-step workflow on each
registry listing (title, brief summary and eligibility criteria as one document):
a cancer-treatment screen, the cohort names, and one request per cohort for its
criteria, copied verbatim with boilerplate included, with the same prompts as
synthetic protocols. No training-owned prompt. For training, the same steps as
protocols (synthetic_protocols/extract_cohorts.py) follow: a trial's cohorts whose
criteria are essentially the same are combined (merge_equivalent_cohorts,
trial.cohorts.merge.user.txt; --no-merge-cohorts keeps every extracted cohort), then a
separate request per cohort separates its boilerplate criteria from the rest
(remove_boilerplate_criteria, trial.cohorts.deboilerplate.user.txt;
--no-deboilerplate-cohorts skips it). Exports: the merged rows (--output-cohorts,
boilerplate kept), the rows before merging (--output-unmerged-cohorts) and the merged
rows with boilerplate removed (--output-deboilerplated-cohorts, which training's
retrieval uses; their boilerplate criteria are in cohort_boilerplate_criteria). Checkpoints retain each trial's decision, cohort names, unmerged,
merged and de-boilerplated rows, reasoning and answers of every step, validation
attempts per step and endpoint provenance.

Two inputs: the parsed registry export (--input), from which interventional
early phase I-IV trials first posted since --start-year are selected, or an
already-sampled set of registry records (--trials-jsonl, e.g. the seeds.jsonl of
a synthetic protocol batch), every one of which is extracted. The second gives
each sampled trial its registry-derived cohorts beside the cohorts extracted
from its synthetic protocol (synthetic_protocols/README.md).
"""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import ExitStack
import csv
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import resource
import sys

import numpy as np
import pandas as pd

from trial_cohort_format import REPRESENTATION, validate_cohort_rows, read_clinical_csv, file_stamp

ROOT = Path(__file__).resolve().parent
PHASES = {'EARLY_PHASE1', 'PHASE1', 'PHASE2', 'PHASE3', 'PHASE4'}


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(temporary, 'wt') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    temporary.replace(path)


def select_trials(frame, *, start_year=2020, seed=42):
    """2020-inclusive interventional early phase I–IV; assign splits by NCT ID."""
    required = {'nct_id', 'study_first_post_date', 'study_type', 'phases',
                'title', 'brief_summary', 'eligibility_criteria'}
    if required - set(frame):
        raise ValueError(f'Missing parsed CT.gov columns: {sorted(required - set(frame))}')
    def eligible_phases(value):
        try:
            phases = json.loads(value)
        except (TypeError, ValueError):
            return False
        return (isinstance(phases, list) and bool(phases)
                and all(isinstance(p, str) and p in PHASES for p in phases))
    dates = pd.to_datetime(frame.study_first_post_date, format='mixed', errors='coerce')
    frame = frame.loc[dates.dt.year.ge(start_year) & frame.study_type.eq('INTERVENTIONAL')
                      & frame.phases.map(eligible_phases)].copy()
    if frame.empty:
        raise ValueError('No trials meet the date, study type and phase selection')
    if not frame.nct_id.str.fullmatch(r'NCT\d{8}', na=False).all():
        raise ValueError('Expected ClinicalTrials.gov NCT identifiers')
    frame = frame.drop_duplicates()
    if frame.nct_id.duplicated().any():
        raise ValueError('Conflicting duplicate source records for an NCT ID')
    frame = frame.sort_values('nct_id').reset_index(drop=True)
    if 'split' in frame:
        if not frame.split.isin(['train', 'val', 'validation', 'test']).all():
            raise ValueError('Missing or invalid source splits')
    else:
        frame['split'] = np.random.default_rng(seed).choice(['train', 'val', 'test'],
                                                         len(frame), p=[.8, .1, .1])
    for column in ('title', 'brief_summary', 'eligibility_criteria'):
        frame[column] = frame[column].fillna('')
    return frame


def load_sampled_trials(path):
    """Registry records of an already-sampled trial set; no selection or split assignment."""
    frame = pd.DataFrame([json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()])
    required = {'nct_id', 'title', 'brief_summary', 'eligibility_criteria'}
    if frame.empty or required - set(frame):
        raise ValueError(f'Sampled trials need columns {sorted(required)}')
    if not frame.nct_id.str.fullmatch(r'NCT\d{8}', na=False).all():
        raise ValueError('Expected ClinicalTrials.gov NCT identifiers')
    if frame.nct_id.duplicated().any():
        raise ValueError('Duplicate NCT IDs in the sampled trials')
    for column in ('title', 'brief_summary', 'eligibility_criteria'):
        frame[column] = frame[column].fillna('')
    return frame.sort_values('nct_id').reset_index(drop=True)


def make_config(args):
    from matchminer_ai.config import load_default_preset
    from matchminer_ai.llm.model_profiles import apply_model_profile
    config = load_default_preset()
    config.debug_mode = True
    family = args.tokenizer_model or args.model
    if apply_model_profile(config, family, sections=['trial']) is None:
        raise ValueError('No vendor sampling profile; supply a recognized --tokenizer-model')
    config.trial['extraction_method'] = 'cohorts'
    config.trial['cohorts'] = {'max_attempts': args.max_attempts}
    config.trial['remote'].update(model_name=args.model, tokenizer_name=family)
    config.trial['remote']['request_params']['max_tokens'] = args.max_tokens
    config.remote.update(enabled=True, server_urls=args.server_urls or [],
        server_urls_file=str(args.server_urls_file) if args.server_urls_file else None,
        max_concurrent_requests=args.max_concurrent_per_server, pool_max_workers=args.workers,
        request_timeout=args.request_timeout, recovery_timeout=args.recovery_timeout)
    config.remote['adaptive_concurrency'].update(enabled=True, metrics_interval=10)
    config.model_metadata_cache_dir = str(args.work_dir / 'model_metadata')
    return config


def extract_one(row, config, merge=True, deboilerplate=True):
    from matchminer_ai.trials import merge_equivalent_cohorts, remove_boilerplate_criteria, summarize_trials
    from matchminer_ai.trials.cohorts import COHORT_COLUMNS
    trial = row['nct_id']
    inputs = pd.DataFrame([dict(trial_id=trial, trial_title=row['title'],
        brief_summary=row['brief_summary'], eligibility_criteria=row['eligibility_criteria'])])
    record = dict(trial_id=trial, result=None, cohort_rows=[])
    try:
        cohorts, metadata = summarize_trials(inputs, config=config, return_metadata=True)
        record.update(result=metadata['trial_decisions'].get(trial),
            error=metadata['trial_errors'].get(trial), metadata=metadata,
            cohort_rows=json.loads(cohorts[COHORT_COLUMNS].to_json(orient='records')))
        if record['result'] is None:
            record['error'] = record['error'] or 'No validated decision returned by summarize_trials'
        elif record['result']['cancer_treatment_trial'] == 'yes':
            if merge and len(cohorts) > 1:
                # Training keeps one row per distinct population; the extraction stays in the checkpoint.
                merged, merging = merge_equivalent_cohorts(cohorts[COHORT_COLUMNS], config=config, return_metadata=True)
                record.update(unmerged_cohort_rows=record['cohort_rows'],
                              cohort_rows=json.loads(merged.to_json(orient='records')),
                              merge={key: merging[key] for key in ('schema_version', 'prompt_sha256', 'trial_groups',
                                                                   'merge_errors', 'trial_attempts')})
            if deboilerplate:
                rows = pd.DataFrame(record['cohort_rows'])
                clean, cleaning = remove_boilerplate_criteria(rows, config=config, return_metadata=True)
                record.update(deboilerplated_cohort_rows=json.loads(clean.to_json(orient='records')),
                              deboilerplate={'schema_version': cleaning['schema_version'],
                                             'prompt_sha256': cleaning['prompt_sha256'],
                                             'cohorts': cleaning['trial_cohorts'][trial],
                                             'errors': cleaning['deboilerplate_errors']})
            for key in ('cohort_rows', 'deboilerplated_cohort_rows') if deboilerplate else ('cohort_rows',):
                rows = pd.DataFrame(record[key])
                rows['cohort_count'] = len(rows)
                validate_cohort_rows(rows)
        elif record['cohort_rows']:
            raise ValueError('Rejected trial unexpectedly returned cohort rows')
    except Exception as exc:
        record.update(result=None, cohort_rows=[], error=f'{type(exc).__name__}: {exc}')
    record['completed_at'] = datetime.now(timezone.utc).isoformat()
    return record


def read_checkpoint(path, fingerprint):
    with gzip.open(path, 'rt') as stream:
        record = json.load(stream)
    if record['run_identity'] != fingerprint:
        raise ValueError(f'Checkpoint identity mismatch: {path}')
    return record


def output_paths(args):
    """Cohort rows, trial audit, decisions JSON, then (when those steps run) the rows before merging
    and the rows with boilerplate removed."""
    paths = [args.output_cohorts, args.output_trials, args.output_json]
    if args.merge_cohorts:
        paths.append(args.output_unmerged_cohorts
                     or args.output_cohorts.with_name(f'{args.output_cohorts.stem}_unmerged{args.output_cohorts.suffix}'))
    if args.deboilerplate_cohorts:
        paths.append(args.output_deboilerplated_cohorts
                     or args.output_cohorts.with_name(f'{args.output_cohorts.stem}_deboilerplated{args.output_cohorts.suffix}'))
    return paths


def export_results(frame, args, fingerprint):
    from matchminer_ai.trials.cohorts import COHORT_COLUMNS
    # Stream exports; long reasoning traces stay once per trial in checkpoints.
    from matchminer_ai.trials.cohort_merge import MERGED_FROM
    from matchminer_ai.trials.cohort_boilerplate import BOILERPLATE_COLUMN
    row_fields = ['cohort_count', 'cohort_index', 'trial_representation']
    cohort_fields = list(frame.columns) + COHORT_COLUMNS + [MERGED_FROM, *row_fields]
    audit_fields = list(frame.columns) + ['cancer_treatment_trial', 'multiple_cohorts',
                                        'cohort_count', 'cohort_result_json', 'extraction_error']
    errors, index, unmerged_index, clean_index, unremoved = [], 0, 0, 0, 0
    outputs = output_paths(args)
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    temps = [p.with_suffix(p.suffix + '.tmp') for p in outputs]
    with ExitStack() as stack:
        streams = [stack.enter_context(p.open('w', newline='')) for p in temps]
        rows_writer = csv.DictWriter(streams[0], fieldnames=cohort_fields)
        audit_writer = csv.DictWriter(streams[1], fieldnames=audit_fields)
        json_out = streams[2]
        extra = iter(streams[3:])
        # Before merging: every extracted cohort (a trial with one cohort, or not merged, is unchanged).
        unmerged_writer = csv.DictWriter(next(extra), fieldnames=list(frame.columns) + COHORT_COLUMNS + row_fields) \
            if args.merge_cohorts else None
        # The merged rows with boilerplate separated: the criteria that are not boilerplate, then the
        # boilerplate criteria in their own column (same rows and order as the merged rows).
        clean_writer = csv.DictWriter(next(extra), fieldnames=cohort_fields + [BOILERPLATE_COLUMN]) \
            if args.deboilerplate_cohorts else None
        for writer in (rows_writer, audit_writer, unmerged_writer, clean_writer):
            if writer:
                writer.writeheader()
        json_out.write('{\n')
        separator = ''
        for source in frame.astype(object).where(frame.notna(), None).to_dict('records'):
            record = read_checkpoint(args.work_dir / 'checkpoints' / (source['nct_id'] + '.json.gz'), fingerprint)
            result = record['result']
            count = len(record['cohort_rows'])
            audit_writer.writerow({**source, 'cancer_treatment_trial': (result or {}).get('cancer_treatment_trial'),
                'multiple_cohorts': (result or {}).get('multiple_cohorts'), 'cohort_count': count,
                'cohort_result_json': json.dumps(result), 'extraction_error': record.get('error')})
            if result is None:
                errors.append({'trial_id': source['nct_id'], 'error': record.get('error')})
                continue
            json_out.write(separator + json.dumps(source['nct_id']) + ': ' + json.dumps(result, ensure_ascii=False))
            separator = ',\n'
            for cohort in record['cohort_rows']:
                rows_writer.writerow({**source, **cohort, 'cohort_count': count,
                                      'cohort_index': index, 'trial_representation': REPRESENTATION})
                index += 1
            if unmerged_writer:
                extracted = record.get('unmerged_cohort_rows', record['cohort_rows'])
                for cohort in extracted:
                    unmerged_writer.writerow({**source, **{k: v for k, v in cohort.items() if k != MERGED_FROM},
                                              'cohort_count': len(extracted), 'cohort_index': unmerged_index,
                                              'trial_representation': REPRESENTATION})
                    unmerged_index += 1
            if clean_writer:
                unremoved += len((record.get('deboilerplate') or {}).get('errors') or {})
                for cohort in record.get('deboilerplated_cohort_rows', []):  # none for rejected trials
                    clean_writer.writerow({**source, **cohort, 'cohort_count': count,
                                           'cohort_index': clean_index, 'trial_representation': REPRESENTATION})
                    clean_index += 1
        json_out.write('\n}\n')
    partial = errors and not args.allow_failures
    for temporary, path in zip(temps, outputs):
        destination = path.with_suffix('.partial' + path.suffix) if partial else path
        temporary.replace(destination)
    atomic_json(args.work_dir / 'errors.json', errors)
    if partial:
        raise RuntimeError(f'{len(errors)} trials failed; partial exports saved. Rerun to retry failures before sampling.')
    if errors:
        print(f'{len(errors)} trials failed every attempt and have no cohorts (errors.json)', flush=True)
    print(f'Exported {len(frame)} decisions and {index} cohort rows'
          + (f' ({unmerged_index} before merging)' if args.merge_cohorts else '')
          + (f'; {clean_index} with boilerplate removed' if args.deboilerplate_cohorts else ''), flush=True)
    if unremoved:
        print(f'{unremoved} cohorts kept their boilerplate: the boilerplate step failed every attempt', flush=True)


def run(args, config):
    from matchminer_ai.config import config_snapshot
    from matchminer_ai.trials.cohort_boilerplate import DEBOILERPLATE_PROMPT, DEBOILERPLATE_SCHEMA_VERSION
    from matchminer_ai.trials.cohort_merge import MERGE_PROMPT, MERGE_SCHEMA_VERSION
    from matchminer_ai.trials.cohort_workflow import SCHEMA_VERSION, load_prompts, prompt_sha256
    from matchminer_ai.llm.prompts import load_prompt_text
    import matchminer_ai
    if args.trials_jsonl:
        frame = load_sampled_trials(args.trials_jsonl)
    else:
        frame = select_trials(read_clinical_csv(args.input), start_year=args.start_year, seed=args.seed)
    package = Path(matchminer_ai.__file__).parent
    source = args.trials_jsonl or args.input
    identity = dict(input=str(source.resolve()), input_stamp=file_stamp(source),
        selection='sampled_trials' if args.trials_jsonl else 'registry_export',
        start_year=args.start_year, seed=args.seed,
        schema=SCHEMA_VERSION, prompt_sha256=prompt_sha256(load_prompts(config.trial['cohorts'].get('prompt_files'))),
        merge=dict(schema=MERGE_SCHEMA_VERSION, prompt_sha256=hashlib.sha256(load_prompt_text(MERGE_PROMPT).encode()).hexdigest())
        if args.merge_cohorts else None,
        deboilerplate=dict(schema=DEBOILERPLATE_SCHEMA_VERSION,
                           prompt_sha256=hashlib.sha256(load_prompt_text(DEBOILERPLATE_PROMPT).encode()).hexdigest())
        if args.deboilerplate_cohorts else None,
        config=config_snapshot(config), source_sha256={str(p.relative_to(package)): sha_file(p)
            for p in sorted(package.rglob('*')) if p.suffix in ('.py', '.txt', '.yaml')},
        runner_sha256=sha_file(__file__), adapter_sha256=sha_file(ROOT / 'trial_cohort_format.py'),
        outputs=[str(p.resolve()) for p in output_paths(args)])
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    manifest = args.work_dir / 'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['run_identity'] != fingerprint:
        raise ValueError('Extraction inputs, prompt, code or settings changed; use a fresh --work-dir')
    if not manifest.exists() and any(p.exists() for p in output_paths(args)):
        raise ValueError('Outputs already exist without this run identity; use fresh output paths')
    atomic_json(manifest, dict(run_identity=fingerprint, identity=identity,
        selected_trials=len(frame), api='matchminer_ai.trials.summarize_trials'))
    directory = args.work_dir / 'checkpoints'
    directory.mkdir(exist_ok=True)
    pending, completed = [], 0
    for row in frame.to_dict('records'):
        path = directory / (row['nct_id'] + '.json.gz')
        if path.exists() and read_checkpoint(path, fingerprint)['result'] is not None:
            completed += 1
        else:
            pending.append(row)
    print(f'Selected {len(frame)} trials; resuming {completed} validated decisions; extracting {len(pending)}', flush=True)
    if pending:
        # Resolve once before launching workers, including an honest fallback for
        # served aliases. Avoid thousands of simultaneous Hub metadata lookups.
        from matchminer_ai.llm.backends import get_endpoint_model_metadata
        metadata = get_endpoint_model_metadata(args.model, cache_dir=config.model_metadata_cache_dir)
        atomic_json(Path(config.model_metadata_cache_dir) / (args.model.replace('/', '_') + '.json'), metadata)
    iterator = iter(pending)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        while True:
            while len(futures) < args.workers:
                row = next(iterator, None)
                if row is None:
                    break
                futures[executor.submit(extract_one, row, config, args.merge_cohorts, args.deboilerplate_cohorts)] = row['nct_id']
            if not futures:
                break
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                trial = futures.pop(future)
                record = future.result()
                record['run_identity'] = fingerprint
                atomic_json(directory / (trial + '.json.gz'), record)
                completed += 1
            atomic_json(args.work_dir / 'status.json', dict(processed=completed, total=len(frame)))
    export_results(frame, args, fingerprint)
    atomic_json(args.work_dir / 'status.json', dict(processed=completed, total=len(frame), status='completed'))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    data = ROOT.parent / 'data/no_phi'
    p.add_argument('--input', type=Path, default=data / 'ctgov_trials.csv')
    p.add_argument('--trials-jsonl', type=Path,
                   help='Already-sampled registry records (nct_id, title, brief_summary, eligibility_criteria), '
                        'such as a synthetic protocol batch seeds.jsonl; all are extracted, replacing --input selection')
    p.add_argument('--output-trials', type=Path, default=data / 'trials_with_cohorts.csv')
    p.add_argument('--output-cohorts', type=Path, default=data / 'trial_cohort_lineitems.csv')
    p.add_argument('--output-json', type=Path, default=data / 'trial_cohorts.json')
    p.add_argument('--output-unmerged-cohorts', type=Path,
                   help='Cohort rows before merging; default <output-cohorts stem>_unmerged.csv (not with --no-merge-cohorts)')
    p.add_argument('--output-deboilerplated-cohorts', type=Path,
                   help='Merged cohort rows with boilerplate removed; default <output-cohorts stem>_deboilerplated.csv '
                        '(not with --no-deboilerplate-cohorts)')
    p.add_argument('--work-dir', type=Path, default=data / 'trial_cohort_extraction')
    p.add_argument('--inference-repo', type=Path, default=ROOT.parent / 'matchminer-ai-inference-kehl')
    p.add_argument('--start-year', type=int, default=2020)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--model', default='nvidia/Gemma-4-31B-IT-NVFP4')
    p.add_argument('--tokenizer-model', help='Underlying model family when --model is a served alias')
    endpoints = p.add_mutually_exclusive_group(required=True)
    endpoints.add_argument('--server_urls', '--server-urls', nargs='+')
    endpoints.add_argument('--server_urls_file', '--server-urls-file', type=Path)
    p.add_argument('--max_concurrent_per_server', type=int, default=768,
                   help='Adaptive ceiling when registry entries do not specify upstream_concurrency')
    p.add_argument('--workers', type=int, default=1536, help='Total in-flight public API calls')
    p.add_argument('--request_timeout', type=float, default=7200)
    p.add_argument('--recovery-timeout', type=float, default=86400)
    p.add_argument('--max-attempts', type=int, choices=range(1, 11), default=3)
    p.add_argument('--allow-failures', action='store_true',
                   help='Export complete files even if some trials fail every attempt (listed in errors.json); '
                        'for registry versions of sampled trials, whose protocol versions remain')
    p.add_argument('--max-tokens', type=int, default=50000)
    p.add_argument('--no-merge-cohorts', dest='merge_cohorts', action='store_false',
                   help='Keep every extracted cohort instead of combining those with essentially the same criteria')
    p.add_argument('--no-deboilerplate-cohorts', dest='deboilerplate_cohorts', action='store_false',
                   help='Do not remove boilerplate criteria (no --output-deboilerplated-cohorts export)')
    p.add_argument('--api-key-env', default='OPENAI_API_KEY')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if min(args.workers, args.max_concurrent_per_server, args.max_tokens,
           args.request_timeout, args.recovery_timeout) <= 0:
        raise ValueError('Concurrency, output budget and timeouts must be positive')
    destinations = [p.resolve() for p in output_paths(args)]
    if len(set(destinations)) != len(destinations) or (args.trials_jsonl or args.input).resolve() in destinations:
        raise ValueError('Input and all output paths must be distinct')
    source = args.inference_repo.resolve() / 'src'
    if not (source / 'matchminer_ai/trials/cohort_workflow.py').is_file():
        raise ValueError('Expected the cohort API in --inference-repo')
    sys.path.insert(0, str(source))
    if args.api_key_env != 'OPENAI_API_KEY':
        os.environ['OPENAI_API_KEY'] = os.environ.get(args.api_key_env, '')
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    needed = 4 * args.workers + 1024
    if soft != resource.RLIM_INFINITY and soft < needed:
        target = needed if hard == resource.RLIM_INFINITY else min(needed, hard)
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    args.work_dir.mkdir(parents=True, exist_ok=True)
    with (args.work_dir / 'extraction.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args, make_config(args))


if __name__ == '__main__':
    main()
