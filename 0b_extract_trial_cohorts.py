#!/usr/bin/env python3
"""Extract cancer-treatment cohorts through the sibling inference package API.

No separate screening pass or training-owned extraction prompt. Checkpoints
retain combined decisions, reasoning, validation attempts and endpoint provenance.
"""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
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


def extract_one(row, config):
    from matchminer_ai.trials import summarize_trials
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
            rows = pd.DataFrame(record['cohort_rows'])
            expected = len(record['result']['cohorts']) if record['result']['multiple_cohorts'] == 'yes' else 1
            rows['cohort_count'] = expected
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


def export_results(frame, args, fingerprint):
    from matchminer_ai.trials.cohorts import COHORT_COLUMNS
    # Stream exports; long reasoning traces stay once per trial in checkpoints.
    cohort_fields = list(frame.columns) + COHORT_COLUMNS + ['cohort_count', 'cohort_index', 'trial_representation']
    audit_fields = list(frame.columns) + ['cancer_treatment_trial', 'multiple_cohorts',
                                        'cohort_count', 'cohort_result_json', 'extraction_error']
    errors, index = [], 0
    outputs = [args.output_cohorts, args.output_trials, args.output_json]
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    temps = [p.with_suffix(p.suffix + '.tmp') for p in outputs]
    with temps[0].open('w', newline='') as rows_out, temps[1].open('w', newline='') as trials_out, temps[2].open('w') as json_out:
        rows_writer = csv.DictWriter(rows_out, fieldnames=cohort_fields)
        audit_writer = csv.DictWriter(trials_out, fieldnames=audit_fields)
        rows_writer.writeheader()
        audit_writer.writeheader()
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
        json_out.write('\n}\n')
    for temporary, path in zip(temps, outputs):
        destination = path.with_suffix('.partial' + path.suffix) if errors else path
        temporary.replace(destination)
    atomic_json(args.work_dir / 'errors.json', errors)
    if errors:
        raise RuntimeError(f'{len(errors)} trials failed; partial exports saved. Rerun to retry failures before sampling.')
    print(f'Exported {len(frame)} decisions and {index} cohort rows', flush=True)


def run(args, config):
    from matchminer_ai.config import config_snapshot
    from matchminer_ai.trials.cohort_schema import COHORT_SCHEMA_VERSION, load_cohort_prompt
    import matchminer_ai
    frame = select_trials(read_clinical_csv(args.input), start_year=args.start_year, seed=args.seed)
    package = Path(matchminer_ai.__file__).parent
    identity = dict(input_stamp=file_stamp(args.input), start_year=args.start_year, seed=args.seed,
        schema=COHORT_SCHEMA_VERSION, prompt_sha256=hashlib.sha256(load_cohort_prompt().encode()).hexdigest(),
        config=config_snapshot(config), source_sha256={str(p.relative_to(package)): sha_file(p)
            for p in sorted(package.rglob('*')) if p.suffix in ('.py', '.txt', '.yaml')},
        runner_sha256=sha_file(__file__), adapter_sha256=sha_file(ROOT / 'trial_cohort_format.py'),
        outputs=[str(p.resolve()) for p in (args.output_trials, args.output_cohorts, args.output_json)])
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    manifest = args.work_dir / 'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['run_identity'] != fingerprint:
        raise ValueError('Extraction inputs, prompt, code or settings changed; use a fresh --work-dir')
    if not manifest.exists() and any(p.exists() for p in (args.output_trials, args.output_cohorts, args.output_json)):
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
                futures[executor.submit(extract_one, row, config)] = row['nct_id']
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
    p.add_argument('--output-trials', type=Path, default=data / 'trials_with_cohorts.csv')
    p.add_argument('--output-cohorts', type=Path, default=data / 'trial_cohort_lineitems.csv')
    p.add_argument('--output-json', type=Path, default=data / 'trial_cohorts.json')
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
    p.add_argument('--max-tokens', type=int, default=50000)
    p.add_argument('--api-key-env', default='OPENAI_API_KEY')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if min(args.workers, args.max_concurrent_per_server, args.max_tokens,
           args.request_timeout, args.recovery_timeout) <= 0:
        raise ValueError('Concurrency, output budget and timeouts must be positive')
    destinations = [p.resolve() for p in (args.output_trials, args.output_cohorts, args.output_json)]
    if len(set(destinations)) != 3 or args.input.resolve() in destinations:
        raise ValueError('Input and all output paths must be distinct')
    source = args.inference_repo.resolve() / 'src'
    if not (source / 'matchminer_ai/trials/cohorts.py').is_file():
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
