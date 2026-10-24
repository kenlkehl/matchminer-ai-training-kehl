#!/usr/bin/env python3
"""Repair derived production cohort rows from the original eligibility field.

No inference is run. Original checkpoints, prompts and model decisions remain
the immutable audit of extraction; a receipt versions this derived-file repair.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import shutil

import pandas as pd

from trial_cohort_format import REPRESENTATION, file_sha256


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def repair(directory):
    directory = Path(directory)
    with (directory / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return repair_locked(directory)


def repair_locked(directory):
    target = directory / 'cohorts.jsonl'
    receipt_path = directory / 'cohort_rows_repair.json'
    source_path = directory / 'inputs.parquet'
    decisions_path = directory / 'combined_outputs.json'
    source_sha = file_sha256(source_path)
    decisions_sha = file_sha256(decisions_path)
    current_sha = file_sha256(target)
    pending_path = receipt_path.with_suffix('.pending.json')
    if receipt_path.exists() or pending_path.exists():
        saved = json.loads((receipt_path if receipt_path.exists() else pending_path).read_text())
        if saved['inputs_sha256'] != source_sha or saved['decisions_sha256'] != decisions_sha:
            raise ValueError('Source inputs/decisions changed since the repair')
        if current_sha == saved['after_sha256']:
            if not receipt_path.exists():
                pending_path.replace(receipt_path)
            return saved
        if receipt_path.exists() or current_sha != saved['before_sha256']:
            raise ValueError('Cohort export changed since the repair')
    status = json.loads((directory / 'status.json').read_text())
    if status.get('state') != 'complete' or status.get('errors') != 0:
        raise ValueError('Repair requires a complete extraction with no unresolved errors')
    source = pd.read_parquet(source_path, columns=['nct_id', 'trial_text', 'eligibility_criteria'])
    if source.nct_id.isna().any() or not source.nct_id.is_unique:
        raise ValueError('Source trial IDs must be unique and nonempty')
    source = source.set_index('nct_id').to_dict('index')
    decisions = json.loads(decisions_path.read_text())
    if set(source) != set(decisions):
        raise ValueError('Source and decision trial IDs differ')
    expected = {tid: (len(d['cohorts']) if d['multiple_cohorts'] == 'yes' else 1)
                for tid, d in decisions.items() if d['cancer_treatment_trial'] == 'yes'}
    backup = directory / 'repairs' / 'single_cohort_eligibility_v3' / 'cohorts.before.jsonl'
    backup.parent.mkdir(parents=True, exist_ok=True)
    if backup.exists():
        if file_sha256(backup) != current_sha:
            raise ValueError('Existing backup does not match the cohort export')
    else:
        shutil.copy2(target, backup)
    temp = target.with_suffix('.repair.tmp')
    counts, ids, changed = Counter(), set(), 0
    with target.open() as src, temp.open('w') as dst:
        for line in src:
            row = json.loads(line)
            tid = row['trial_id']
            if tid not in expected or row['cohort_trial_id'] in ids:
                raise ValueError('Unknown/rejected trial or duplicate cohort')
            ids.add(row['cohort_trial_id'])
            counts[tid] += 1
            decision = decisions[tid]
            if row['multiple_cohorts'] != decision['multiple_cohorts'] or row['cancer_treatment_trial'] != 'yes':
                raise ValueError('Cohort decision mismatch')
            if decision['multiple_cohorts'] == 'no':
                raw = source[tid]['eligibility_criteria']
                if not isinstance(raw, str) or not raw.strip():
                    raise ValueError(f'Missing original eligibility_criteria for {tid}')
                old = row['cohort_eligibility_criteria']
                if old not in (source[tid]['trial_text'], raw) or row['cohort_number'] != 0:
                    raise ValueError(f'Unexpected original single-cohort row for {tid}')
                if old != raw or row['cohort_criteria_source'] != 'eligibility_criteria':
                    row['cohort_eligibility_criteria'] = raw
                    row['cohort_criteria_source'] = 'eligibility_criteria'
                    changed += 1
                    line = json.dumps(row, ensure_ascii=False) + '\n'
            else:
                number = row['cohort_number']
                specific, common = decision['cohorts'][number], decision['common_eligibility_criteria']
                full = (f'Cohort-specific eligibility criteria:\n{specific}\n\n'
                        f'Common eligibility criteria:\n{common}') if common else specific
                if (row['cohort_specific_eligibility_criteria'] != specific
                        or row['common_eligibility_criteria'] != common
                        or row['cohort_eligibility_criteria'] != full):
                    raise ValueError(f'Multiple-cohort criteria mismatch for {tid}')
            dst.write(line)
    if counts != expected or len(ids) != status['cohort_rows']:
        raise ValueError('Incomplete cohort export; original file preserved')
    if file_sha256(target) != current_sha:
        raise ValueError('Export changed during repair')
    receipt = dict(representation_version=REPRESENTATION,
        repaired_at=datetime.now(timezone.utc).isoformat(),
        extraction_run_identity=json.loads((directory / 'manifest.json').read_text())['run_identity'],
        inputs_sha256=source_sha, decisions_sha256=decisions_sha,
        before_sha256=current_sha, after_sha256=file_sha256(temp),
        backup=str(backup.resolve()), changed_single_cohort_rows=changed,
        cohort_rows=len(ids), multi_cohort_rows_unchanged=True,
        raw_checkpoint_policy='Original extraction checkpoints are retained unchanged as historical evidence.',
        repair_code_sha256=file_sha256(__file__))
    atomic_json(pending_path, receipt)
    temp.replace(target)
    pending_path.replace(receipt_path)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extraction-dir', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(repair(args.extraction_dir), indent=2))


if __name__ == '__main__':
    main()
