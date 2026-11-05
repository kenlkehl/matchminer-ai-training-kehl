"""Lossless adapters and provenance-preserving TrialSpace corpus files.

A corpus entity's ``text`` is what retrieval and the match teacher read: a patient's key summary
section (the cancer history) and a cohort's key criteria. ``boilerplate`` is what the
BoilerplateChecker teacher reads: a patient's boilerplate section (other relevant history) and a
cohort's boilerplate criteria.
"""
from pathlib import Path
import hashlib
import json

import pandas as pd
from trial_cohort_format import read_clinical_csv


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path, value, *, compact=False, create_parents=True):
    path = Path(path)
    if create_parents:
        path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    options = dict(separators=(',', ':')) if compact else dict(indent=2)
    temporary.write_text(json.dumps(value, ensure_ascii=False, **options) + '\n')
    temporary.replace(path)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    temporary.replace(path)


def text(value):
    return value if isinstance(value, str) and value.strip() else ''


def patient_sections(row):
    """(key, boilerplate): a summary's cancer history and its other relevant history.

    Accepts the separate training columns (``patient_summary`` key section plus
    ``patient_boilerplate_text``), the public ones (``cancer_history_summary`` plus
    ``general_exclusion_criteria_evidence``), or one complete summary with an ``Other relevant
    history:`` heading, which is split.
    """
    from matchminer_ai.patients.postprocess import normalize_patient_summary, split_patient_summary
    key = text(row.get('patient_summary')) or text(row.get('cancer_history_summary'))
    other = text(row.get('patient_boilerplate_text')) or text(row.get('general_exclusion_criteria_evidence'))
    if not key:
        raise ValueError('Missing patient summary')
    key_section, embedded = split_patient_summary(normalize_patient_summary(key, other or None))
    return key_section, embedded


def read_table(path):
    path = Path(path)
    if path.suffix == '.parquet':
        return pd.read_parquet(path)
    if path.suffix == '.jsonl':
        return pd.DataFrame(read_jsonl(path))
    return read_clinical_csv(path)


def split_name(value):
    value = 'val' if value == 'validation' else value
    if value not in ('train', 'val', 'test'):
        raise ValueError('Every input needs an explicit train/val/test split; no random pair splits')
    return value


def selected_splits(value):
    """Canonical source partitions; selecting them for fitting does not rename them."""
    values = [value] if isinstance(value, str) else list(value)
    if not values:
        raise ValueError('Select at least one source split')
    chosen = {split_name(item) for item in values}
    return [name for name in ('train', 'val', 'test') if name in chosen]


def prepare(patients_path, cohorts_path, output, *, merge_source_splits=False):
    from trial_cohort_format import cohort_version, training_rows
    patients, cohorts = read_table(patients_path), read_table(cohorts_path)
    if 'cohort_trial_id' not in cohorts:
        raise ValueError('Use the native complete cohort table, not historical trial spaces')
    normalized = training_rows(cohorts)
    patient_rows = []
    for row in patients.to_dict('records'):
        identifier = row.get('patient_id', row.get('pseudo_mrn'))
        if pd.isna(identifier):
            raise ValueError('Missing patient identifier')
        key, boilerplate = patient_sections(row)
        item = dict(id=str(identifier), text=key, boilerplate=boilerplate, split=split_name(row.get('split')))
        origin = row.get('source_trial_id', row.get('nct_id'))
        if isinstance(origin, str) and origin.strip():
            item['source_trial_id'] = origin
        patient_rows.append(item)
    # source: which version of its trial a cohort is ('protocol' or 'registry');
    # mining represents each trial by one version (retrieval.TrialSampler).
    key_column = 'cohort_key_criteria' if 'cohort_key_criteria' in normalized else 'cohort_eligibility_criteria'
    cohort_rows = [dict(id=row['cohort_trial_id'], trial_id=row['nct_id'], source=version,
                       text=row[key_column], boilerplate=text(row.get('cohort_boilerplate_criteria')),
                       split=split_name(row.get('split')))
                   for row, version in zip(normalized.to_dict('records'), cohort_version(normalized))]
    if 'cohort_name' in normalized:
        for row, name in zip(cohort_rows, normalized.cohort_name):
            row['cohort_name'] = name if isinstance(name, str) else None
    validate_corpus(patient_rows, cohort_rows)
    if merge_source_splits:
        # Validate the source lineage first, then make one shared retrieval and
        # fitting pool. The original assignments remain available for provenance.
        for row in patient_rows + cohort_rows:
            row['source_split'], row['split'] = row['split'], 'train'
    output = Path(output)
    write_jsonl(output / 'patients.jsonl', patient_rows)
    write_jsonl(output / 'cohorts.jsonl', cohort_rows)
    write_json(output / 'corpus.json', dict(patients=len(patient_rows), cohorts=len(cohort_rows),
        patient_sha256=digest(patient_rows), cohort_sha256=digest(cohort_rows),
        source_patients=str(Path(patients_path).resolve()), source_cohorts=str(Path(cohorts_path).resolve()),
        split_policy='all_synthetic_training' if merge_source_splits else 'preserve_source_splits'))


def validate_corpus(patients, cohorts):
    trial_splits = {}
    for records in (patients, cohorts):
        if len({r['id'] for r in records}) != len(records):
            raise ValueError('Duplicate entity IDs')
        for row in records:
            split_name(row['split'])
            if not text(row['text']):
                raise ValueError('Empty entity text')
    for row in cohorts:
        if trial_splits.setdefault(row['trial_id'], row['split']) != row['split']:
            raise ValueError('Sibling cohorts must remain in the same split')
    for row in patients:
        if row.get('source_trial_id') in trial_splits and trial_splits[row['source_trial_id']] != row['split']:
            raise ValueError('Synthetic patient and source trial cross splits')


def load_corpus(directory):
    directory = Path(directory)
    patients, cohorts = read_jsonl(directory / 'patients.jsonl'), read_jsonl(directory / 'cohorts.jsonl')
    validate_corpus(patients, cohorts)
    return patients, cohorts
