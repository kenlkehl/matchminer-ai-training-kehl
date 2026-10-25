"""Lossless adapters and provenance-preserving experiment files."""
from pathlib import Path
import hashlib
import json

import pandas as pd


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path, value, *, compact=False):
    path = Path(path)
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


def combined_patient_history(row):
    from matchminer_ai.patients.postprocess import normalize_patient_summary
    if text(row.get('combined_patient_summary')):
        return normalize_patient_summary(row['combined_patient_summary'])
    cancer = text(row.get('patient_summary')) or text(row.get('cancer_history_summary'))
    other = text(row.get('other_relevant_history')) or text(row.get('general_exclusion_criteria_evidence')) or text(row.get('patient_boilerplate_text'))
    if not cancer:
        raise ValueError('Missing patient summary')
    return normalize_patient_summary(cancer, other)


def read_table(path):
    path = Path(path)
    if path.suffix == '.parquet':
        return pd.read_parquet(path)
    if path.suffix == '.jsonl':
        return pd.DataFrame(read_jsonl(path))
    return pd.read_csv(path)


def split_name(value):
    value = 'val' if value == 'validation' else value
    if value not in ('train', 'val', 'test'):
        raise ValueError('Every input needs an explicit train/val/test split; no random pair splits')
    return value


def prepare(patients_path, cohorts_path, output):
    from trial_cohort_format import training_rows
    patients, cohorts = read_table(patients_path), read_table(cohorts_path)
    if 'cohort_trial_id' not in cohorts:
        raise ValueError('Use the native complete cohort table, not historical trial spaces')
    normalized = training_rows(cohorts)
    patient_rows = []
    for row in patients.to_dict('records'):
        identifier = row.get('patient_id', row.get('pseudo_mrn'))
        if pd.isna(identifier):
            raise ValueError('Missing patient identifier')
        item = dict(id=str(identifier), text=combined_patient_history(row), split=split_name(row.get('split')))
        origin = row.get('source_trial_id', row.get('nct_id'))
        if isinstance(origin, str) and origin.strip():
            item['source_trial_id'] = origin
        patient_rows.append(item)
    cohort_rows = [dict(id=row['cohort_trial_id'], trial_id=row['nct_id'],
                       text=row['cohort_eligibility_criteria'], split=split_name(row.get('split')))
                   for row in normalized.to_dict('records')]
    validate_corpus(patient_rows, cohort_rows)
    output = Path(output)
    write_jsonl(output / 'patients.jsonl', patient_rows)
    write_jsonl(output / 'cohorts.jsonl', cohort_rows)
    write_json(output / 'corpus.json', dict(patients=len(patient_rows), cohorts=len(cohort_rows),
        patient_sha256=digest(patient_rows), cohort_sha256=digest(cohort_rows),
        source_patients=str(Path(patients_path).resolve()), source_cohorts=str(Path(cohorts_path).resolve())))


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


def sample_v22(patients_path, cohorts_path, output, *, train_count=256, val_count=64, seed=42):
    """Freeze a diverse patient sample without moving source trials across splits."""
    from trial_cohort_format import file_sha256
    output = Path(output)
    signature = dict(patient_source=str(Path(patients_path).resolve()),
                     cohort_source=str(Path(cohorts_path).resolve()),
                     patient_sha256=file_sha256(patients_path), cohort_sha256=file_sha256(cohorts_path),
                     train_count=train_count, val_count=val_count, seed=seed)
    path = output / 'sample_manifest.json'
    if path.exists():
        saved = json.loads(path.read_text())
        if saved['request'] != signature:
            raise ValueError('Patient sampling identity changed; use a new corpus directory')
        load_corpus(output)
        return saved
    columns = ['pseudo_mrn', 'patient_summary', 'patient_boilerplate_text', 'nct_id', 'split']
    import pyarrow.parquet as pq
    available = pq.read_schema(patients_path).names
    patients = pd.read_parquet(patients_path, columns=[c for c in columns if c in available])
    cohorts = read_table(cohorts_path)
    trial_splits = cohorts[['trial_id', 'split']].drop_duplicates().set_index('trial_id')['split']
    if not trial_splits.index.is_unique:
        raise ValueError('Trial cohorts disagree on their split')
    mapped = patients.nct_id.map(trial_splits)
    conflicts = mapped.notna() & mapped.ne(patients.split)
    eligible = patients.loc[~conflicts].copy()
    sampled = []
    for split, count in [('train', train_count), ('val', val_count)]:
        # One patient per source trial first; additional patients only if requested.
        pool = eligible[eligible.split.eq(split)].sample(frac=1, random_state=seed)
        if len(pool) < count or count < 1:
            raise ValueError(f'Insufficient patients or invalid requested count for {split}')
        diverse = pool.drop_duplicates('nct_id')
        ordered = pd.concat([diverse, pool.loc[~pool.index.isin(diverse.index)]])
        sampled.append(ordered.iloc[:count])
    selected = pd.concat(sampled).reset_index(drop=True)
    output.mkdir(parents=True, exist_ok=True)
    patient_file = output / 'sampled_patients.parquet'
    selected.to_parquet(patient_file, index=False)
    prepare(patient_file, cohorts_path, output)
    normalized_patients, normalized_cohorts = load_corpus(output)
    lineage = [dict(patient_id=p['id'], cohort_id=c['id'], split=p['split'])
               for p in normalized_patients for c in normalized_cohorts
               if p.get('source_trial_id') == c['trial_id'] and p['split'] == c['split']]
    write_jsonl(output / 'source_trial_pairs.jsonl', lineage)
    saved = dict(request=signature, source_patient_rows=len(patients), conflicting_origin_splits=int(conflicts.sum()),
                 sampled_patients=len(selected), patient_source_trials=int(selected.nct_id.nunique()),
                 patient_splits=selected.split.value_counts().to_dict(), cohort_rows=len(normalized_cohorts),
                 trials=int(cohorts.trial_id.nunique()), source_trial_pairs=len(lineage),
                 bridge='Other relevant history:', all_trial_cohorts_retained=True)
    write_json(path, saved)
    return saved
