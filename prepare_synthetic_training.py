#!/usr/bin/env python3
"""Adapt generated synthetic notes and final summaries to training interfaces.

Generated ages provide approximate intervals, not calendar dates. Missing dates
are explicitly synthetic: preserve event order and age intervals, ending at a
fixed configurable anchor. Existing complete dates are retained. Never infer
dates from filenames or use this adapter for real patient records.
"""
import argparse
from datetime import date, timedelta
from itertools import groupby
import json
from pathlib import Path
import re
import sqlite3

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def split_name(value):
    names = {'train': 'train', 'val': 'validation', 'validation': 'validation', 'test': 'test'}
    if value not in names:
        raise ValueError(f'Missing or unknown trial split: {value!r}')
    return names[value]


def load_spaces(path):
    frame = pd.read_csv(path)
    required = ['space_index', 'nct_id', 'this_space', 'split', 'trial_boilerplate_text']
    if any(c not in frame for c in required) or frame.empty:
        raise ValueError('Sampled spaces need IDs, text, boilerplate, and source trial splits')
    if frame[['space_index', 'nct_id', 'this_space', 'split']].isna().any().any():
        raise ValueError('Missing space identity or split')
    if frame.space_index.duplicated().any():
        raise ValueError('Duplicate space_index')
    frame['split'] = frame.split.map(split_name)
    if frame.groupby('nct_id').split.nunique().gt(1).any():
        raise ValueError('A trial crosses dataset splits')
    return frame.set_index('space_index', drop=False)


def age_years(text):
    match = re.search(r'\bage(?:d)?\s+(\d+(?:\.\d+)?)\s*(months?|years?)?'
                      r'(?:\s+and\s+(\d+)\s+weeks?\b)?', str(text), re.I)
    if not match:
        return None
    age = float(match[1])
    if (match[2] or '').lower().startswith('month'):
        age /= 12
    if match[3] is not None:
        if not (match[2] or '').lower().startswith('year') or not 0 <= int(match[3]) <= 51:
            raise ValueError('Expected whole years plus 0 through 51 additional weeks')
        # ordered_dates converts years back to days; one explicit week is seven days.
        age += int(match[3]) * 7 / 365.2425
    return age if 0 <= age <= 130 else None


def ordered_dates(rows, anchor):
    """rows: (row_id, event_number, age, supplied_date), in source event order."""
    supplied = [row[3] for row in rows]
    if any(value is not None for value in supplied):
        if not all(value is not None for value in supplied):
            raise ValueError('Partially dated patient history; do not mix provided and invented dates')
        parsed = pd.to_datetime(supplied, errors='raise', utc=True)
        if not parsed.is_monotonic_increasing:
            raise ValueError('Provided dates contradict source event order')
        return [(row[0], stamp.isoformat(), 'provided') for row, stamp in zip(rows, parsed)]
    known = [i for i, row in enumerate(rows) if row[2] is not None]
    if known:
        ages = np.interp([r[1] for r in rows], [rows[i][1] for i in known], [rows[i][2] for i in known])
        offsets = [round((age - ages[0]) * 365.2425) for age in ages]
        provenance = 'synthetic_age_and_event_order'
    else:
        offsets = list(range(len(rows)))
        provenance = 'synthetic_event_order_only'
    for i in range(1, len(offsets)):
        offsets[i] = max(offsets[i], offsets[i - 1] + 1)
    end = date.fromisoformat(anchor)
    return [(row[0], (end + timedelta(days=offset - offsets[-1])).isoformat(), provenance)
            for row, offset in zip(rows, offsets)]


def prepare_notes(source, spaces_path, destination, patients_path, *, anchor='2025-01-01'):
    source, destination, patients_path = map(Path, (source, destination, patients_path))
    if source.resolve() in (destination.resolve(), patients_path.resolve()):
        raise ValueError('Prepared outputs must not overwrite the original generated notes')
    spaces = load_spaces(spaces_path)
    trial = spaces.to_dict('index')
    required = ['row_id', 'pseudo_mrn', 'patient_id', 'space_index', 'nct_id',
                'patient_class', 'event_number', 'event_text', 'synthetic_note']
    parquet = pq.ParquetFile(source)
    if any(name not in parquet.schema_arrow.names for name in required):
        raise ValueError('Generated notes lack patient/space/event provenance')
    destination.parent.mkdir(parents=True, exist_ok=True)
    patients_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    patient_temp = patients_path.with_suffix(patients_path.suffix + '.tmp')
    database = destination.with_suffix('.preparation.sqlite')
    database.unlink(missing_ok=True)
    db = sqlite3.connect(database)
    writer = patient_writer = None
    count = 0
    try:
        db.execute('PRAGMA foreign_keys=ON')
        db.execute('PRAGMA journal_mode=OFF')
        db.executescript('''
            CREATE TABLE patients (mrn TEXT PRIMARY KEY, patient_id TEXT UNIQUE,
                space_index INTEGER, patient_class TEXT, nct_id TEXT, split TEXT,
                UNIQUE(mrn, patient_id, space_index, patient_class));
            CREATE TABLE notes (row_id INTEGER PRIMARY KEY, mrn TEXT, patient_id TEXT,
                space_index INTEGER, patient_class TEXT, event_number INTEGER, age REAL,
                date TEXT, date_provenance TEXT,
                UNIQUE(patient_id, event_number),
                FOREIGN KEY(mrn,patient_id,space_index,patient_class)
                REFERENCES patients(mrn,patient_id,space_index,patient_class));
        ''')
        columns = required + [c for c in ('date', 'split') if c in parquet.schema_arrow.names]
        for batch in parquet.iter_batches(batch_size=1024, columns=columns):
            for row in batch.to_pylist():
                if any(row[k] is None for k in required):
                    raise ValueError('Missing note content or provenance')
                space = trial.get(row['space_index'])
                if space is None or space['nct_id'] != row['nct_id']:
                    raise ValueError('Note does not belong to the supplied sampled trial spaces')
                if row['patient_class'] not in ('positive', 'negative'):
                    raise ValueError('Unknown synthetic patient class')
                if not row['synthetic_note'].strip() or row['synthetic_note'].lstrip().startswith('ERROR:'):
                    raise ValueError('Empty or failed synthetic note')
                if row.get('split') is not None and split_name(row['split']) != space['split']:
                    raise ValueError('Patient and originating trial splits disagree')
                key = (str(row['pseudo_mrn']), row['patient_id'], row['space_index'], row['patient_class'])
                db.execute('INSERT OR IGNORE INTO patients VALUES (?,?,?,?,?,?)', (*key, row['nct_id'], space['split']))
                db.execute('INSERT INTO notes VALUES (?,?,?,?,?,?,?,?,?)',
                           (row['row_id'], *key, row['event_number'], age_years(row['event_text']),
                            str(row['date']) if row.get('date') is not None else None, None))
            db.commit()
        coverage = {(space, kind): count for space, kind, count in db.execute(
            'SELECT space_index,patient_class,COUNT(*) FROM patients GROUP BY space_index,patient_class')}
        expected = {(int(space), kind): 5 for space in spaces.index for kind in ('positive', 'negative')}
        if coverage != expected:
            raise ValueError('Expected five positive and five negative patients for every sampled space')
        cursor = db.execute('SELECT mrn,row_id,event_number,age,date FROM notes ORDER BY mrn,event_number')
        for _, rows in groupby(cursor, key=lambda r: r[0]):
            dated = ordered_dates([row[1:] for row in rows], anchor)
            db.executemany('UPDATE notes SET date=?,date_provenance=? WHERE row_id=?',
                           [(day, provenance, key) for key, day, provenance in dated])
        db.commit()
        for batch in parquet.iter_batches(batch_size=1024):
            table = pa.Table.from_batches([batch])
            ids = table['row_id'].to_pylist()
            query = 'SELECT row_id,date,date_provenance,split FROM notes JOIN patients USING(mrn) WHERE row_id IN (' + ','.join('?' for _ in ids) + ')'
            mapped = {row[0]: row[1:] for row in db.execute(query, ids)}
            for name, position in [('date', 0), ('date_provenance', 1), ('split', 2)]:
                values = pa.array([mapped[key][position] for key in ids], type=pa.string())
                index = table.schema.get_field_index(name)
                table = table.set_column(index, name, values) if index >= 0 else table.append_column(name, values)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression='zstd')
            writer.write_table(table)
            count += len(table)
        if writer is None:
            raise ValueError('No synthetic notes to prepare')
        writer.close()
        writer = None
        cursor = db.execute('SELECT mrn,patient_id,space_index,patient_class,nct_id,split FROM patients ORDER BY mrn')
        while rows := cursor.fetchmany(1024):
            frame = pd.DataFrame(rows, columns=['pseudo_mrn', 'patient_id', 'space_index', 'patient_class', 'nct_id', 'split'])
            for name in ('this_space', 'trial_boilerplate_text'):
                frame[name] = frame.space_index.map(spaces[name]).fillna('')
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if patient_writer is None:
                patient_writer = pq.ParquetWriter(patient_temp, table.schema, compression='zstd')
            patient_writer.write_table(table)
        patient_writer.close()
        patient_writer = None
        report = dict(notes=count, patients=db.execute('SELECT COUNT(*) FROM patients').fetchone()[0],
                      date_anchor=anchor, date_provenance=dict(db.execute('SELECT date_provenance,COUNT(*) FROM notes GROUP BY date_provenance')),
                      source=str(source.resolve()), sampled_spaces=str(Path(spaces_path).resolve()))
        temporary.replace(destination)
        patient_temp.replace(patients_path)
        destination.with_suffix('.preparation.json').write_text(json.dumps(report, indent=2) + '\n')
        return report
    finally:
        if writer: writer.close()
        if patient_writer: patient_writer.close()
        db.close()
        database.unlink(missing_ok=True)


def prepare_summaries(source, patients_path, destination):
    frame, patients = pd.read_parquet(source), pd.read_parquet(patients_path)
    for table in (frame, patients):
        if table.pseudo_mrn.isna().any():
            raise ValueError('Missing patient ID')
        table['pseudo_mrn'] = table.pseudo_mrn.astype(str)
        if table.pseudo_mrn.duplicated().any():
            raise ValueError('Expected exactly one final summary/metadata row per patient')
    if set(frame.pseudo_mrn) != set(patients.pseudo_mrn):
        raise ValueError('Summary patient coverage differs from the generated patients')
    if frame.patient_summary.isna().any() or frame.patient_summary.str.strip().eq('').any() or frame.patient_summary.str.lstrip().str.startswith('ERROR:').any():
        raise ValueError('Empty or failed patient summaries must be resolved before training')
    for column in set(patients) & set(frame) - {'pseudo_mrn'}:
        left = frame.set_index('pseudo_mrn')[column].sort_index().fillna('')
        right = patients.set_index('pseudo_mrn')[column].sort_index().fillna('')
        if not left.equals(right):
            raise ValueError(f'Conflicting summary metadata: {column}')
        frame = frame.drop(columns=column)
    output = frame.merge(patients, on='pseudo_mrn', validate='one_to_one')
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    output.to_parquet(temporary, index=False)
    temporary.replace(path)
    return dict(patients=len(output), splits=output.split.value_counts().to_dict())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='stage', required=True)
    notes = sub.add_parser('notes')
    notes.add_argument('--input', required=True)
    notes.add_argument('--sampled-spaces', required=True)
    notes.add_argument('--output', required=True)
    notes.add_argument('--patients', required=True)
    notes.add_argument('--date-anchor', default='2025-01-01')
    summaries = sub.add_parser('summaries')
    summaries.add_argument('--input', required=True)
    summaries.add_argument('--patients', required=True)
    summaries.add_argument('--output', required=True)
    args = parser.parse_args()
    result = (prepare_notes(args.input, args.sampled_spaces, args.output, args.patients, anchor=args.date_anchor)
              if args.stage == 'notes' else prepare_summaries(args.input, args.patients, args.output))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
