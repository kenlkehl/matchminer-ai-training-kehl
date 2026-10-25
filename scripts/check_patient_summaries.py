"""Audit saved summaries using inference's shared, read-only quality checks."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
LOCAL_INFERENCE = ROOT.parent / 'matchminer-ai-inference-kehl' / 'src'
if LOCAL_INFERENCE.is_dir():
    sys.path.insert(0, str(LOCAL_INFERENCE))

from matchminer_ai.patients.summary_quality import (  # noqa: E402 - use sibling checkout
    SUMMARY_QUALITY_VERSION,
    check_patient_summary,
)

TEXT_COLUMNS = ('patient_summary', 'text', 'cancer_history_summary')
ID_COLUMNS = ('patient_id', 'id', 'pseudo_mrn')


def records(path, text_column=None, id_column=None):
    if path.suffix.lower() in ('.jsonl', '.ndjson'):
        with path.open() as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)
    elif path.suffix.lower() == '.csv':
        with path.open(newline='') as stream:
            yield from csv.DictReader(stream)
    elif path.suffix.lower() == '.parquet':
        import pyarrow.parquet as pq
        source = pq.ParquetFile(path)
        wanted = set(TEXT_COLUMNS + ID_COLUMNS + ('source_split', 'split',
                     'general_exclusion_criteria_evidence', 'patient_boilerplate_text'))
        wanted.update(c for c in (text_column, id_column) if c)
        columns = [c for c in source.schema_arrow.names if c in wanted]
        for batch in source.iter_batches(batch_size=4096, columns=columns):
            yield from batch.to_pylist()
    else:
        raise ValueError('Use JSONL, CSV, or Parquet summaries')


def audit(input_path, output, *, text_column=None, id_column=None, include_text=False):
    input_path, output = Path(input_path), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    counts = Counter(total=0, flagged=0, high=0, review=0)
    by_split = defaultdict(lambda: Counter(total=0, flagged=0, high=0, review=0))
    by_code = Counter()
    sections = Counter()
    columns = ['patient_id', 'source_split', 'confidence', 'codes', 'findings']
    if include_text:
        columns.append('summary_text')
    with (output / 'flagged.jsonl').open('w') as jsonl, (output / 'flagged.csv').open('w', newline='') as stream:
        csv_out = csv.DictWriter(stream, fieldnames=columns)
        csv_out.writeheader()
        for row in records(input_path, text_column, id_column):
            tc = text_column or next((c for c in TEXT_COLUMNS if c in row), None)
            ic = id_column or next((c for c in ID_COLUMNS if c in row), None)
            if tc is None or ic is None or tc not in row or ic not in row or row[ic] is None:
                raise ValueError('Each summary needs a text column and patient identifier')
            text = row[tc] or ''
            other = row.get('general_exclusion_criteria_evidence') or row.get('patient_boilerplate_text')
            if other and 'Other relevant history:' not in text:
                text += '\n\nOther relevant history:\n' + other
            check = check_patient_summary(text)
            split = str(row.get('source_split', row.get('split', 'unspecified')))
            counts['total'] += 1
            by_split[split]['total'] += 1
            sections[str(check['cancer_sections'])] += 1
            if not check['flagged']:
                continue
            codes = sorted({f['code'] for f in check['findings']})
            confidence = 'high' if any(f['confidence']=='high' for f in check['findings']) else 'review'
            counts['flagged'] += 1
            counts[confidence] += 1
            by_split[split]['flagged'] += 1
            by_split[split][confidence] += 1
            by_code.update(codes)
            result = dict(patient_id=str(row[ic]), source_split=split, confidence=confidence,
                          codes=codes, findings=check['findings'])
            if include_text:
                result['summary_text'] = text
            jsonl.write(json.dumps(result) + '\n')
            csv_out.writerow({**result, 'codes': ';'.join(codes), 'findings': json.dumps(check['findings'])})
    with input_path.open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
    report = dict(version=SUMMARY_QUALITY_VERSION, source=str(input_path.resolve()),
                  source_sha256=digest, include_text=include_text,
                  counts=dict(counts), by_code=dict(by_code),
                  by_source_split={k:dict(v) for k,v in by_split.items()},
                  cancer_section_counts=dict(sections),
                  interpretation='Heuristic review flags, not confirmed errors. Distinct primaries may share core field values. No source records are changed.')
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path, help='New audit directory')
    parser.add_argument('--text-column')
    parser.add_argument('--id-column')
    parser.add_argument('--include-text', action='store_true', help='Include flagged summary text in local review files')
    args = parser.parse_args()
    report = audit(args.input, args.output, text_column=args.text_column,
                   id_column=args.id_column, include_text=args.include_text)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
