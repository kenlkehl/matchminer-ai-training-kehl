"""Bounded CSV export: full trial reasoning once, individual spaces separately."""
import csv
import os
from pathlib import Path
import re

import pyarrow.parquet as pq

from ctgov_metadata import REGISTRY_METADATA_COLUMNS
from trial_space_format import has_required_space_fields

GENERATED_TRIAL_COLUMNS = (
    'space_reasoning_and_output', 'space_output_no_reasoning',
    'space_text', 'trial_boilerplate_text',
    'space_extraction_status', 'space_rejection_reason', 'space_rejection_evidence',
)
TRIAL_ONLY_COLUMNS = frozenset({
    'brief_summary', 'space_reasoning_and_output', 'space_output_no_reasoning', 'space_text',
    'space_extraction_status', 'space_rejection_reason', 'space_rejection_evidence',
})


def trial_output_columns(available_columns):
    ordered = ['nct_id', 'study_type', 'phases', *REGISTRY_METADATA_COLUMNS,
               'split', 'title', 'brief_summary', 'eligibility_criteria', 'trial_text',
               *GENERATED_TRIAL_COLUMNS]
    return [column for column in ordered if column in available_columns]


def lineitem_columns(trial_columns):
    return [column for column in trial_columns if column not in TRIAL_ONLY_COLUMNS] + [
        'this_space', 'space_number', 'reasoning_trials_path',
    ]


def iter_space_rows(row, output_trials):
    """Retain valid numbered spaces and reference their NCT row in the trial CSV.

    Drop whole-trial fields before expanding rows, so neither memory nor output
    size grows as the product of a trial's audit size and its number of spaces.
    """
    if row.get('space_extraction_status') == 'rejected':
        return
    base = {key: value for key, value in row.items() if key not in TRIAL_ONLY_COLUMNS}
    base['reasoning_trials_path'] = str(Path(output_trials).resolve())
    number = 0
    for line in str(row.get('space_text', '')).splitlines():
        if not re.match(r'^\s*\d+\.', line) or not has_required_space_fields(line):
            continue
        number += 1
        yield {**base, 'this_space': re.sub(r'^\s*\d+\.', '', line).strip(), 'space_number': number}


def write_trial_exports(shard_paths, output_trials, output_spaces, base_columns):
    """Stream local/remote, one-shot/hierarchical checkpoints with atomic outputs."""
    trial_columns = trial_output_columns([*base_columns, *GENERATED_TRIAL_COLUMNS])
    columns = [trial_columns, lineitem_columns(trial_columns)]
    destinations = [Path(output_trials), Path(output_spaces)]
    if destinations[0].resolve() == destinations[1].resolve():
        raise ValueError('Trial and space output paths must differ')
    temporary = [path.with_suffix(path.suffix + '.tmp') for path in destinations]
    streams = []
    trials, spaces, seen = 0, 0, set()
    try:
        for path in temporary:
            path.parent.mkdir(parents=True, exist_ok=True)
            streams.append(path.open('w', newline='', encoding='utf-8'))
        writers = [csv.DictWriter(stream, names, extrasaction='ignore')
                   for stream, names in zip(streams, columns, strict=True)]
        for writer in writers:
            writer.writeheader()
        for path in shard_paths:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=1):
                for row in batch.to_pylist():
                    if row['nct_id'] in seen:
                        raise ValueError(f"Duplicate trial in extraction checkpoints: {row['nct_id']}")
                    seen.add(row['nct_id'])
                    row.setdefault('space_extraction_status', 'completed')
                    writers[0].writerow(row)
                    trials += 1
                    for space in iter_space_rows(row, output_trials):
                        writers[1].writerow(space)
                        spaces += 1
        for stream in streams:
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        for stream in streams:
            stream.close()
    for source, destination in zip(temporary, destinations, strict=True):
        source.replace(destination)
    return {'trials': trials, 'spaces': spaces}
