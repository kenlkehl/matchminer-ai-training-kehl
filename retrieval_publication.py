"""Verify that a registered retrieval process is only publishing completed results."""
import hashlib
import json
from pathlib import Path

from colbert_experiments.local_teacher import process_identity
from trial_cohort_format import file_stamp


def publication_ready(command, expected_process):
    try:
        return _publication_ready(command, expected_process)
    except FileNotFoundError:
        # Publication can finish and remove scratch between our reads. The
        # controller still joins the publisher and verifies its public outputs.
        destination = Path(command[command.index('--output') + 1]).resolve()
        status = json.loads((destination/'retrieval_status.json').read_text())
        if status.get('phase') == 'complete' or process_identity(expected_process['pid']) is None:
            return None
        raise


def _publication_ready(command, expected_process):
    if process_identity(expected_process['pid']) != expected_process:
        return None
    destination = Path(command[command.index('--output') + 1]).resolve()
    corpus = Path(command[command.index('--corpus') + 1]).resolve()
    status_path = destination/'retrieval_status.json'
    if not status_path.exists():
        return None
    status = json.loads(status_path.read_text())
    if status.get('phase') != 'publishing':
        return None
    scratch = (Path(command[command.index('--retrieval-scratch-dir') + 1])
               if '--retrieval-scratch-dir' in command else Path('/tmp/mmai-retrieval'))
    workspace = scratch.resolve()/hashlib.sha256(str(destination).encode()).hexdigest()[:24]
    if Path(status['workspace']).resolve() != workspace or status.get('pid', expected_process['pid']) != expected_process['pid']:
        raise ValueError('Retrieval publication owner/workspace changed')
    saved = json.loads((workspace/'workspace.json').read_text())
    if saved['destination'] != str(destination) or saved['corpus'] != str(corpus):
        raise ValueError('Retrieval publication inputs/destination changed')
    marker = workspace/'output/complete.json'
    completed = json.loads(marker.read_text())
    required = {'candidates.jsonl', 'rankings.jsonl', 'patient_to_cohort.parquet', 'cohort_to_patient.parquet'}
    if not required.issubset(completed['outputs']):
        raise ValueError('Retrieval has not completed both directional exports')
    for name, stamp in completed['outputs'].items():
        path = workspace/'output'/name
        if not path.resolve().is_relative_to((workspace/'output').resolve()) or file_stamp(path) != stamp:
            raise ValueError('Completed local retrieval product changed: '+name)
    if process_identity(expected_process['pid']) != expected_process:
        return None
    return dict(process=expected_process, workspace=str(workspace), destination=str(destination),
                completion=completed, completion_stamp=file_stamp(marker))
