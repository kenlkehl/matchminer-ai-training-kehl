"""Regression checks for downstream scheduling, source reuse and resume identity."""
import json
from threading import Event, Lock

import pytest

from oncoreasoning_training import contracts
from oncoreasoning_training.create_all_training_data import check_manifest, preparation_inputs
from oncoreasoning_training.scheduling import run_tasks


def test_scheduler_refills_without_waiting_for_slow_request():
    slow_started, later_started = Event(), Event()
    def worker(task):
        if task['id'] == 'slow':
            slow_started.set()
            assert later_started.wait(3), 'Slow first request blocked refill'
        elif task['id'] == 'fast':
            assert slow_started.wait(3)
        else:
            later_started.set()
    assert run_tasks(({'id': i} for i in ('slow', 'fast', 'later')), worker, concurrency=2) == 3


def test_scheduler_preserves_serial_summary_dependencies():
    completed, lock = set(), Lock()
    tasks = [{'id': f'{p}-{n}', 'parent_id': f'{p}-{n-1}' if n else None}
             for n in range(12) for p in range(8)]
    def worker(task):
        with lock:
            assert not task['parent_id'] or task['parent_id'] in completed
            completed.add(task['id'])
    assert run_tasks(tasks, worker, concurrency=5) == len(tasks)


@pytest.mark.parametrize('old_key', ['server_urls', 'servers'])
def test_endpoint_changes_preserve_resume_but_model_changes_do_not(tmp_path, old_key):
    identity = {'teacher': 'approved-model', 'prepared': 'unchanged-prompts'}
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps({**identity, old_key: ['http://lost-spot/v1']}))
    check_manifest(tmp_path, path.name, identity, transport_keys=(old_key,))
    assert json.loads(path.read_text()) == identity
    with pytest.raises(ValueError, match='Incompatible'):
        check_manifest(tmp_path, path.name, {**identity, 'teacher': 'changed'}, transport_keys=(old_key,))


def test_preparation_reuses_original_notes_identity(tmp_path, monkeypatch):
    notes, derived = tmp_path / 'original-not-opened.parquet', tmp_path / 'round3.parquet'
    recorded = {'path': str(notes), 'sha256': 'already-recorded'}
    manifest = tmp_path / 'pipeline.json'
    manifest.write_text(json.dumps({'inputs': [recorded]}))
    def source(path):
        assert path == derived, 'Original source must not be rehashed'
        return {'path': str(path), 'sha256': 'derived-only'}
    monkeypatch.setattr('oncoreasoning_training.create_all_training_data.source_record', source)
    assert preparation_inputs([notes, derived], manifest, required_recorded=[notes])[0] == recorded
    with pytest.raises(ValueError, match='missing'):
        preparation_inputs([notes], manifest, required_recorded=[tmp_path / 'other-original'])


def test_prior_treatment_specificity_requires_response_on_therapy():
    rules = contracts.TRIAL_COMPONENTS['prior_treatment_specificity'][1]
    assert 'WHILE RECEIVING THAT THERAPY' in rules
