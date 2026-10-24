import fcntl
import json

import pytest

from run_pipeline import file_record
from scripts import register_event_validation_upgrade as upgrade
from generate_synthetic_data import compatible_validation_upgrade


def fixture_run(tmp_path, monkeypatch):
    repo, run = tmp_path / 'repo', tmp_path / 'run'
    repo.mkdir()
    pipeline = run / 'pipeline'
    (pipeline / 'requests').mkdir(parents=True)
    (run / 'generation_failures').mkdir()
    for path in (run / 'generation.lock', pipeline / 'pipeline.lock'):
        path.touch()
    for name in ('generate_synthetic_data.py', 'prompt.py', 'run_pipeline.py'):
        (repo / name).write_text('original ' + name)
    source = [file_record(repo / name, {}) for name in ('run_pipeline.py', 'generate_synthetic_data.py', 'prompt.py')]
    input_path = run / 'sample.csv'
    input_path.write_text('cohort\n0\n')
    manifest = dict(version='synthetic-cohort-seeds-patient-events-notes-v5',
                    input_sha256=upgrade.sha_file(input_path), seed=42,
                    source_sha256={name: upgrade.sha_file(repo / name)
                                   for name in ('generate_synthetic_data.py', 'prompt.py')})
    (run / 'manifest.json').write_text(json.dumps(manifest))
    (run / 'status.json').write_text(json.dumps(dict(stage='events')))
    identity = dict(command=['python', str(repo / 'generate_synthetic_data.py')],
                    inputs=[file_record(input_path, {})], sources=source)
    (pipeline / 'requests/generate-synthetic.json').write_text(json.dumps(dict(fingerprint=upgrade.digest(identity), **identity)))
    (repo / 'generate_synthetic_data.py').write_text('updated validator')
    monkeypatch.setattr(upgrade, 'ROOT', repo)
    return repo, run, manifest


def test_registration_keeps_original_manifest_and_is_idempotent(tmp_path, monkeypatch):
    repo, run, original = fixture_run(tmp_path, monkeypatch)
    before = (run / 'manifest.json').read_bytes()
    request_before = json.loads((run / 'pipeline/requests/generate-synthetic.json').read_text())
    receipt = upgrade.register(run)
    assert receipt == upgrade.register(run)
    assert (run / 'manifest.json').read_bytes() == before
    assert json.loads((run / 'event_validation_upgrade_backup/pipeline_request.json').read_text()) == request_before
    current = {**original, 'source_sha256': {**original['source_sha256'],
               'generate_synthetic_data.py': upgrade.sha_file(repo / 'generate_synthetic_data.py')}}
    assert compatible_validation_upgrade(original, current, receipt)
    request_after = json.loads((run / 'pipeline/requests/generate-synthetic.json').read_text())
    assert request_after['fingerprint'] == receipt['approved_request_sha256']


@pytest.mark.parametrize('lock_name', ['generation.lock', 'pipeline/pipeline.lock'])
def test_registration_rejects_live_writer(tmp_path, monkeypatch, lock_name):
    _, run, _ = fixture_run(tmp_path, monkeypatch)
    with (run / lock_name).open('r') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            upgrade.register(run)
    assert not (run / 'event_validation_upgrade.json').exists()


@pytest.mark.parametrize('changed', ['prompt.py', 'run_pipeline.py', 'sample.csv'])
def test_registration_rejects_prompt_input_and_coordinator_changes(tmp_path, monkeypatch, changed):
    repo, run, _ = fixture_run(tmp_path, monkeypatch)
    path = (run if changed == 'sample.csv' else repo) / changed
    path.write_text('unexpected change')
    with pytest.raises(ValueError, match='cannot change'):
        upgrade.register(run)
    assert not (run / 'event_validation_upgrade.json').exists()


def test_standalone_followup_preserves_prior_approval_and_coordinator_identity(tmp_path, monkeypatch):
    repo, run, original = fixture_run(tmp_path, monkeypatch)
    prior = upgrade.register(run)
    request = run / 'pipeline/requests/generate-synthetic.json'
    before = request.read_bytes()
    (repo / 'run_pipeline.py').write_text('new unrelated training orchestration')
    (repo / 'generate_synthetic_data.py').write_text('second validator correction')
    receipt = upgrade.register(run, standalone=True)
    assert request.read_bytes() == before
    assert json.loads((run / 'manifest.json').read_text()) == original
    assert receipt == upgrade.register(run, standalone=True)
    assert receipt['execution_mode'] == 'recorded-generator-only'
    assert receipt['previous_receipt_sha256'] == upgrade.digest(prior)
    archived = run / 'event_validation_upgrade_backup/receipts' / (upgrade.digest(prior) + '.json')
    assert json.loads(archived.read_text()) == prior
    (repo / 'prompt.py').write_text('unauthorized prompt change')
    with pytest.raises(ValueError, match='cannot change'):
        upgrade.register(run, standalone=True)
