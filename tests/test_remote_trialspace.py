from argparse import Namespace
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from scripts.remote_trialspace import maybe_run_remote, project_labels, remote_path
import scripts.remote_trialspace as remote


def test_projection_preserves_all_training_inputs_and_omits_teacher_trace(tmp_path):
    frame = pd.DataFrame({
        'patient_summary': ['Synthetic history', 'Synthetic history 2'],
        'this_space': ['Space A', 'Space B'],
        'eligibility_result': [0, 4],
        'split': ['train', 'val'],
        'patient_long_text': ['', 'Synthetic note'],
        'trialcheck_llm_reasoning': ['Private teacher trace', 'Teacher trace 2'],
        'trialcheck_llm_response': ['Final score: 0', 'Final score: 4'],
    })
    source, target = tmp_path / 'labels.parquet', tmp_path / 'projected.parquet'
    frame.to_parquet(source, index=False)
    transfer = project_labels(source, target)
    expected = frame.drop(columns=['trialcheck_llm_reasoning', 'trialcheck_llm_response'])
    pd.testing.assert_frame_equal(pd.read_parquet(target), expected)
    assert transfer['rows'] == pq.ParquetFile(source).metadata.num_rows == 2
    assert not target.with_suffix('.partial.parquet').exists()


def test_projection_rejects_missing_score(tmp_path):
    source = tmp_path / 'invalid.parquet'
    pd.DataFrame({'patient_summary': ['synthetic'], 'this_space': ['space']}).to_parquet(source)
    with pytest.raises(ValueError, match='eligibility_result'):
        project_labels(source, tmp_path / 'projected.parquet')


def test_remote_paths_cannot_escape_approved_workspace(tmp_path):
    assert remote_path(tmp_path / 'models/model', tmp_path, '/home/user/mmai') == '/home/user/mmai/models/model'
    with pytest.raises(ValueError):
        remote_path(tmp_path.parent / 'other', tmp_path, '/home/user/mmai')


def test_local_training_is_unchanged_without_activation(tmp_path):
    assert maybe_run_remote(Namespace(output_model=tmp_path / 'model'), tmp_path) is False


def test_refinement_maps_completed_round_one_and_rejects_original_base(tmp_path):
    previous = tmp_path / 'models/trialspace_round1'
    previous.mkdir(parents=True)
    output = previous.with_name('trialspace_round2')
    with pytest.raises(ValueError, match='missing'):
        remote.remote_base_model(str(previous), output, tmp_path, '/cloud')
    (previous / 'modules.json').write_text('[]')
    assert remote.remote_base_model(str(previous), output, tmp_path, '/cloud') == '/cloud/models/trialspace_round1'
    for wrong in ('google/embeddinggemma-2', str(tmp_path / 'other_run/trialspace_round1')):
        with pytest.raises(ValueError, match='initialize'):
            remote.remote_base_model(wrong, output, tmp_path, '/cloud')


def test_cloud_rejects_cumulative_labels_or_wrong_initial_weights():
    from copy import deepcopy
    from scripts.cloud_trialspace_job import validate
    root = '/home/klkehl/mmai'
    spec = {'remote_root': root, 'job': 'trialspace_round2', 'gpu_count': 8,
            'base_model': root + '/models/gemma_pipeline_response_v3/trialspace_round1',
            'output': root + '/models/gemma_pipeline_response_v3/trialspace_round2',
            'checkpoints': root + '/models/gemma_pipeline_response_v3/trialspace_round2_checkpoints',
            'inputs': [root + f'/cloud_workers/jobs/trialspace_round2/labels-{i}.parquet' for i in range(2)],
            'projection': [{'source': f'/run/top_{d}_tocheck_round2_labels/labels.parquet'}
                           for d in ('cohorts', 'patients')]}
    validate(spec)
    for field, value in [('base_model', 'google/embeddinggemma-2'),
                         ('inputs', spec['inputs'] * 2),
                         ('projection', [{'source': p['source'].replace('round2', 'round1')}
                                         for p in spec['projection']])]:
        wrong = deepcopy(spec)
        wrong[field] = value
        with pytest.raises(ValueError):
            validate(wrong)


def test_resume_does_not_overwrite_active_labels_and_recovers_copyback(tmp_path, monkeypatch):
    root = tmp_path / 'training'
    output = tmp_path / 'models/trialspace_round1'
    output.mkdir(parents=True)
    (output / 'modules.json').write_text('[]')
    (output / 'model.safetensors').write_bytes(b'completed model')
    control = root / '.mmai_cache/local-monitor/gcp_pool_config.json'
    remote.atomic_json(control, {'enabled': True, 'desired_running': True})
    config = {'remote_root': '/home/user/mmai', 'run_id': 'run', 'execution_overlay': {}, 'training_commit': 'training', 'inference_commit': 'inference'}
    spec = {'operation': 'run/trialspace_round1', 'execution_overlay': {},
            'base_model': 'google/embeddinggemma-2', 'training_commit': 'training', 'inference_commit': 'inference', 'projection': [], 'output': '/home/user/mmai/models/trialspace_round1'}
    state = {'operation': spec['operation'], 'status': 'complete',
             'model_files': {'modules.json': 2, 'model.safetensors': len(b'completed model')}}

    class FakeCloud:
        host = 'vm'
        def __init__(self, config):
            pass
        def retry(self, operation, description):
            return operation()
        def ssh(self, command):
            if command.startswith('python3 -c'):
                return json.dumps({'status': state, 'spec': spec})
            if command.startswith('cat '):
                return json.dumps(state)
            assert ' submit --spec ' in command
            return ''
        def scp(self, *args, **kwargs):
            pytest.fail('Resuming a saved model must not reupload active labels or download again')

    monkeypatch.setattr(remote, 'Cloud', FakeCloud)
    monkeypatch.setattr(remote, 'project_labels', lambda *args: pytest.fail('Active inputs must remain untouched'))
    remote.execute(Namespace(output_model=output, base_model=spec['base_model'], input_parquet=[]), config, root)
    receipt = root / '.mmai_cache/remote_trialspace/trialspace_round1/receipt.json'
    assert json.loads(receipt.read_text())['status'] == 'copied_back'
    assert json.loads(control.read_text())['enabled'] is True


def test_failed_remote_job_never_publishes_model(tmp_path, monkeypatch):
    root = tmp_path / 'training'
    output = tmp_path / 'models/trialspace_round1'
    remote.atomic_json(root / '.mmai_cache/local-monitor/gcp_pool_config.json',
                       {'enabled': True, 'desired_running': True})
    spec = {'operation': 'run/trialspace_round1', 'execution_overlay': {},
            'base_model': 'google/embeddinggemma-2', 'training_commit': 'training', 'inference_commit': 'inference', 'projection': []}
    calls = 0

    class FakeCloud:
        def __init__(self, config):
            pass
        def retry(self, operation, description):
            return operation()
        def ssh(self, command):
            if command.startswith('python3 -c'):
                return json.dumps({'status': {**spec, 'status': 'running'}, 'spec': spec})
            if command.startswith('cat '):
                return json.dumps({**spec, 'status': 'failed', 'error': 'training error'})
            return ''

    monkeypatch.setattr(remote, 'Cloud', FakeCloud)
    with pytest.raises(RuntimeError, match='training error'):
        remote.execute(Namespace(output_model=output, base_model=spec['base_model'], input_parquet=[]),
                       {'remote_root': '/home/user/mmai', 'run_id': 'run', 'execution_overlay': {}, 'training_commit': 'training', 'inference_commit': 'inference'}, root)
    assert not output.exists()
