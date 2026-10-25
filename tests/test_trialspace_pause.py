import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from colbert_experiments.checkpointing import deadline_timestamp, save_checkpoint
from colbert_experiments.data import read_jsonl, write_jsonl
from colbert_experiments.fixtures import create_fixture
from colbert_experiments.labeling import label_key
from colbert_experiments.regression import aggregate_pair
from scripts.label_to_trialspace import Handoff, labels_ready, save


def test_rotation_and_failed_write_preserve_committed_resume(tmp_path, monkeypatch):
    for step in range(4):
        save_checkpoint(tmp_path, dict(step=step, value=torch.tensor(step)))
        assert len(list(tmp_path.glob('checkpoint*.pt'))) <= 2
    assert sorted(p.name for p in tmp_path.glob('checkpoint*.pt')) == [
        'checkpoint-00000002.pt', 'checkpoint-00000003.pt']
    def fail(state, stream):
        stream.write(b'partial checkpoint')
        raise OSError('simulated full filesystem')
    monkeypatch.setattr(torch, 'save', fail)
    with pytest.raises(OSError):
        save_checkpoint(tmp_path, dict(step=4))
    latest = json.loads((tmp_path / 'latest.json').read_text())
    assert torch.load(tmp_path / latest['path'], weights_only=True)['step'] == 3
    assert not list(tmp_path.glob('*.tmp'))


def test_deadline_is_unambiguous():
    assert deadline_timestamp('2026-10-26T08:55:00-04:00') == 1793019300
    with pytest.raises(ValueError, match='timezone'):
        deadline_timestamp('2026-10-26T08:55:00')


def test_handoff_requires_complete_labels_and_successful_receipt(tmp_path):
    output = tmp_path / 'scores.jsonl'
    output.touch()
    receipt = dict(outputs=[str(output)])
    labels = dict(state='complete', scored_pairs=4, total_pairs=4, failures={})
    assert not labels_ready(dict(phase='labeling'), labels, receipt, 4)
    assert not labels_ready(dict(phase='completed'), labels, None, 4)
    assert labels_ready(dict(phase='completed'), labels, receipt, 4)
    with pytest.raises(ValueError, match='coverage'):
        labels_ready(dict(phase='completed'), {**labels, 'scored_pairs': 3}, receipt, 4)
    with pytest.raises(RuntimeError, match='hard-failed'):
        labels_ready(dict(phase='failed'), labels, receipt, 4)


def test_handoff_stops_cloud_then_local_teachers_before_starting_fit(tmp_path, monkeypatch):
    handoff = Handoff.__new__(Handoff)
    handoff.root = tmp_path
    handoff.workflow_root = tmp_path / 'workflow'
    handoff.pause_at = float('inf')
    handoff.config = dict(label_status=str(tmp_path / 'labels.json'), expected_pairs=2,
                         fit_status=str(tmp_path / 'fit.json'))
    output = tmp_path / 'scores.jsonl'
    output.touch()
    save(handoff.workflow_root / 'completed/labeling.json', dict(outputs=[str(output)]))
    save(handoff.workflow_root / 'status.json', dict(phase='completed'))
    save(handoff.config['label_status'], dict(state='complete', scored_pairs=2, total_pairs=2, failures={}))
    save(handoff.config['fit_status'], dict(state='complete'))
    calls = []
    monkeypatch.setattr(handoff, 'cloud_stop', lambda wait: calls.append(('cloud', wait)))
    monkeypatch.setattr(handoff, 'stop_local', lambda: calls.append(('local', True)))
    monkeypatch.setattr(handoff, 'check_gpu_availability', lambda: calls.append(('gpu_check', True)))
    monkeypatch.setattr(handoff, 'launch_training', lambda: calls.append(('train', True)) or True)
    monkeypatch.setattr(handoff, 'service', lambda: dict(ActiveState='inactive', ExecMainStatus='0'))
    handoff.run()
    assert calls == [('cloud', True), ('local', True), ('gpu_check', True), ('train', True)]
    assert json.loads((tmp_path / 'status.json').read_text())['phase'] == 'complete'


def test_reused_pid_is_never_signaled(monkeypatch):
    from scripts import label_to_trialspace as module
    monkeypatch.setattr(module, 'process_identity', lambda pid: dict(pid=pid, start_ticks=22, command=['other']))
    monkeypatch.setattr(module, 'open_pidfd', lambda pid: pytest.fail('Must not open a reused PID'))
    assert not module.signal_owned(dict(pid=1, start_ticks=11, command=['owned']), 15)


def test_deadline_guard_targets_only_its_registered_processes_and_service(tmp_path, monkeypatch):
    from scripts import label_to_trialspace as module
    handoff = Handoff.__new__(Handoff)
    handoff.root = tmp_path
    handoff.fleet_root = tmp_path / 'fleet'
    handoff.config = dict(pause_file=str(tmp_path/'PAUSE'), deadline='2026-10-26T08:55:00-04:00',
        pause_at='2026-10-26T08:50:00-04:00',
        workflow_process=dict(pid=1), label_process=dict(pid=2), unit='owned-training.service',
        fit_status=str(tmp_path/'fit.json'))
    actions = []
    monkeypatch.setattr(handoff, 'cloud_stop', lambda wait: actions.append(('cloud', wait)))
    monkeypatch.setattr(handoff, 'stop_local', lambda grace: actions.append(('local', grace)))
    monkeypatch.setattr(handoff, 'control', lambda *args: actions.append(('systemctl', args)))
    monkeypatch.setattr(handoff, 'service', lambda: dict(ActiveState='inactive'))
    monkeypatch.setattr(module, 'signal_owned', lambda expected, sig: actions.append(('signal', expected['pid'], sig)))
    handoff.watchdog(immediate=True)
    assert (tmp_path/'PAUSE').exists() and (tmp_path/'DEADLINE').exists()
    assert {row[1] for row in actions if row[0] == 'signal'} == {1, 2}
    assert ('systemctl', ('stop', '--no-block', 'owned-training.service')) in actions
    assert (tmp_path/'deadline_enforced.json').exists()


WORKER = '''
import json, os, sys
from pathlib import Path
import torch
from colbert_experiments import training
import trialspace_training as production
root, mode = Path(sys.argv[1]), sys.argv[2]
class Student(torch.nn.Module):
    def __init__(self, *a, **kw):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.9))
        self.calls = 0
    @property
    def device(self): return self.weight.device
    def fingerprint(self): return dict(toy='trainable')
    def forward(self, query, documents):
        self.calls += 1
        if mode == 'pause' and os.environ['RANK'] == '0' and self.calls == 3:
            (root / 'PAUSE').touch()
        return self.weight * torch.tensor([len(d) / 1000 for d in documents])
    def save(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), Path(path) / 'model.pt')
training.ColBERT = Student
directory = root / ('reference' if mode == 'reference' else 'resumed')
args = production.parser().parse_args(['train', '--device', 'cpu', '--corpus', str(root/'corpus'),
    '--labels', str(root/'scores.jsonl'), '--output', str(directory/'fit'),
    '--output-model', str(directory/'model'), '--steps', '4', '--accumulation', '2',
    '--eval-every', '0', '--save-steps', '1', '--pause-file', str(root/'PAUSE'), '--loss', sys.argv[3]])
result = training.train(args)
(directory / ('rank-' + os.environ['RANK'] + '.json')).write_text(json.dumps(result))
'''


@pytest.mark.parametrize('loss, paused_step', [('mse', 1), ('legacy_ranking', 0)])
def test_four_rank_mid_accumulation_pause_resumes_identically(tmp_path, loss, paused_step):
    create_fixture(tmp_path / 'corpus')
    grouped = {}
    for row in read_jsonl(tmp_path / 'corpus/reference_labels.jsonl'):
        row['key'] = label_key(row)
        grouped.setdefault((row['patient_id'], row['cohort_id']), []).append(row)
    write_jsonl(tmp_path / 'scores.jsonl', [aggregate_pair(rows[0], rows) for rows in grouped.values()])
    worker = tmp_path / 'worker.py'
    worker.write_text(WORKER)
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1',
           'PYTHONPATH': str(Path(__file__).resolve().parents[1]) + os.pathsep + os.environ.get('PYTHONPATH', '')}
    def run(mode):
        result = subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone',
            '--nproc_per_node=4', str(worker), str(tmp_path), mode, loss], env=env,
            text=True, capture_output=True, timeout=180)
        assert result.returncode == 0, result.stdout + result.stderr
    run('pause')
    status = json.loads((tmp_path / 'resumed/fit/status.json').read_text())
    assert status['state'] == 'paused' and status['step'] == paused_step
    for rank in range(4):
        assert json.loads((tmp_path / f'resumed/rank-{rank}.json').read_text()) == 'paused'
    assert not (tmp_path / 'resumed/model').exists()
    (tmp_path / 'PAUSE').unlink()
    run('resume')
    run('reference')
    resumed = torch.load(tmp_path / 'resumed/model/model.pt', weights_only=True)
    reference = torch.load(tmp_path / 'reference/model/model.pt', weights_only=True)
    assert torch.equal(resumed['weight'], reference['weight'])
    history = json.loads((tmp_path / 'resumed/fit/metrics.json').read_text())['history']
    assert [row['step'] for row in history] == [1, 2, 3, 4]
    assert len(list((tmp_path / 'resumed/fit').glob('checkpoint*.pt'))) == 2
