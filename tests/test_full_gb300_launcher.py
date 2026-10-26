import pytest

from colbert_experiments import train_full_gb300 as launcher
from colbert_experiments.data import write_json


def test_full_launcher_uses_production_parser_and_complete_data(tmp_path):
    from trialspace_training import parser
    args = launcher.parser().parse_args(['--source-run', str(tmp_path / 'source'), '--output', str(tmp_path / 'out')])
    command = launcher.command(args)
    train = parser().parse_args(command[command.index('train'):])
    assert train.training_splits == ['train', 'val', 'test'] and train.require_complete_coverage
    assert train.loss == 'legacy_ranking' and train.steps is None and train.epochs == 1
    assert train.gradient_checkpointing is False
    assert train.query_groups_per_forward == 4 and train.encoder_token_budget == 65536
    assert train.token_cache_tokens == 8000000
    assert train.documents_per_query * train.accumulation == 1024
    assert train.save_total_limit == 2 and train.stop_at is None
    assert train.labels == [str(tmp_path / 'source/labels/scores.jsonl')]
    assert train.output_model == tmp_path / 'out/model'


def test_waits_for_full_export_and_rejects_incomplete_or_wrong_coverage(tmp_path):
    labels = dict(state='exporting_labels', total_pairs=launcher.EXPECTED_PAIRS,
                  scored_pairs=launcher.EXPECTED_PAIRS, failures={})
    write_json(tmp_path / 'status.json', dict(phase='labeling'))
    write_json(tmp_path / 'labels/status.json', labels)
    assert launcher.ready(tmp_path)[0] is False
    write_json(tmp_path / 'status.json', dict(phase='completed'))
    write_json(tmp_path / 'labels/status.json', {**labels, 'state': 'complete'})
    with pytest.raises(FileNotFoundError):
        launcher.ready(tmp_path)
    for name in ['completed/labeling.json', 'labels/scores.jsonl', 'labels/labels.jsonl',
                 'corpus_all/patients.jsonl', 'corpus_all/cohorts.jsonl']:
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.touch()
    assert launcher.ready(tmp_path)[0] is True
    write_json(tmp_path / 'labels/status.json', {**labels, 'state': 'complete', 'scored_pairs': 1})
    with pytest.raises(ValueError, match='incomplete score coverage'):
        launcher.ready(tmp_path)
    write_json(tmp_path / 'labels/status.json', {**labels, 'failures': {'key': 'failed'}})
    with pytest.raises(RuntimeError, match='labeling failed'):
        launcher.ready(tmp_path)


def test_dry_run_never_waits_launches_or_writes(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    monkeypatch.setattr(launcher.sys, 'argv', ['launcher', '--dry-run', '--output', str(tmp_path / 'out')])
    monkeypatch.setattr(launcher, 'ready', lambda path: pytest.fail('must not read source on dry run'))
    monkeypatch.setattr(launcher.subprocess, 'run', lambda *a, **k: pytest.fail('must not launch on dry run'))
    launcher.main()
    assert '1,024 directional pairs per update' in capsys.readouterr().out
    assert not (tmp_path / 'out').exists()


def test_camus_cannot_start_training(tmp_path, monkeypatch):
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    monkeypatch.setattr(launcher.sys, 'argv', ['launcher', '--output', str(tmp_path / 'out')])
    monkeypatch.setattr(launcher.socket, 'gethostname', lambda: 'camus.dfci.harvard.edu')
    monkeypatch.setattr(launcher.subprocess, 'run', lambda *a, **k: pytest.fail('must not train on Camus'))
    with pytest.raises(RuntimeError, match='moved off Camus'):
        launcher.main()


@pytest.mark.parametrize('gpu,visible,expected', [
    ('0', None, '0'), ('1', None, '1'), ('1', '1', '1'), ('1', '3,1', '1'),
    ('0', '1', '0'), ('1', '', '1'), ('GPU-first', 'GPU-second', 'GPU-first')])
def test_explicit_gpu_is_always_a_physical_selection(gpu, visible, expected):
    assert launcher.selected_gpu(gpu, visible) == expected


@pytest.mark.parametrize('gpu', ['-1', '', '0,1'])
def test_invalid_physical_gpu_is_rejected(gpu):
    with pytest.raises(ValueError):
        launcher.selected_gpu(gpu, None)


@pytest.mark.parametrize('visible,index,expected', [
    (None, None, '0'), ('1', None, '1'), ('3,1', 1, '1'), ('GPU-first', 0, 'GPU-first')])
def test_omitted_physical_selection_uses_the_visibility_mask(visible, index, expected):
    assert launcher.selected_gpu(None, visible, visible_index=index) == expected


@pytest.mark.parametrize('visible,index', [('', 0), ('-1', 0), ('1', 1), ('1', -1)])
def test_invalid_visible_selection_is_rejected(visible, index):
    with pytest.raises(ValueError):
        launcher.selected_gpu(None, visible, visible_index=index)


@pytest.mark.parametrize('visible', [None, '1', '3,1'])
def test_explicit_gpu_one_dry_run_is_unaffected_by_inherited_mask(monkeypatch, capsys, visible):
    if visible is None:
        monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    else:
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', visible)
    monkeypatch.setattr(launcher.sys, 'argv', ['launcher', '--gpu', '1', '--dry-run'])
    launcher.main()
    output = capsys.readouterr().out
    assert output.startswith('CUDA_VISIBLE_DEVICES=1 ')
    assert 'physical CUDA device 1 as cuda:0 (explicit --gpu)' in output


def test_physical_and_visible_selection_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        launcher.parser().parse_args(['--gpu', '1', '--visible-gpu', '0'])


def test_memory_saving_checkpointing_can_be_enabled_for_the_new_run():
    args = launcher.parser().parse_args(['--gradient-checkpointing'])
    assert '--gradient-checkpointing' in launcher.command(args)
    assert launcher.OUTPUT.name == 'colbert_production_gb300_batched_20261026'
