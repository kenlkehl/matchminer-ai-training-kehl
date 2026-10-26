import pytest

from colbert_experiments import train_full_gb300 as launcher
from colbert_experiments.data import write_json


def test_full_launcher_uses_production_parser_and_complete_data(tmp_path):
    from trialspace_training import parser
    args = launcher.parser().parse_args(['--source-run', str(tmp_path / 'source'), '--output', str(tmp_path / 'out')])
    command = launcher.command(args)
    train = parser().parse_args(command[command.index('train'):])
    assert train.training_splits == ['train', 'val', 'test'] and train.require_complete_coverage
    assert train.loss == 'legacy_ranking' and train.steps is None and train.epochs == 3
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
    monkeypatch.setattr(launcher.sys, 'argv', ['launcher', '--dry-run', '--output', str(tmp_path / 'out')])
    monkeypatch.setattr(launcher, 'ready', lambda path: pytest.fail('must not read source on dry run'))
    monkeypatch.setattr(launcher.subprocess, 'run', lambda *a, **k: pytest.fail('must not launch on dry run'))
    launcher.main()
    assert '1,024 directional pairs per update' in capsys.readouterr().out
    assert not (tmp_path / 'out').exists()


def test_camus_cannot_start_training(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher.sys, 'argv', ['launcher', '--output', str(tmp_path / 'out')])
    monkeypatch.setattr(launcher.socket, 'gethostname', lambda: 'camus.dfci.harvard.edu')
    monkeypatch.setattr(launcher.subprocess, 'run', lambda *a, **k: pytest.fail('must not train on Camus'))
    with pytest.raises(RuntimeError, match='moved off Camus'):
        launcher.main()
