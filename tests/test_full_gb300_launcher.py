import json
from types import SimpleNamespace

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
    assert train.query_groups_per_forward == 8 and train.encoder_token_budget == 131072
    assert train.token_cache_tokens == 8000000
    assert train.ranking_backward == 'replay' and train.activation_storage == 'auto'
    assert train.documents_per_query * train.accumulation == 1024
    assert train.save_total_limit == 2 and train.stop_at is None
    assert train.labels == [str(tmp_path / 'source/labels/scores.jsonl')]
    assert train.output_model == tmp_path / 'out/model'


def test_unpublished_scores_cannot_start_training_even_if_old_status_says_complete(tmp_path):
    assert launcher.ready(tmp_path)[0] is False
    write_json(tmp_path / 'mining_all/complete.json', dict(pairs=17))
    write_json(tmp_path / 'status.json', dict(phase='completed'))
    write_json(tmp_path / 'labels/status.json', dict(state='complete', total_pairs=17, scored_pairs=17))
    done, status = launcher.ready(tmp_path)
    assert not done and status['total_pairs'] == 17


def test_preparation_uses_promoted_cohorts_existing_patients_and_one_fresh_corpus():
    from trialspace_training import parser
    args = launcher.parser().parse_args([])
    commands = launcher.preparation_commands(args)
    parsed = {name: parser().parse_args(command[command.index(name):]) for name, command in commands}
    assert parsed['prepare'].cohorts == str(launcher.COHORTS)
    assert parsed['prepare'].patients == str(launcher.PATIENTS)
    assert parsed['prepare'].merge_source_splits
    assert parsed['prepare'].output == parsed['mine'].corpus == parsed['label'].corpus
    assert parsed['mine'].top_k_patient_to_cohort == 20
    assert parsed['mine'].top_k_cohort_to_patient == 40
    assert parsed['mine'].splits == ['train', 'val', 'test']
    assert parsed['label'].server_urls == 'http://127.0.0.1:8000/v1'
    assert parsed['label'].max_concurrent_per_server == 320 and parsed['label'].gated_regression
    assert parsed['label'].candidates == str(launcher.SOURCE / 'mining_all/candidates.jsonl')
    assert 'post_notes_colbert_20261025' not in str(launcher.SOURCE)


@pytest.mark.parametrize('stop_after,expected', [('prepare', ['prepare']), ('mine', ['prepare', 'mine']),
                                             ('label', ['prepare', 'mine', 'label']),
                                             ('train', ['prepare', 'mine', 'label', 'train'])])
def test_stages_stop_at_requested_boundary_and_train_only_after_publication(tmp_path, monkeypatch, stop_after, expected):
    args = launcher.parser().parse_args(['--source-run', str(tmp_path / 'source'),
                                        '--output', str(tmp_path / 'out'), '--stop-after', stop_after])
    calls = []
    def run(command, **kwargs):
        calls.append(next(name for name in ['prepare', 'mine', 'label', 'train'] if name in command))
        return 0
    monkeypatch.setattr(launcher, 'run_process', run)
    monkeypatch.setattr(launcher, 'ready', lambda path: ('label' in calls, {'scored_pairs': 17}))
    assert launcher.run_stages(args, {}) == 0
    assert calls == expected
    assert (tmp_path / 'out').exists() == (stop_after == 'train')


def test_upstream_failure_never_launches_training(tmp_path, monkeypatch):
    args = launcher.parser().parse_args(['--source-run', str(tmp_path / 'source'), '--output', str(tmp_path / 'out')])
    calls = []
    def fail(command, **kwargs):
        calls.append(command)
        return 1
    monkeypatch.setattr(launcher, 'run_process', fail)
    with pytest.raises(SystemExit) as error:
        launcher.run_stages(args, {})
    assert error.value.code == 1 and len(calls) == 1 and 'prepare' in calls[0]
    assert not args.output.exists()


def test_complete_scores_skip_teacher_calls_and_preserve_pause_exit(tmp_path, monkeypatch):
    args = launcher.parser().parse_args(['--source-run', str(tmp_path / 'source'), '--output', str(tmp_path / 'out')])
    calls = []
    def run(command, **kwargs):
        calls.append(next(name for name in ['prepare', 'mine', 'label', 'train'] if name in command))
        return 75 if calls[-1] == 'train' else 0
    monkeypatch.setattr(launcher, 'run_process', run)
    monkeypatch.setattr(launcher, 'ready', lambda path: (True, {'scored_pairs': 17}))
    assert launcher.run_stages(args, {}) == 75
    assert calls == ['prepare', 'mine', 'train']
    assert json.loads((args.source_run / 'gb300_status.json').read_text())['phase'] == 'paused'


def test_changed_teacher_cannot_reuse_existing_run(tmp_path, monkeypatch):
    args = launcher.parser().parse_args(['--source-run', str(tmp_path), '--teacher-model', 'new-model'])
    write_json(tmp_path / 'gb300_label_settings.json', launcher.label_settings(launcher.parser().parse_args([])))
    monkeypatch.setattr(launcher.subprocess, 'run', lambda *a, **k: pytest.fail('must not start any stages'))
    with pytest.raises(ValueError, match='Teacher settings changed'):
        launcher.run_stages(args, {})


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
    args = launcher.parser().parse_args(['--ranking-backward', 'replay', '--gradient-checkpointing'])
    assert '--gradient-checkpointing' in launcher.command(args)
    assert launcher.OUTPUT.name == 'colbert_production_gb300_namedcohorts_20261026'


def test_retained_launcher_rejects_activation_recomputation(monkeypatch):
    monkeypatch.setattr(launcher.sys, 'argv', ['launcher', '--dry-run', '--ranking-backward', 'retained', '--gradient-checkpointing'])
    with pytest.raises(ValueError, match='one forward'):
        launcher.main()


def test_direct_activation_storage_reaches_production_parser():
    from trialspace_training import parser
    args = launcher.parser().parse_args(['--ranking-backward', 'retained', '--activation-storage', 'direct',
                                        '--activation-gpu-budget-gb', '96'])
    command = launcher.command(args)
    train = parser().parse_args(command[command.index('train'):])
    assert train.activation_storage == 'direct' and train.activation_gpu_budget_gb == 96
    assert train.accumulation == 64 and train.documents_per_query == 16


def test_live_pool_and_bounded_mining_flags_reach_production_parser(tmp_path):
    from trialspace_training import parser
    args = launcher.parser().parse_args(['--teacher-servers-file', str(tmp_path / 'servers.json'),
        '--teacher-concurrency-file', str(tmp_path / 'concurrency.json'), '--label-concurrency', '1024',
        '--mining-batch-size', '2', '--mining-pair-batch-size', '16', '--mining-device-cache-gb', '0'])
    parsed = {name: parser().parse_args(command[command.index(name):])
              for name, command in launcher.preparation_commands(args)}
    assert parsed['label'].server_urls is None
    assert parsed['label'].server_urls_file == str(tmp_path / 'servers.json')
    assert parsed['label'].concurrency_control_file == str(tmp_path / 'concurrency.json')
    assert parsed['label'].max_concurrent_per_server == 1024
    assert parsed['mine'].query_batch_size == parsed['mine'].document_batch_size == 2
    assert parsed['mine'].pair_batch_size == 16 and parsed['mine'].device_cache_gb == 0


def test_child_status_is_compatible_with_cloud_inference_completion(tmp_path):
    import sys
    from scripts.gcp_note_fleet import workflow_terminal_reason
    status = tmp_path / 'status.json'
    assert launcher.run_process([sys.executable, '-c', 'pass'], environment=None,
                                status_path=status, phase='labeling') == 0
    workflow = json.loads(status.read_text())
    labels = dict(pid=workflow['child_pid'], state='exporting_labels', inference_complete=True,
                  total_pairs=10, scored_pairs=10, failures={})
    assert workflow_terminal_reason({}, False, workflow, True, mode='labeling',
                                    label_status=labels) == 'label_inference_finished'
