import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import torch

from colbert_experiments import smoke_train_completed as smoke
from colbert_experiments.contracts import TASKS, contract_hash, judge_messages, prompt_sources
from colbert_experiments.data import digest, read_jsonl, write_json, write_jsonl
from colbert_experiments import regression
from colbert_experiments import export_completed_scores as packed
from colbert_experiments.complete_pass_batches import CompletePassBatchSampler


def live_source(root):
    patients = [dict(id=s, text=f'Complete synthetic {s} history. Other relevant history: none.',
                     split='train', source_split=s) for s in ('train', 'val', 'test')]
    cohorts = [dict(id=str(i), text=f'Complete synthetic cohort {i} criteria.', split='train',
                    source_split='val' if i == 0 else 'train', trial_id=f'T{i}') for i in range(3)]
    write_jsonl(root / 'corpus_all/patients.jsonl', patients)
    write_jsonl(root / 'corpus_all/cohorts.jsonl', cohorts)
    pairs = [dict(patient_id=p['id'], cohort_id=c['id'], split='train') for p in patients for c in cohorts]
    write_jsonl(root / 'mining_all/candidates.jsonl', pairs)
    write_json(root / 'labels/labeling_manifest.json', dict(contract=contract_hash(),
        aggregation_contract=regression.score_contract(), patients=digest(patients), cohorts=digest(cohorts)))
    pmap, cmap = {p['id']: p for p in patients}, {c['id']: c for c in cohorts}
    for pair in pairs:
        # Cohort 0: complete zero. Cohort 1: complete 5. Cohort 2: incomplete yes.
        tasks = list(TASKS) if pair['cohort_id'] == '1' else [regression.GATE]
        for task in tasks:
            key = digest([pair['patient_id'], pair['cohort_id'], task])
            row = dict(**pair, key=key, task=task, answer='no' if pair['cohort_id'] == '0' else 'yes',
                evidence_status='supported', reason='Synthetic test.', finish_reason='stop', contract=contract_hash(),
                prompt_sha256=digest(judge_messages(task, pmap[pair['patient_id']]['text'], cmap[pair['cohort_id']]['text'])))
            write_json(root / 'labels/responses' / key[:2] / (key+'.json'), row)
    return patients, cohorts, pairs


def arguments(source, output):
    return smoke.parser().parse_args(['--source-run', str(source), '--output', str(output),
        '--inspect-checkpoints',
        '--train-pairs', '2', '--val-pairs', '2', '--candidate-multiplier', '5', '--device', 'cpu',
        '--steps', '3', '--accumulation', '2', '--eval-pairs', '2', '--eval-every', '1', '--save-steps', '1'])


def test_snapshot_recovers_source_splits_and_omits_partial_and_test_pairs(tmp_path):
    source, output = tmp_path/'source', tmp_path/'out'
    patients, cohorts, pairs = live_source(source)
    args = arguments(source, output)
    manifest = smoke.prepare_snapshot(args)
    scores = read_jsonl(output/'snapshot/scores.jsonl')
    assert len(scores) == 4
    assert {r['patient_id'] for r in scores} == {'train', 'val'}
    assert {r['cohort_id'] for r in scores} == {'0', '1'}
    assert {r['score'] for r in scores} == {0, 5}
    assert all(r['split'] == 'train' for r in scores)  # Production pooled split.
    parts = smoke.split_scores(patients, scores)
    assert all(r['patient_id'] == 'train' for r in parts['train'])
    assert all(r['patient_id'] == 'val' for r in parts['val'])
    groups, _ = regression.build_groups(patients, cohorts, parts['train'])
    assert all(g['query_id'] == 'train' if g['direction'] == 'patient_to_cohort'
               else all(d['id'] == 'train' for d in g['documents']) for g in groups)
    assert manifest['counts']['val']['pairs'] == 2
    # Continued production writes do not modify an existing experiment snapshot.
    assert smoke.prepare_snapshot(args) == manifest
    assert smoke.fixed_eval_pairs(parts['val'], 2, 42) == smoke.fixed_eval_pairs(parts['val'], 2, 42)


def test_default_snapshot_includes_all_completed_train_pairs(tmp_path):
    source, output = tmp_path/'source', tmp_path/'out'
    patients, cohorts, pairs = live_source(source)
    # Complete a third training pair, leaving the corresponding val/test pairs
    # incomplete. A multiplier of one must never become an implicit train cap.
    pair = dict(patient_id='train', cohort_id='2', split='train')
    for task in regression.SPECIFICITY:
        key = digest([pair['patient_id'], pair['cohort_id'], task])
        write_json(source/'labels/responses'/key[:2]/(key+'.json'), dict(**pair,
            key=key, task=task, answer='yes', evidence_status='supported', reason='Synthetic test.',
            finish_reason='stop', contract=contract_hash(),
            prompt_sha256=digest(judge_messages(task, patients[0]['text'], cohorts[2]['text']))))
    args = smoke.parser().parse_args(['--source-run', str(source), '--output', str(output),
        '--val-pairs', '3', '--candidate-multiplier', '1', '--inspect-checkpoints'])
    assert args.train_pairs == 0
    manifest = smoke.prepare_snapshot(args)
    scores = read_jsonl(output/'snapshot/scores.jsonl')
    parts = smoke.split_scores(patients, scores)
    assert {r['cohort_id'] for r in parts['train']} == {'0', '1', '2'}
    assert {r['cohort_id'] for r in parts['val']} == {'0', '1'}
    assert manifest['counts']['train']['requested'] == 'all'
    assert manifest['counts']['train']['inspected'] == manifest['available_candidates']['train'] == 3
    assert manifest['counts']['train']['selection'] == 'all_completed_candidate_pairs'
    assert smoke.prepare_snapshot(args) == manifest


def test_uncapped_run_rejects_existing_capped_snapshot(tmp_path):
    source, output = tmp_path/'source', tmp_path/'out'
    live_source(source)
    args = arguments(source, output)
    smoke.prepare_snapshot(args)
    args.train_pairs = 0
    with pytest.raises(ValueError, match='Snapshot settings changed'):
        smoke.prepare_snapshot(args)


def test_response_index_supports_sharded_and_legacy_checkpoints(tmp_path):
    (tmp_path/'ab').mkdir()
    (tmp_path/'ab'/'sharded.json').write_text('{}')
    (tmp_path/'ab'/'writing.json.tmp').write_text('{}')
    (tmp_path/'legacy.json').write_text('{}')
    assert smoke.response_keys(tmp_path) == {'sharded', 'legacy'}


def test_missing_compact_export_does_not_silently_start_nfs_inspection(tmp_path):
    source = tmp_path/'source'
    live_source(source)
    args = smoke.parser().parse_args(['--source-run', str(source), '--output', str(tmp_path/'out')])
    with pytest.raises(ValueError, match='No compact completed-score archive'):
        smoke.prepare_snapshot(args)


def test_compact_export_omits_incomplete_pairs_and_bypasses_question_inspection(tmp_path, monkeypatch):
    source, output = tmp_path/'source', tmp_path/'out'
    patients, cohorts, _ = live_source(source)
    archive = source/'labels/completed_scores'
    exported = packed.export(source, archive, workers=1)
    assert exported['counts']['completed_pairs'] == 6
    assert exported['counts']['incomplete_pairs'] == 3
    rows = read_jsonl(archive/exported['scores_file'])
    for row in rows:
        expected = smoke.completed_score(row, responses=source/'labels/responses',
            patients={p['id']: p for p in patients}, cohorts={c['id']: c for c in cohorts},
            cutoff_ns=time.time_ns(), question_contract=contract_hash(), sources=prompt_sources())
        assert all(expected[k] == v for k, v in row.items())
    # The consumer needs no individual responses or candidate scan at all.
    (source/'labels/responses').rename(source/'labels/responses-unavailable')
    for name in ('response_keys', 'candidate_sample', 'all_candidates', 'completed_score'):
        monkeypatch.setattr(smoke, name, lambda *a, **kw: pytest.fail('Unexpected question inspection'))
    args = smoke.parser().parse_args(['--source-run', str(source), '--output', str(output)])
    manifest = smoke.prepare_snapshot(args)
    assert manifest['checkpoint_files_inspected'] == 0
    assert manifest['counts']['train']['requested'] == 'all'
    assert manifest['counts']['train']['pairs'] == manifest['counts']['val']['pairs'] == 2
    assert {r['patient_id'] for r in read_jsonl(output/'snapshot/scores.jsonl')} == {'train', 'val'}
    path = archive/exported['scores_file']
    raw = path.read_bytes()
    path.write_bytes(raw.replace(b'{', b'{ ', 1))  # Same JSON values, different checksum.
    args.output = tmp_path/'corrupt-archive'
    with pytest.raises(ValueError, match='checksum'):
        smoke.prepare_snapshot(args)
    path.write_bytes(raw)
    # Prefer the final production export over an older partial compact snapshot.
    (source/'labels/scores.jsonl').write_bytes(raw)
    args.output = tmp_path/'final-scores'
    final = smoke.prepare_snapshot(args)
    assert final['checkpoint_files_inspected'] == 0
    assert final['source_score_file'] == str(source/'labels/scores.jsonl')


def test_compact_score_validation_rejects_partial_and_wrong_contract():
    patients = {'p': dict(id='p', split='train')}
    cohorts = {'c': dict(id='c', split='train')}
    q, a = contract_hash(), regression.score_contract()
    row = packed.score_from_answers('p', 'c', 'train', ['no', None, None, None, None], q, a)
    packed.validate_score(row, patients, cohorts, q, a)
    with pytest.raises(ValueError, match='incomplete'):
        packed.validate_score({**row, 'contract': 'wrong'}, patients, cohorts, q, a)
    with pytest.raises(ValueError, match='incomplete'):
        packed.validate_score({**row, 'answers': {**row['answers'], regression.GATE: 'yes'}}, patients, cohorts, q, a)


def test_checkpoint_validation_and_cutoff(tmp_path):
    patients, cohorts, pairs = live_source(tmp_path)
    pair = pairs[0]
    key = digest([pair['patient_id'], pair['cohort_id'], regression.GATE])
    path = tmp_path/'labels/responses'/key[:2]/(key+'.json')
    options = dict(responses=tmp_path/'labels/responses', patients={p['id']: p for p in patients},
        cohorts={c['id']: c for c in cohorts}, question_contract=contract_hash(), sources=prompt_sources())
    assert smoke.completed_score(pair, cutoff_ns=0, **options) is None
    assert smoke.completed_score(pair, cutoff_ns=time.time_ns(), **options)['score'] == 0
    row = json.loads(path.read_text())
    row['prompt_sha256'] = 'different-text'
    write_json(path, row)
    with pytest.raises(ValueError, match='mismatch'):
        smoke.completed_score(pair, cutoff_ns=time.time_ns(), **options)


def test_auc_uses_continuous_maxsim_and_nonzero_as_positive():
    rows = [dict(score=0, maxsim=-.3), dict(score=1, maxsim=-.2),
            dict(score=0, maxsim=.1), dict(score=5, maxsim=.4)]
    result = smoke.prediction_metrics(rows)
    assert result['auroc'] == pytest.approx(.75)  # Thresholding raw scores would change this.
    assert result['loss'] == pytest.approx((.09+.16+.01+.36)/4)
    assert result['positives'] == result['negatives'] == 2
    assert smoke.prediction_metrics([dict(score=0, maxsim=.5)])['auroc'] is None
    assert smoke.prediction_metrics([dict(score=0, maxsim=.5), dict(score=2, maxsim=.5)])['auroc'] == .5


def test_complete_epochs_visit_every_pair_once_and_resume_order_is_reproducible():
    groups = [dict(direction=direction, query_id=f'q{q}',
                   documents=[dict(id=f'd{i}') for i in range(count)],
                   labels=[i % 6 for i in range(count)])
              for direction in smoke.DIRECTIONS for q, count in enumerate((1, 3, 7, 19))]
    expected = {(g['direction'], g['query_id'], i) for g in groups for i in range(len(g['labels']))}
    sampler = CompletePassBatchSampler(groups, documents_per_query=2, query_groups_per_update=4, seed=42)
    assert sampler.microbatches == 34 and sampler.updates_per_epoch == 9
    schedules = []
    for epoch in range(3):
        visited = []
        for update in range(sampler.updates_per_epoch):
            batch = sampler.batch(epoch * sampler.updates_per_epoch + update)
            assert 1 <= len(batch) <= 4
            visited.extend((g['direction'], g['query_id'], i) for g, indices in batch for i in indices)
        assert len(visited) == len(expected) == sampler.pairs
        assert set(visited) == expected
        schedules.append(visited)
    assert schedules[0] != schedules[1]
    resumed = CompletePassBatchSampler(groups, documents_per_query=2, query_groups_per_update=4, seed=42)
    assert resumed.batch(13) == sampler.batch(13)
    defaults = smoke.parser().parse_args(['--output', '/tmp/unused'])
    assert defaults.steps is None and defaults.epochs == 3


WORKER = '''
import sys, torch
from pathlib import Path
from colbert_experiments import smoke_train_completed as smoke
class Toy(torch.nn.Module):
    def __init__(self, *a, **kw):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.9))
    @property
    def device(self): return self.weight.device
    def fingerprint(self): return dict(toy=True)
    def encode_texts(self, texts, is_queries):
        assert len(texts) >= 2 and len(texts) == len(is_queries)
        return [self.weight.expand(1, 1) for text in texts]
    def forward(self, query, documents):
        return self.weight.square().expand(len(documents))
    def save(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), Path(path)/'toy.pt')
smoke.ColBERT = Toy
args = smoke.parser().parse_args(sys.argv[1:])
smoke.train_worker(args)
'''


@pytest.mark.parametrize('accumulation, updates_per_epoch, pair_counts', [(2, 1, [4]), (1, 2, [2, 2])])
def test_two_rank_complete_epochs_report_validation_and_preserve_the_partial_batch(
        tmp_path, accumulation, updates_per_epoch, pair_counts):
    source, output = tmp_path/'source', tmp_path/'out'
    live_source(source)
    args = arguments(source, output)
    smoke.prepare_snapshot(args)
    worker = tmp_path/'worker.py'
    worker.write_text(WORKER)
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2', str(worker),
        '--source-run', str(source), '--output', str(output), '--device', 'cpu',
        '--accumulation', str(accumulation), '--eval-pairs', '2', '--eval-every', '1', '--save-steps', '1']
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1',
           'PYTHONPATH': str(smoke.ROOT) + os.pathsep + os.environ.get('PYTHONPATH', '')}
    result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    total_updates = 3 * updates_per_epoch
    assert result.stdout.count('[eval ') == total_updates + 1
    assert f'{total_updates} total updates (3 complete epochs)' in result.stdout
    assert 'train_loss=' in result.stdout and 'val_loss=' in result.stdout
    assert 'val_auroc_patient_to_cohort=0.5000' in result.stdout
    assert 'val_auroc_cohort_to_patient=0.5000' in result.stdout
    metrics = json.loads((output/'fit/metrics.json').read_text())
    assert len(metrics['history']) == total_updates
    assert metrics['history'][0]['train_batch_loss'] == pytest.approx(.3461, abs=1e-6)
    assert metrics['history'][0]['gradient_norm'] == pytest.approx(1.116, abs=1e-5)
    # An odd number of query chunks requires a zero-weight dummy microbatch,
    # possibly for an entire rank's update; all four directional pairs count once.
    assert [entry['directional_pairs'] for entry in metrics['history']] == pair_counts * 3
    assert [entry['epoch'] for entry in metrics['history']] == [e for e in (1, 2, 3) for _ in range(updates_per_epoch)]
    assert metrics['evaluations'][-1]['train']['loss'] < metrics['evaluations'][0]['train']['loss']
    assert len(list((output/'fit').glob('checkpoint-*.pt'))) == 2
    status = json.loads((output/'fit/status.json').read_text())
    assert status['state'] == 'complete' and status['completed_epochs'] == 3
    assert status['directional_pairs_seen'] == 12


def test_ranking_smoke_reports_objective_preserves_auroc_and_resumes_scheduler(tmp_path):
    source, output = tmp_path/'source', tmp_path/'out'
    live_source(source)
    smoke.prepare_snapshot(arguments(source, output))
    worker = tmp_path/'worker.py'
    worker.write_text(WORKER)
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2', str(worker),
        '--source-run', str(source), '--output', str(output), '--device', 'cpu', '--accumulation', '2',
        '--eval-pairs', '2', '--eval-every', '1', '--save-steps', '1', '--loss', 'legacy_ranking',
        '--learning-rate', '2e-5', '--lr-scheduler', 'linear', '--warmup-ratio', '.01']
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1',
           'PYTHONPATH': str(smoke.ROOT) + os.pathsep + os.environ.get('PYTHONPATH', '')}
    result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'objective=legacy_ranking' in result.stdout and 'val_mse_diagnostic=' in result.stdout
    assert 'val_auroc_patient_to_cohort=0.5000' in result.stdout
    metrics = json.loads((output/'fit/metrics.json').read_text())
    assert metrics['history'][0]['train_batch_loss'] == pytest.approx(2 * math.log(2))
    assert [r['learning_rate'] for r in metrics['history']] == pytest.approx([0, 2e-5, 1e-5])
    assert all(r['ordinal_comparisons'] == 2 and r['contrastive_queries'] == 1 for r in metrics['history'])
    assert len(list((output/'fit').glob('checkpoint-*.pt'))) == 2
    state = torch.load(output/'fit'/json.loads((output/'fit/latest.json').read_text())['path'], weights_only=True)
    assert state['scheduler']['last_epoch'] == 3
    assert state['optimizer']['param_groups'][0]['lr'] == 0
    scoring = json.loads((output/'model/regression_scoring.json').read_text())
    assert scoring['calibrated'] is False and 'output_scale' not in scoring
    # Simulate interruption immediately after step 2 and resume its saved LR
    # schedule, model and optimizer. The final state must equal the uninterrupted one.
    checkpoints = sorted((output/'fit').glob('checkpoint-*.pt'))
    write_json(output/'fit/latest.json', dict(path=checkpoints[0].name))
    resumed = subprocess.run(command, env=env, text=True, capture_output=True, timeout=180)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    resumed_state = torch.load(output/'fit'/json.loads((output/'fit/latest.json').read_text())['path'], weights_only=True)
    assert resumed_state['step'] == state['step'] == 3
    assert resumed_state['scheduler'] == state['scheduler']
    torch.testing.assert_close(resumed_state['model']['weight'], state['model']['weight'])
