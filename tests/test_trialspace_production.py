import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from colbert_experiments.contracts import TASKS
from colbert_experiments.data import load_corpus, read_jsonl, write_jsonl
from colbert_experiments.fixtures import create_fixture
from colbert_experiments.labeling import label_key
from colbert_experiments.model import maxsim
from colbert_experiments.packed_scoring import PackedTokens, score_pairs
from colbert_experiments.regression import aggregate_pair, INSTRUCTION
import train_from_summaries as runner
import trialspace_training as production


def test_production_plan_uses_only_gated_colbert_and_parses_real_commands(tmp_path):
    args = runner.parser().parse_args(['--data-dir', str(tmp_path), '--run-dir', str(tmp_path / 'run'),
        '--models-dir', str(tmp_path / 'models'), '--gpus', '2,3', '--trialspace-steps', '8'])
    plan = runner.stages(args)
    assert args.embedding_model == 'lightonai/GTE-ModernColBERT-v1'
    assert plan[0].name == 'prepare-trialspace'
    assert runner.input_paths(args)['trial_spaces'].name == 'trial_cohort_lineitems.csv'
    for stage in plan:
        if 'trialspace_training.py' not in stage.command:
            continue
        parsed = production.parser().parse_args(stage.command[stage.command.index('trialspace_training.py') + 1:])
        if parsed.command == 'prepare':
            assert parsed.merge_source_splits
        if parsed.command == 'label':
            assert parsed.gated_regression and not parsed.return_records
            assert parsed.scheduler_adaptive_concurrency
            assert parsed.max_concurrent_per_server == 768
            assert parsed.initial_concurrent_per_server == 96
            assert parsed.pending_requests == 0 and parsed.max_auto_pending_requests == 16384
            assert parsed.teacher_tokenizer == args.teacher_model
        elif parsed.command in ('mine', 'train'):
            assert parsed.objective == 'gated_regression'
            assert parsed.query_length == parsed.document_length == 8192
            if parsed.command == 'train':
                assert parsed.loss == 'legacy_ranking' and parsed.learning_rate == 2e-5
                assert parsed.lr_scheduler == 'linear' and parsed.warmup_ratio == .01
                assert parsed.ranking_scale == 20 and parsed.contrastive_weight == 1
                assert parsed.prediction_link == 'identity' and parsed.steps == 8
                assert parsed.output_model.parent == args.models_dir
                assert parsed.accumulation == 32 and parsed.documents_per_query == 8
                assert parsed.query_groups_per_forward == 2 and parsed.encoder_token_budget == 32768
                assert parsed.token_cache_tokens == 8000000
                assert parsed.save_total_limit == 2
                assert parsed.training_splits == ['train', 'val', 'test'] and parsed.require_complete_coverage
            else:
                assert parsed.splits == ['train', 'val', 'test']
                assert parsed.device_token_cache
                assert parsed.device_cache_gb == 8
                assert parsed.pair_batch_size == 512
                assert parsed.score_query_batches == 4
    assert not any(script in str([s.command for s in plan]) for script in ('finetune_embedder.py', 'llm_check_trials.py', 'make_top_matches.py'))
    stages = {s.name: s for s in plan}
    for iteration in (1, 2):
        fit = stages[f'train-trialspace-{iteration}'].command
        label = stages[f'label-trialspace-{iteration}'].outputs[0]
        assert fit[fit.index('--labels') + 1] == str(label)
        previous = args.embedding_model if iteration == 1 else str(args.models_dir / 'trialspace_round1')
        assert fit[fit.index('--model') + 1] == previous
    assert stages['mine-3'].command[stages['mine-3'].command.index('--model') + 1] == str(args.models_dir / 'trialspace_round2')
    downstream_prepare = stages['prepare-oncoreasoning'].command
    downstream_fit = stages['train-oncoreasoning'].command
    assert downstream_prepare[downstream_prepare.index('--splits') + 1:] == ['train', 'validation', 'test']
    index = downstream_fit.index('--training-splits')
    assert downstream_fit[index + 1:index + 4] == ['train', 'validation', 'test']


def test_direct_label_defaults_match_pipeline_and_explicit_limits_pass_through():
    import run_pipeline
    direct = production.parser().parse_args(['label', '--corpus', 'corpus', '--candidates', 'pairs',
        '--output', 'labels', '--teacher-model', 'teacher', '--teacher-tokenizer', 'teacher'])
    assert (direct.max_concurrent_per_server, direct.initial_concurrent_per_server,
            direct.pending_requests, direct.max_auto_pending_requests) == (768, 96, 0, 16384)
    for parser in (runner.parser, run_pipeline.parser):
        args = parser().parse_args(['--label-concurrency-per-server', '64',
            '--label-initial-concurrency-per-server', '48', '--trialspace-label-workers', '8192',
            '--trialspace-label-max-auto-workers', '12000'])
        for stage in runner.stages(args):
            if not stage.name.startswith('label-trialspace'):
                continue
            child = production.parser().parse_args(stage.command[2:])
            assert (child.max_concurrent_per_server, child.initial_concurrent_per_server,
                    child.pending_requests, child.max_auto_pending_requests) == (64, 48, 8192, 12000)


def test_explicit_holdout_experiment_preserves_original_partitions(tmp_path):
    args = runner.parser().parse_args(['--data-dir', str(tmp_path), '--synthetic-training-splits', 'train'])
    stages = {stage.name: stage for stage in runner.stages(args)}
    assert '--no-merge-source-splits' in stages['prepare-trialspace'].command
    for stage in stages.values():
        if 'trialspace_training.py' not in stage.command:
            continue
        parsed = production.parser().parse_args(stage.command[stage.command.index('trialspace_training.py') + 1:])
        if parsed.command == 'train':
            assert parsed.training_splits == ['train']
        elif parsed.command == 'mine':
            assert parsed.splits == ['train', 'val']


def test_pipeline_passes_training_pause_and_checkpoint_controls():
    import run_pipeline
    for parser in (runner.parser, run_pipeline.parser):
        args = parser().parse_args(['--trialspace-save-steps', '25', '--trialspace-save-total-limit', '2',
            '--trialspace-stop-at', '2026-10-26T08:50:00-04:00', '--trialspace-pause-file', '/tmp/PAUSE'])
        for stage in runner.stages(args):
            if stage.name.startswith('train-trialspace-'):
                tail = stage.command[stage.command.index('trialspace_training.py') + 1:]
                child = production.parser().parse_args(tail)
                assert child.save_steps == 25 and child.save_total_limit == 2
                assert child.stop_at == '2026-10-26T08:50:00-04:00'
                assert child.pause_file == Path('/tmp/PAUSE')


def test_pipeline_passes_grouped_encoder_and_token_cache_controls():
    import run_pipeline
    for parser in (runner.parser, run_pipeline.parser):
        args = parser().parse_args(['--trialspace-query-groups-per-forward', '4',
            '--trialspace-encoder-token-budget', '65536', '--trialspace-token-cache-tokens', '123456',
            '--trialspace-ranking-backward', 'retained', '--trialspace-activation-storage', 'gpu'])
        for stage in runner.stages(args):
            if stage.name.startswith('train-trialspace-'):
                tail = stage.command[stage.command.index('trialspace_training.py') + 1:]
                child = production.parser().parse_args(tail)
                assert child.query_groups_per_forward == 4 and child.encoder_token_budget == 65536
                assert child.token_cache_tokens == 123456
                assert child.ranking_backward == 'retained' and child.activation_storage == 'gpu'
                assert child.gradient_checkpointing is False


def test_deadline_and_checkpoint_changes_do_not_invalidate_pipeline_resume():
    import run_pipeline
    original = run_pipeline.parser().parse_args([])
    paused = run_pipeline.parser().parse_args(['--trialspace-save-steps', '25',
        '--trialspace-stop-at', '2026-10-26T08:50:00-04:00', '--trialspace-pause-file', '/tmp/PAUSE'])
    assert runner.training_identity_arguments(original) == runner.training_identity_arguments(paused)
    assert runner.training_identity_command(run_pipeline.training_command(original)) == runner.training_identity_command(
        run_pipeline.training_command(paused))
    changed_loss = run_pipeline.parser().parse_args(['--trialspace-learning-rate', '0.01'])
    assert runner.training_identity_arguments(original) != runner.training_identity_arguments(changed_loss)


def test_prepare_reconstructs_full_criteria_history_and_lineage(tmp_path):
    patients = tmp_path / 'patients.parquet'
    pd.DataFrame([dict(pseudo_mrn='p', patient_summary='Cancer history.', patient_boilerplate_text='Renal history.',
                      nct_id='NCT00000001', split='train')]).to_parquet(patients)
    rows = []
    for number in (0, 1):
        rows.append(dict(trial_id='NCT00000001', cohort_trial_id=f'NCT00000001-cohort-{number}',
            cohort_number=number, cohort_count=2, cancer_treatment_trial='yes', multiple_cohorts='yes',
            cohort_eligibility_criteria='Stale', cohort_specific_eligibility_criteria=f'Population {number}.',
            common_eligibility_criteria='All shared requirements.', eligibility_criteria='Original', split='train'))
    rows.append(dict(trial_id='NCT00000002', cohort_trial_id='NCT00000002-cohort-0', cohort_number=0,
        cohort_count=1, cancer_treatment_trial='yes', multiple_cohorts='no', cohort_eligibility_criteria='TITLE SUMMARY ELIGIBILITY',
        eligibility_criteria='Only original eligibility.', split='val'))
    cohorts = tmp_path / 'cohorts.csv'
    pd.DataFrame(rows).to_csv(cohorts, index=False)
    output = tmp_path / 'corpus'
    result = production.prepare_corpus(patients, cohorts, output)
    ps, cs = load_corpus(output)
    assert 'Other relevant history:\nRenal history.' in ps[0]['text']
    assert cs[-1]['text'] == 'Only original eligibility.'
    assert all('All shared requirements.' in c['text'] and f'Population {i}.' in c['text'] for i, c in enumerate(cs[:2]))
    assert result['source_trial_pairs'] == 2 and result['trials'] == 2
    assert production.prepare_corpus(patients, cohorts, output) == result
    changed = pd.read_parquet(patients)
    changed['patient_summary'] = 'Changed'
    changed.to_parquet(patients)
    with pytest.raises(ValueError, match='Changed TrialSpace'):
        production.prepare_corpus(patients, cohorts, output)


def test_batched_maxsim_matches_exact_and_masks_padding():
    q = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]])
    docs = [torch.tensor([[-.8, -.6]]), torch.tensor([[1., 0.], [-.2, .98]]), torch.randn(7, 2)]
    expected = torch.stack([maxsim(q, d) for d in docs])
    actual = production.score_documents(q, docs, query_block=2, document_block=2)
    torch.testing.assert_close(actual, expected)
    assert actual[0] < .01  # Padded zeros must not replace negative similarities.


def test_packed_pair_batching_matches_reference_with_padding_and_duplicates():
    generator = torch.Generator().manual_seed(123)
    queries = [torch.randn(n, 8, generator=generator) for n in (1, 7, 12)]
    docs = [torch.randn(n, 8, generator=generator) for n in (1, 4, 11, 19)]
    qids, dids = [2, 0, 1, 2, 1, 0], [1, 3, 0, 2, 3, 3]
    expected = torch.stack([maxsim(queries[q], docs[d], query_block=3, document_block=5) for q, d in zip(qids, dids)])
    actual = score_pairs(PackedTokens(queries, 'cpu'), PackedTokens(docs, 'cpu'), qids, dids,
                         pair_batch_size=4, query_block=3, document_block=5)
    torch.testing.assert_close(torch.tensor(actual), expected)
    negative = score_pairs(PackedTokens([torch.ones(7, 8), torch.ones(1, 8)], 'cpu'),
        PackedTokens([-torch.ones(1, 8), -2 * torch.ones(9, 8)], 'cpu'), [0, 1], [0, 1],
        query_block=3, document_block=4)
    assert negative == [-8., -16.]


class ToyModel(torch.nn.Module):
    def __init__(self, folder):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.folder, self.calls = Path(folder), []
    @property
    def device(self):
        return self.weight.device
    def fingerprint(self):
        return dict(toy='deterministic-token-model')
    def encode_batch(self, texts, is_query):
        self.calls.append((is_query, len(texts)))
        if is_query:
            assert all(t.startswith(INSTRUCTION) for t in texts)
        return [torch.eye(4)[[sum(word.encode()) % 4 for word in text.split()]].half() for text in texts]


def test_mining_batches_caches_resumes_and_exports_complete_cohorts(tmp_path):
    corpus, out = tmp_path / 'corpus', tmp_path / 'mining'
    create_fixture(corpus)
    patients, cohorts = load_corpus(corpus)
    write_jsonl(corpus / 'source_trial_pairs.jsonl', read_jsonl(corpus / 'train_pairs.jsonl')[:1])
    args = production.parser().parse_args(['mine', '--corpus', str(corpus), '--output', str(out),
        '--device', 'cpu', '--candidate-limit', '3', '--top-k', '1', '--lexical-k', '1', '--random-k', '1',
        '--document-batch-size', '4', '--query-batch-size', '4', '--score-batch-size', '2', '--cache-batches', '2'])
    model = ToyModel(tmp_path / 'toy')
    production.mine(args, model)
    assert any(is_query and n > 1 for is_query, n in model.calls)
    assert any(not is_query and n > 1 for is_query, n in model.calls)
    for direction in ('patient_to_cohort', 'cohort_to_patient'):
        frame = pd.read_parquet(out / f'{direction}.parquet')
        assert frame.patient_summary.str.contains('Other relevant history:').all()
        assert frame.this_space.equals(frame.cohort_eligibility_criteria)
        assert frame.patient_boilerplate_text.eq('').all() and frame.trial_boilerplate_text.eq('').all()
        assert frame.cohort_trial_id.isin([c['id'] for c in cohorts]).all()
        assert set(frame.split) <= {'train', 'val'}
    calls = len(model.calls)
    production.mine(args, model)
    assert len(model.calls) == calls
    # Interrupted mining reuses token files and completed query checkpoints.
    (out / 'complete.json').unlink()
    production.mine(args, model)
    assert len(model.calls) == calls
    store = production.TokenStore(out / 'tokens', patients, cohorts, args.splits, 4, cache_batches=2)
    for c in cohorts:
        store.get('cohorts', c['id'])
    assert len(store.cache) <= 2
    (out / 'candidates.jsonl').write_text('changed')
    with pytest.raises(ValueError, match='artifacts changed'):
        production.mine(args, model)


def test_resident_mining_preserves_candidates_scores_and_runtime_tuning_resume(tmp_path):
    corpus = tmp_path / 'corpus'
    create_fixture(corpus)
    outputs = []
    for name, option in [('resident', []), ('fallback', ['--device-cache-gb', '0'])]:
        out = tmp_path / name
        args = production.parser().parse_args(['mine', '--corpus', str(corpus), '--output', str(out),
            '--device', 'cpu', '--candidate-limit', '3', '--top-k', '2', '--lexical-k', '1', '--random-k', '1',
            '--document-batch-size', '4', '--query-batch-size', '2', '--score-batch-size', '2',
            '--pair-batch-size', '5', '--score-query-batches', '3', *option])
        model = ToyModel(tmp_path / 'toy')
        production.mine(args, model)
        outputs.append(out)
        if name == 'resident':
            # Runtime cache/pair scheduling may change without invalidating
            # established embedding/candidate/scoring identities.
            (out / 'complete.json').unlink()
            calls = len(model.calls)
            args.pair_batch_size, args.score_query_batches = 3, 2
            production.mine(args, model)
            assert len(model.calls) == calls
    for artifact in ('rankings.jsonl', 'candidates.jsonl'):
        assert read_jsonl(outputs[0] / artifact) == read_jsonl(outputs[1] / artifact)


def test_batched_checkpoints_resume_legacy_and_change_rank_batch_partition(tmp_path):
    from colbert_experiments.query_checkpoints import QueryCheckpoints
    from colbert_experiments.data import write_json
    old = tmp_path / 'queries/aa/aa1.json'
    write_json(old, [dict(score=1)])
    store = QueryCheckpoints(tmp_path)
    store.write_batch({'bb2': [dict(score=2)], 'cc3': [dict(score=3)]})
    # A partial temporary file is invisible on restart.
    (tmp_path / 'query_batches/partial.json.tmp').write_text('{')
    resumed = QueryCheckpoints(tmp_path)
    assert resumed.keys == {'aa1', 'bb2', 'cc3'}
    resumed.write_batch({'dd4': [dict(score=4)]})
    assert store.ordered_hits(['dd4', 'aa1', 'cc3', 'bb2']) == [dict(score=i) for i in (4, 1, 3, 2)]
    with pytest.raises(ValueError, match='only new queries'):
        resumed.write_batch({'cc3': []})


def test_training_uses_runtime_encoder_scorer_and_query_contract():
    from matchminer_ai.embedding.colbert import ColBERT
    from matchminer_ai.embedding.packed_scoring import score_pairs as runtime_score_pairs
    from matchminer_ai.embedding.colbert_contract import regression_query
    from colbert_experiments import model, regression
    assert model.ColBERT is ColBERT
    assert score_pairs is runtime_score_pairs
    assert regression.regression_query is regression_query


def test_vectorized_candidate_order_preserves_negative_scores_and_index_ties():
    import numpy as np
    from colbert_experiments.retrieval import top_indices
    scores = np.random.default_rng(42).normal(size=54910)
    scores[::3] = 0
    expected = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
    for count in (0, 5, 128, len(scores)):
        assert top_indices(scores, count) == expected[:count]


def test_candidate_export_freezes_prompt_contract_once_for_both_directions(tmp_path, monkeypatch):
    corpus = tmp_path / 'corpus'
    create_fixture(corpus)
    pairs = read_jsonl(corpus / 'train_pairs.jsonl')[:3]
    for pair in pairs:
        pair['sources'] = [dict(direction='both', source='fixture')]
    candidates = tmp_path / 'candidates.jsonl'
    write_jsonl(candidates, pairs)
    calls = []
    def contract():
        calls.append(1)
        return 'frozen-test-contract'
    monkeypatch.setattr(production, 'score_contract', contract)
    outputs = [tmp_path / 'patient.parquet', tmp_path / 'cohort.parquet']
    production.export_candidates(corpus, candidates, *outputs)
    assert len(calls) == 1
    for path in outputs:
        table = pd.read_parquet(path)
        assert len(table) == len(pairs)
        assert table.trialspace_contract.eq('frozen-test-contract').all()


def test_regression_cpu_fit_resumes_and_saves_separate_production_artifact(tmp_path, monkeypatch):
    from colbert_experiments import training
    corpus = tmp_path / 'corpus'
    create_fixture(corpus)
    answers = read_jsonl(corpus / 'reference_labels.jsonl')
    grouped = {}
    for row in answers:
        row['key'] = label_key(row)
        grouped.setdefault((row['patient_id'], row['cohort_id']), []).append(row)
    scores = [aggregate_pair(rows[0], rows) for rows in grouped.values()]
    write_jsonl(tmp_path / 'scores.jsonl', scores)
    class Student(torch.nn.Module):
        def __init__(self, *a, **kw):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(.9))
        @property
        def device(self):
            return self.weight.device
        def fingerprint(self):
            return dict(toy='trainable')
        def forward(self, query, documents):
            assert query.startswith(INSTRUCTION)
            return self.weight.expand(len(documents))
        def save(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)
            torch.save(self.state_dict(), Path(path) / 'model.pt')
    monkeypatch.setattr(training, 'ColBERT', Student)
    command = ['train', '--device', 'cpu', '--corpus', str(corpus), '--labels', str(tmp_path / 'scores.jsonl'),
        '--output', str(tmp_path / 'fit'), '--output-model', str(tmp_path / 'model'),
        '--loss', 'mse', '--lr-scheduler', 'constant', '--learning-rate', '1e-5',
        '--epochs', '1', '--accumulation', '2', '--eval-every', '2', '--save-steps', '2']
    training.train(production.parser().parse_args(command))
    status = json.loads((tmp_path / 'fit/status.json').read_text())
    assert status['state'] == 'complete' and status['model'] == str(tmp_path / 'model')
    assert status['steps'] == 11  # Both source partitions: 14 patient + 8 cohort groups.
    assert torch.load(tmp_path / 'model/model.pt', weights_only=True)['weight'] < .9
    saved = json.loads((tmp_path / 'model/regression_scoring.json').read_text())
    assert saved['prediction_link'] == 'identity' and saved['output_scale'] == 5
    training.train(production.parser().parse_args(command))
    metrics = json.loads((tmp_path / 'fit/metrics.json').read_text())
    assert len(metrics['history']) == 11
    assert all('val' not in entry for entry in metrics['evaluations'])
    assert all(sum(entry['batch_score_counts'].values()) == entry['batch_pairs']
               and entry['batch_query_groups'] == 2 for entry in metrics['history'])
    manifest = json.loads((tmp_path / 'fit/training_manifest.json').read_text())
    assert manifest['batch_sampling']['policy'] == 'direction_score_stratified_v1'
    assert manifest['max_pairs_per_update'] == 16
    assert manifest['training_splits'] == ['train', 'val', 'test']
    assert not manifest['validation_is_held_out']
    with pytest.raises(ValueError, match='identity changed'):
        training.train(production.parser().parse_args(command + ['--documents-per-query', '4']))


@pytest.mark.parametrize('loss', ['legacy_ranking', 'cosent'])
@pytest.mark.parametrize('backward_mode', ['replay', 'retained'])
def test_production_ranking_defaults_cover_all_pairs_evaluate_and_resume(tmp_path, monkeypatch, loss, backward_mode):
    from colbert_experiments import training
    corpus = tmp_path/'corpus'
    create_fixture(corpus)
    grouped = {}
    for row in read_jsonl(corpus/'reference_labels.jsonl'):
        row['key'] = label_key(row)
        grouped.setdefault((row['patient_id'], row['cohort_id']), []).append(row)
    scores = [aggregate_pair(rows[0], rows) for rows in grouped.values()]
    write_jsonl(tmp_path/'scores.jsonl', scores)

    class Student(torch.nn.Module):
        def __init__(self, *a, **kw):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(.9))
        @property
        def device(self): return self.weight.device
        def fingerprint(self): return dict(toy='ranking')
        def prepare_groups(self, queries, documents): return list(zip(queries, documents))
        def prepared_batch_tokens(self, groups): return sum(len(q) + sum(map(len, ds)) for q, ds in groups)
        def forward(self, query=None, documents=None, *, prepared_groups=None):
            groups = prepared_groups if prepared_groups is not None else [(query, documents)]
            assert all(q.startswith(INSTRUCTION) for q, _ in groups)
            documents = [d for _, ds in groups for d in ds]
            return self.weight * torch.tensor([len(d) / 1000 for d in documents])
        def save(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)
            torch.save(self.state_dict(), Path(path)/'model.pt')
    monkeypatch.setattr(training, 'ColBERT', Student)
    command = ['train', '--device', 'cpu', '--corpus', str(corpus), '--labels', str(tmp_path/'scores.jsonl'),
        '--output', str(tmp_path/'fit'), '--epochs', '1', '--documents-per-query', '3',
        '--accumulation', '5', '--eval-every', '2', '--save-steps', '2']
    if loss != 'legacy_ranking':
        command += ['--loss', loss]
    if backward_mode == 'retained':
        command += ['--ranking-backward', 'retained', '--no-gradient-checkpointing']
    args = production.parser().parse_args(command)
    assert args.loss == loss and args.learning_rate == 2e-5 and args.lr_scheduler == 'linear'
    training.train(args)
    manifest = json.loads((tmp_path/'fit/training_manifest.json').read_text())
    assert manifest['loss'] == loss and 'normalization' not in manifest
    assert manifest['batch_sampling']['policy'] == 'complete_query_pair_epochs_v1'
    metrics = json.loads((tmp_path/'fit/metrics.json').read_text())
    assert sum(r['batch_pairs'] for r in metrics['history']) == 2 * len(scores)
    assert all(r['batch_unique_directional_pairs'] == r['batch_pairs'] for r in metrics['history'])
    assert any(r['gradient_norm'] > 0 for r in metrics['history'])
    if backward_mode == 'replay':
        assert all(r['replay_max_error'] == 0 for r in metrics['history'])
    else:
        assert manifest['effective_batch_comparisons'] == 'single forward with retained activations'
        assert all(r['encoder_forward_passes'] == r['model_backward_calls'] == 1 for r in metrics['history'])
        assert all('replay_backward_seconds' not in r for r in metrics['history'])
    assert any(r['encoder_batches_per_pass'] < r['batch_query_groups'] for r in metrics['history'])
    for evaluation in metrics['evaluations']:
        assert 'val' not in evaluation  # Production includes every original split.
        train = evaluation['train']
        assert train['objective'] == loss and 'mse_diagnostic' in train and 'mae' not in train
        assert all('auroc' in values for values in train['strata'].values())
    scoring = json.loads((tmp_path/'fit/model/regression_scoring.json').read_text())
    assert scoring['objective'] == loss and not scoring['calibrated'] and 'output_scale' not in scoring
    # The inference dependency accepts ordinal metadata using the same native
    # MaxSim and instruction, without requiring a regression calibration scale.
    from matchminer_ai.embedding import late_interaction as runtime
    def runtime_student(path, **kwargs):
        model = Student()
        model.folder = Path(path)
        return model
    monkeypatch.setattr(runtime, 'ColBERT', runtime_student)
    loaded = runtime._model.__wrapped__(str(tmp_path/'fit/model'), None, 'cpu', 8192, 8192, None)
    assert loaded.folder == tmp_path/'fit/model'
    assert len(list((tmp_path/'fit').glob('checkpoint-*.pt'))) == 2
    latest = json.loads((tmp_path/'fit/latest.json').read_text())
    state = torch.load(tmp_path/'fit'/latest['path'], weights_only=True)
    assert state['scheduler']['last_epoch'] == manifest['steps']
    training.train(production.parser().parse_args(command))
    resumed = torch.load(tmp_path/'fit/model/model.pt', weights_only=True)
    torch.testing.assert_close(resumed['weight'], state['model']['weight'])
    with pytest.raises(ValueError, match='identity changed|Retained ranking requires'):
        training.train(production.parser().parse_args(command + ['--loss', 'mse']))


def test_standard_entrypoints_forward_explicit_ranking_overrides():
    import run_pipeline
    args = run_pipeline.parser().parse_args(['--trialspace-loss', 'cosent', '--trialspace-ranking-scale', '10',
        '--trialspace-contrastive-weight', '.5', '--trialspace-learning-rate', '3e-5',
        '--trialspace-lr-scheduler', 'constant', '--trialspace-warmup-ratio', '.02'])
    for stage in runner.stages(args):
        if stage.name.startswith('train-trialspace-'):
            parsed = production.parser().parse_args(stage.command[stage.command.index('trialspace_training.py') + 1:])
            assert parsed.loss == 'cosent' and parsed.ranking_scale == 10 and parsed.contrastive_weight == .5
            assert parsed.learning_rate == 3e-5 and parsed.lr_scheduler == 'constant' and parsed.warmup_ratio == .02


def test_directional_mining_returns_twenty_cohorts_and_forty_patients(tmp_path):
    corpus, out = tmp_path / 'corpus', tmp_path / 'mining'
    write_jsonl(corpus / 'patients.jsonl', [dict(id=f'p{i}', text=f'Synthetic cancer history number {i}.', split='train') for i in range(45)])
    write_jsonl(corpus / 'cohorts.jsonl', [dict(id=f'c{i}', trial_id=f't{i}', text=f'Synthetic cancer eligibility number {i}.', split='train') for i in range(25)])
    args = production.parser().parse_args(['mine', '--corpus', str(corpus), '--output', str(out),
        '--device', 'cpu', '--candidate-limit', '50', '--top-k-patient-to-cohort', '20',
        '--top-k-cohort-to-patient', '40', '--lexical-k', '0', '--random-k', '0'])
    production.mine(args, ToyModel(tmp_path / 'toy'))
    rankings = pd.DataFrame(read_jsonl(out / 'rankings.jsonl'))
    assert rankings[rankings.direction.eq('patient_to_cohort')].groupby('patient_id').cohort_id.nunique().eq(20).all()
    assert rankings[rankings.direction.eq('cohort_to_patient')].groupby('cohort_id').patient_id.nunique().eq(40).all()
    assert len(rankings) == 45 * 20 + 25 * 40


def test_all_synthetic_pool_retrieves_across_original_splits_and_requires_complete_labels(tmp_path):
    from colbert_experiments import training
    from colbert_experiments.contracts import contract_hash
    from colbert_experiments.labeling import questions
    patients, cohorts = [], []
    for i, split in enumerate(('train', 'val', 'test')):
        trial = f'NCT{i+1:08d}'
        patients.append(dict(pseudo_mrn=f'p{i}', patient_summary=f'Complete synthetic history {i}.',
                             split=split, nct_id=trial))
        cohorts.append(dict(trial_id=trial, cohort_trial_id=f'c{i}', cohort_number=0, cohort_count=1,
                            cancer_treatment_trial='yes', multiple_cohorts='no',
                            eligibility_criteria=f'Full synthetic eligibility {i}.',
                            cohort_eligibility_criteria=f'Full synthetic eligibility {i}.', split=split))
    patient_path, cohort_path, corpus = tmp_path / 'patients.parquet', tmp_path / 'cohorts.csv', tmp_path / 'corpus'
    pd.DataFrame(patients).to_parquet(patient_path)
    pd.DataFrame(cohorts).to_csv(cohort_path, index=False)
    production.prepare_corpus(patient_path, cohort_path, corpus)
    ps, cs = load_corpus(corpus)
    assert {r['split'] for r in ps + cs} == {'train'}
    assert {r['source_split'] for r in ps} == {'train', 'val', 'test'}
    assert len(read_jsonl(corpus / 'source_trial_pairs.jsonl')) == 3
    out = tmp_path / 'mining'
    args = production.parser().parse_args(['mine', '--corpus', str(corpus), '--output', str(out),
        '--device', 'cpu', '--candidate-limit', '3', '--top-k-patient-to-cohort', '3',
        '--top-k-cohort-to-patient', '3', '--lexical-k', '0', '--random-k', '0'])
    production.mine(args, ToyModel(tmp_path / 'toy'))
    pairs = read_jsonl(out / 'candidates.jsonl')
    assert {(r['patient_id'], r['cohort_id']) for r in pairs} == {(f'p{i}', f'c{j}') for i in range(3) for j in range(3)}
    assert len(list(questions(pairs, ps, cs))) == 45
    for direction in ('patient_to_cohort', 'cohort_to_patient'):
        rows = pd.read_parquet(out / f'{direction}.parquet')
        cross = rows[rows.patient_id.eq('p0') & rows.cohort_trial_id.eq('c2')].iloc[0]
        assert cross['split'] == 'train' and cross.patient_source_split == 'train' and cross.cohort_source_split == 'test'
    scores = []
    for pair in pairs:
        if pair['patient_id'] == 'p2':
            continue
        row = dict(**pair, task='potentially_eligible', answer='no', evidence_status='supported',
                   reason='Fabricated negative.', contract=contract_hash())
        row['key'] = label_key(row)
        scores.append(aggregate_pair(pair, [row]))
    write_jsonl(tmp_path / 'partial_scores.jsonl', scores)
    fit = production.parser().parse_args(['train', '--device', 'cpu', '--corpus', str(corpus),
        '--labels', str(tmp_path / 'partial_scores.jsonl'), '--output', str(tmp_path / 'fit'), '--steps', '1'])
    with pytest.raises(ValueError, match='1 patients, 0 cohorts'):
        training.train(fit)
