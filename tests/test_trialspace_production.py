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
        if parsed.command == 'label':
            assert parsed.gated_regression and not parsed.return_records
            assert parsed.scheduler_adaptive_concurrency
            assert parsed.teacher_tokenizer == args.teacher_model
        elif parsed.command in ('mine', 'train'):
            assert parsed.objective == 'gated_regression'
            assert parsed.query_length == parsed.document_length == 8192
            if parsed.command == 'train':
                assert parsed.prediction_link == 'identity' and parsed.steps == 8
                assert parsed.output_model.parent == args.models_dir
    assert not any(script in str([s.command for s in plan]) for script in ('finetune_embedder.py', 'llm_check_trials.py', 'make_top_matches.py'))
    stages = {s.name: s for s in plan}
    for iteration in (1, 2):
        fit = stages[f'train-trialspace-{iteration}'].command
        label = stages[f'label-trialspace-{iteration}'].outputs[0]
        assert fit[fit.index('--labels') + 1] == str(label)
        previous = args.embedding_model if iteration == 1 else str(args.models_dir / 'trialspace_round1')
        assert fit[fit.index('--model') + 1] == previous
    assert stages['mine-3'].command[stages['mine-3'].command.index('--model') + 1] == str(args.models_dir / 'trialspace_round2')


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
        '--epochs', '1', '--accumulation', '2', '--eval-every', '2', '--save-steps', '2']
    training.train(production.parser().parse_args(command))
    status = json.loads((tmp_path / 'fit/status.json').read_text())
    assert status['state'] == 'complete' and status['model'] == str(tmp_path / 'model')
    assert status['steps'] == 6  # 7 patient + 4 cohort query groups; ceil(11 / 2).
    assert torch.load(tmp_path / 'model/model.pt', weights_only=True)['weight'] < .9
    saved = json.loads((tmp_path / 'model/regression_scoring.json').read_text())
    assert saved['prediction_link'] == 'identity' and saved['output_scale'] == 5
    training.train(production.parser().parse_args(command))
    metrics = json.loads((tmp_path / 'fit/metrics.json').read_text())
    assert len(metrics['history']) == 6


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
