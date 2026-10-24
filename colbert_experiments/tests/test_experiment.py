import asyncio
import json
import random
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from colbert_experiments.contracts import TASKS, VERSION, judge_messages, parse_judgment, prompt_sources, query_text
from colbert_experiments.data import combined_patient_history, load_corpus, prepare, read_jsonl, validate_corpus, write_jsonl
from colbert_experiments.fixtures import create_fixture
from colbert_experiments.labeling import questions
from colbert_experiments.model import ColBERT, grouped_loss, maxsim
from colbert_experiments.retrieval import evaluate_rankings, merge_candidates, proposal_indices
from colbert_experiments.training import build_groups
from colbert_experiments.learning import make_probes, score_statistics


def test_maxsim_matches_dense_and_backpropagates():
    torch.manual_seed(2)
    q, d = torch.randn(7, 4, requires_grad=True), torch.randn(13, 4, requires_grad=True)
    expected = (q @ d.T).max(dim=1).values.mean()
    actual = maxsim(q, d, query_block=3, document_block=5)
    torch.testing.assert_close(actual, expected)
    gradients = torch.autograd.grad(actual, (q, d), retain_graph=True)
    dense = torch.autograd.grad(expected, (q, d))
    for a, b in zip(gradients, dense):
        torch.testing.assert_close(a, b)


def test_explicit_multiple_positives_are_never_negatives():
    scores = torch.tensor([.8, .7, .6], requires_grad=True)
    loss = grouped_loss(scores, [1, 1, 0])
    loss.backward()
    assert (scores.grad[:2] < 0).all() and scores.grad[2] > 0
    with pytest.raises(ValueError, match='positive and negative'):
        grouped_loss(scores, [1, 1, 1])


def test_bounded_proposals_retain_lexical_and_random_candidates():
    scores = list(range(20))
    chosen = proposal_indices(scores, 6, random.Random(42), 'lexical_random')
    assert len(chosen) == len(set(chosen)) == 6
    assert {17, 18, 19} <= set(chosen)
    assert chosen == proposal_indices(scores, 6, random.Random(42), 'lexical_random')
    assert proposal_indices(scores, None, random.Random(1)) == list(range(20))


def test_round_concurrency_can_change_without_reusing_different_labels():
    from colbert_experiments.run_round import semantic_commands
    first = dict(labeling=['python', '--model', 'gemma', '--pending-requests', '16'], training=['train', '--steps', '500'])
    resized = dict(first, labeling=['python', '--model', 'gemma', '--pending-requests', '96'])
    assert semantic_commands(first) == semantic_commands(resized)
    changed_model = dict(first, labeling=['python', '--model', 'other', '--pending-requests', '96'])
    assert semantic_commands(first) != semantic_commands(changed_model)
    assert semantic_commands(first) != semantic_commands(dict(first, training=['train', '--steps', '600']))
    rerouted = dict(first, labeling=first['labeling'] + ['--server_urls', 'http://local/v1',
                                                       '--teacher-served-model', 'verified-root'])
    assert semantic_commands(first) == semantic_commands(rerouted)


def test_training_devices_can_change_only_before_fit_and_with_equal_effective_batch():
    from colbert_experiments.run_round import compatible_commands
    old = dict(labeling=['label'], training=['train', '--nproc_per_node', '2', '--accumulation', '2', '--steps', '500'])
    new = dict(old, training=['train', '--nproc_per_node', '1', '--accumulation', '4', '--steps', '500'])
    assert compatible_commands(old, new, training_started=False)
    assert not compatible_commands(old, new, training_started=True)
    smaller = dict(new, training=['train', '--nproc_per_node', '1', '--accumulation', '2', '--steps', '500'])
    assert not compatible_commands(old, smaller, training_started=False)
    changed_steps = dict(new, training=['train', '--nproc_per_node', '1', '--accumulation', '4', '--steps', '600'])
    assert not compatible_commands(old, changed_steps, training_started=False)


def test_gpu_guard_ignores_unselected_devices(monkeypatch):
    from colbert_experiments import run_round
    def query(command, **kwargs):
        if '--query-gpu=index,uuid' in command:
            return '2, GPU-two\n3, GPU-three\n'
        return 'GPU-three, 12345\n'
    monkeypatch.setattr(run_round.subprocess, 'check_output', query)
    assert run_round.busy_training_gpus(['2']) == []
    assert run_round.busy_training_gpus(['2', '3']) == [dict(gpu='3', pid=12345)]
    with pytest.raises(ValueError, match='unavailable'):
        run_round.busy_training_gpus(['4'])


def test_registered_teacher_shutdown_refuses_reused_pid_and_other_gpu(tmp_path, monkeypatch):
    from colbert_experiments import local_teacher
    target = dict(purpose='stop_this_teacher_after_labeling', gpu=2, port=8003,
                  process=dict(pid=12345, start_ticks=100, command=['vllm', 'serve', 'model', '--port', '8003']))
    registration, receipt = tmp_path / 'teacher.json', tmp_path / 'receipt.json'
    registration.write_text(json.dumps(target))
    calls = []
    monkeypatch.setattr(local_teacher, 'open_pidfd', lambda pid: 123)
    monkeypatch.setattr(local_teacher.os, 'close', lambda fd: calls.append(('close', fd)))
    monkeypatch.setattr(local_teacher, 'send_pidfd_signal', lambda fd, sig: calls.append(('signal', fd, sig)))
    monkeypatch.setattr(local_teacher, 'process_identity', lambda pid: {**target['process'], 'start_ticks': 101})
    with pytest.raises(ValueError, match='identity changed'):
        local_teacher.stop_registered_teacher(registration, receipt, training_gpus=['2'])
    assert not any(c[0] == 'signal' for c in calls)
    with pytest.raises(ValueError, match='selected training GPU'):
        local_teacher.stop_registered_teacher(registration, receipt, training_gpus=['3'])
    monkeypatch.setattr(local_teacher, 'process_identity', lambda pid: target['process'])
    local_teacher.stop_registered_teacher(registration, receipt, training_gpus=['2'])
    assert [c for c in calls if c[0] == 'signal'] == [('signal', 123, local_teacher.signal.SIGTERM)]
    assert json.loads(receipt.read_text())['state'] == 'termination_requested'
    monkeypatch.setattr(local_teacher, 'process_identity', lambda pid: None)
    local_teacher.stop_registered_teacher(registration, receipt, training_gpus=['2'])
    assert json.loads(receipt.read_text())['state'] == 'already_stopped'
    assert len([c for c in calls if c[0] == 'signal']) == 1


def test_teacher_endpoint_alias_requires_an_approved_model_root():
    from colbert_experiments.labeling import validate_served_teacher
    root = 'RedHatAI/Gemma-4-31B-IT-FP8-Dynamic'
    model = dict(id=root, root=root, max_model_len=262144)
    assert validate_served_teacher([model], requested=root, tokenizer='nvidia/Gemma-4-31B-IT-NVFP4') == model
    with pytest.raises(ValueError, match='model family'):
        validate_served_teacher([dict(id=root, root='unrelated-model')], requested=root,
                                tokenizer='nvidia/Gemma-4-31B-IT-NVFP4')
    with pytest.raises(ValueError, match='not uniquely advertised'):
        validate_served_teacher([model], requested='missing-alias', tokenizer=root)


def test_fixed_probes_and_metrics_compare_the_same_explicit_labels(tmp_path):
    create_fixture(tmp_path)
    patients, cohorts = load_corpus(tmp_path)
    groups, _ = build_groups(patients, cohorts, read_jsonl(tmp_path / 'reference_labels.jsonl'))
    probes = make_probes(groups, per_stratum=2, seed=42)
    assert probes == make_probes(groups, per_stratum=2, seed=42)
    assert len({p['stratum'] for p in probes}) == 10
    assert all(set(p['labels']) == {0, 1} for p in probes)
    before = score_statistics(torch.tensor([.3, .7]), [1, 0], .05)
    after = score_statistics(torch.tensor([.7, .3]), [1, 0], .05)
    assert after['loss'] < before['loss']
    assert before['pairwise_accuracy'] == 0 and after['pairwise_accuracy'] == 1
    assert after['precision_at_1'] == 1 and after['margin'] > 0


def test_v22_sampling_preserves_cohorts_and_excludes_conflicting_lineage(tmp_path):
    from colbert_experiments.data import sample_v22
    rows = [dict(pseudo_mrn=i, nct_id=trial, split=split, patient_summary='Cancer',
                 patient_boilerplate_text='Renal history') for i, trial, split in
            [(1, 'T', 'train'), (2, 'T', 'train'), (3, 'V', 'val'), (4, 'V', 'train')]]
    pd.DataFrame(rows).to_parquet(tmp_path / 'patients.parquet')
    cohorts = pd.DataFrame([dict(trial_id=trial, cohort_trial_id=trial + '-0', cohort_number=0,
        cohort_count=1, cancer_treatment_trial='yes', multiple_cohorts='no', cohort_eligibility_criteria='Criteria',
        eligibility_criteria='Criteria', split=split) for trial, split in [('T', 'train'), ('V', 'val')]])
    cohorts.to_csv(tmp_path / 'cohorts.csv', index=False)
    report = sample_v22(tmp_path / 'patients.parquet', tmp_path / 'cohorts.csv', tmp_path / 'out',
                        train_count=1, val_count=1)
    p, c = load_corpus(tmp_path / 'out')
    assert report['conflicting_origin_splits'] == 1 and len(c) == 2
    assert '4' not in {r['id'] for r in p}
    assert all('Other relevant history:\nRenal history' in r['text'] for r in p)


def test_patient_keeps_other_history_and_no_core_only_fallback():
    text = combined_patient_history(dict(patient_summary='Cancer summary.', patient_boilerplate_text='Dialysis.'))
    assert text == 'Cancer history:\nCancer summary.\n\nOther relevant history:\nDialysis.'
    with pytest.raises(ValueError, match='Missing patient'):
        combined_patient_history(dict(patient_boilerplate_text='Dialysis.'))


def test_prepare_rebuilds_complete_criteria(tmp_path):
    patients = pd.DataFrame([dict(patient_id='p', patient_summary='Cancer', split='train')])
    cohorts = pd.DataFrame([dict(trial_id='T', cohort_trial_id='T-0', cohort_number=0, cohort_count=1,
        cancer_treatment_trial='yes', multiple_cohorts='no', cohort_eligibility_criteria='STALE TITLE SUMMARY',
        eligibility_criteria='ELIGIBILITY ONLY', split='train')])
    patients.to_csv(tmp_path / 'p.csv', index=False)
    cohorts.to_csv(tmp_path / 'c.csv', index=False)
    prepare(tmp_path / 'p.csv', tmp_path / 'c.csv', tmp_path / 'out')
    _, result = load_corpus(tmp_path / 'out')
    assert result[0]['text'] == 'ELIGIBILITY ONLY'
    multi = []
    for i in range(2):
        multi.append(dict(cohorts.iloc[0], cohort_trial_id=f'T-{i}', cohort_number=i,
            cohort_count=2, multiple_cohorts='yes', cohort_specific_eligibility_criteria=f'Group {i}',
            common_eligibility_criteria='Full common criteria'))
    pd.DataFrame(multi).to_csv(tmp_path / 'c.csv', index=False)
    prepare(tmp_path / 'p.csv', tmp_path / 'c.csv', tmp_path / 'out2')
    _, result = load_corpus(tmp_path / 'out2')
    assert result[1]['text'] == 'Cohort-specific eligibility criteria:\nGroup 1\n\nCommon eligibility criteria:\nFull common criteria'


def test_grouping_dedup_split_isolation_and_all_task_directions(tmp_path):
    create_fixture(tmp_path)
    patients, cohorts = load_corpus(tmp_path)
    labels = read_jsonl(tmp_path / 'reference_labels.jsonl')
    groups, report = build_groups(patients, cohorts, labels + [labels[0]])
    assert len(report['strata']) == 10 and all(report['strata'].values())
    assert all('TRAIN' in group['query_id'] for group in groups)
    assert all(group['positive'] and group['negative'] for group in groups)
    with pytest.raises(ValueError, match='Conflicting'):
        build_groups(patients, cohorts, labels + [{**labels[0], 'answer': 'no'}])
    with pytest.raises(ValueError, match='crosses'):
        build_groups(patients, cohorts, [{**labels[0], 'split': 'test'}])
    with pytest.raises(ValueError, match='contract'):
        build_groups(patients, cohorts, [{**labels[0], 'contract': 'old-rubric'}])


def test_source_trial_and_sibling_split_guards():
    cohort = dict(id='T-0', trial_id='T', text='Criteria', split='train')
    patient = dict(id='p', text='Summary', source_trial_id='T', split='val')
    with pytest.raises(ValueError, match='source trial'):
        validate_corpus([patient], [cohort])
    with pytest.raises(ValueError, match='Sibling'):
        validate_corpus([], [cohort, {**cohort, 'id': 'T-1', 'split': 'val'}])


def test_union_labels_all_questions_once_for_both_directions(tmp_path):
    create_fixture(tmp_path)
    patients, cohorts = load_corpus(tmp_path)
    hits = [dict(patient_id=patients[0]['id'], cohort_id=cohorts[0]['id'], task=task,
                 direction=direction, source='colbert')
            for task in TASKS for direction in ('patient_to_cohort', 'cohort_to_patient')]
    pairs = merge_candidates(patients, cohorts, hits)
    assert len(pairs) == 1 and len(pairs[0]['sources']) == 10
    assert len(list(questions(pairs + pairs, patients, cohorts))) == 5


def test_json_fences_and_binary_contract():
    value = dict(answer='yes', evidence_status='insufficient_information', reason='Undocumented ECOG.')
    assert parse_judgment('```json\n' + json.dumps(value) + '\n```') == value
    for bad in [{**value, 'answer': 1}, {**value, 'reason': ''}, {**value, 'extra': True}]:
        with pytest.raises(ValueError):
            parse_judgment(json.dumps(bad))


def test_prompt_biomarker_requirements_and_retrieval_cues():
    for task in TASKS:
        messages = judge_messages(task, 'FULL PATIENT', 'FULL CRITERIA')
        assert messages[0]['role'] == 'system'
        assert messages[0]['content'].count('Question:') == 1
        assert 'exactly the ONE question' in messages[0]['content']
        assert 'FULL PATIENT' in messages[1]['content'] and 'FULL CRITERIA' in messages[1]['content']
        assert query_text(task, 'cohort_to_patient', 'FULL CRITERIA').endswith('FULL CRITERIA')
    biomarker_prompt = judge_messages('biomarker', 'p', 'c')[0]['content']
    assert 'required eligibility biomarkers' in biomarker_prompt
    assert 'not whether the study treatment targets' in biomarker_prompt


def test_five_tasks_merge_eligibility_and_reject_retired_question(tmp_path):
    assert VERSION == 'cohort-colbert-five-tasks-v2'
    assert set(TASKS) == {'potentially_eligible', 'cancer_type', 'burden', 'prior_treatment', 'biomarker'}
    assert 'reasonable.txt' not in prompt_sources()
    with pytest.raises(ValueError, match='Unknown task'):
        judge_messages('reasonable', 'Patient', 'Criteria')
    prompt = judge_messages('potentially_eligible', 'Patient', 'Criteria')[0]['content']
    assert 'yes if there is no clear evidence the patient is ineligible' in prompt
    assert 'age, sex, cancer type/histology' in prompt and 'organ function, comorbidities' in prompt
    create_fixture(tmp_path)
    labels = read_jsonl(tmp_path / 'reference_labels.jsonl')
    by_key = {(r['patient_id'], r['cohort_id'], r['task']): r for r in labels}
    unknown = by_key['FABRICATED_TRAIN_a_unknown_biomarker', 'FABRICATED_TRAIN_A-0', 'potentially_eligible']
    assert unknown['answer'] == 'yes' and unknown['evidence_status'] == 'insufficient_information'
    assert by_key['FABRICATED_TRAIN_a_unknown_biomarker', 'FABRICATED_TRAIN_A-0', 'biomarker']['answer'] == 'no'
    for patient, cohort in [('b_threshold', 'B'), ('a_brain', 'A'), ('a_good', 'D')]:
        assert by_key[f'FABRICATED_TRAIN_{patient}', f'FABRICATED_TRAIN_{cohort}-0', 'potentially_eligible']['answer'] == 'no'
    with pytest.raises(ValueError, match='contract'):
        evaluate_rankings([], [{**labels[0], 'contract': 'six-task-prompt-hash'}])


def test_recall_counts_missing_queries_and_does_not_mix_splits():
    labels = [dict(patient_id=p, cohort_id='c', task='potentially_eligible', answer='yes', split='val') for p in ('p1', 'p2')]
    rankings = [dict(patient_id=p, cohort_id='c', task='potentially_eligible', direction='patient_to_cohort', rank=1)
                for p in ('p1', 'train_patient')]
    report = evaluate_rankings(rankings, labels, k=1)['patient_to_cohort/potentially_eligible']
    assert report['queries'] == 2 and report['recall_at_k'] == .5


class FakeTokenizer:
    def __call__(self, text, **kwargs):
        assert kwargs['truncation'] is False
        ids = [1] + [4] * len(text.split()) + [2]
        return dict(input_ids=torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long))


def test_marker_and_explicit_overflow_without_loading_weights():
    fake = SimpleNamespace(tokenizer=FakeTokenizer(), prefixes={'query': 8, 'document': 9},
                           query_length=5, document_length=6, device=torch.device('cpu'))
    assert ColBERT.tokens(fake, 'one two', is_query=True)['input_ids'].tolist() == [[1, 8, 4, 4, 2]]
    with pytest.raises(ValueError, match='no truncation'):
        ColBERT.tokens(fake, 'one two three', is_query=True)
    assert ColBERT.tokens(fake, 'one two three', is_query=False)['input_ids'].shape[1] == 6


@pytest.mark.parametrize('objective', ['binary', 'regression'])
def test_all_encoder_paths_batch_roles_and_keep_training_gradients(objective):
    class Tokenizer:
        padding_side = 'right'
        def __call__(self, texts, **kwargs):
            assert kwargs['truncation'] is False and kwargs['padding'] is True
            vocab = {'one': 4, 'two': 5, '.': 6, 'three': 7}
            rows = [[1] + [vocab[word] for word in text.split()] + [2] for text in texts]
            width = max(map(len, rows))
            return dict(input_ids=torch.tensor([r + [0] * (width - len(r)) for r in rows]),
                        attention_mask=torch.tensor([[1] * len(r) + [0] * (width - len(r)) for r in rows]))

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(10, 4)
            self.batch_sizes = []
        def forward(self, input_ids, attention_mask):
            self.batch_sizes.append(input_ids.shape[0])
            return SimpleNamespace(last_hidden_state=self.embedding(input_ids))

    torch.manual_seed(5)
    model = ColBERT.__new__(ColBERT)
    torch.nn.Module.__init__(model)
    model.tokenizer, model.encoder = Tokenizer(), Encoder()
    model.projection = torch.nn.Linear(4, 2, bias=False)
    model.prefixes, model.skiplist = {'query': 8, 'document': 9}, {6}
    model.query_length, model.document_length = 5, 8
    scores = model('one .', ['one .', 'two . three'])
    if objective == 'regression':
        from colbert_experiments.regression import regression_loss
        loss = regression_loss(scores, [5, 0])
    else:
        loss = grouped_loss(scores, [1, 0])
    loss.backward()
    assert model.encoder.batch_sizes == [3]
    assert model.projection.weight.grad.abs().sum() > 0
    query, document = model.encode_texts(['one .', 'one . two three three'], is_queries=[True, False])
    assert len(query) == 5 and len(document) == 7  # Query punctuation stays; document punctuation is masked.
    model.encoder.batch_sizes.clear()
    model.encode(['one .', 'two .', 'three .'], is_query=False, batch_size=2)
    assert model.encoder.batch_sizes == [2, 1]
    with pytest.raises(ValueError, match='no truncation'):
        model.encode_batch(['one two three'], is_query=True)


def test_teacher_makes_five_separate_calls_and_resumes_only_failed_question(tmp_path, monkeypatch):
    from colbert_experiments import labeling
    from remote_vllm_pool import OutputValidationError
    create_fixture(tmp_path / 'corpus')
    pair = read_jsonl(tmp_path / 'corpus/train_pairs.jsonl')[0]
    write_jsonl(tmp_path / 'one_pair.jsonl', [pair])
    prompts, fail_once = [], [True]

    class Tokenizer:
        chat_template = 'fake test template'
        def apply_chat_template(self, messages, **kwargs):
            return '\n'.join(m['content'] for m in messages)
        def encode(self, prompt, **kwargs):
            return prompt.split()

    class Models:
        async def list(self):
            return SimpleNamespace(data=[SimpleNamespace(id='test', model_dump=lambda: dict(id='test', max_model_len=4096))])

    client = SimpleNamespace(base_url='http://unused.test/v1', models=Models())

    def completion_factory(*args):
        async def complete(client, payload):
            prompts.append(payload['prompt'])
            if fail_once[0]:
                fail_once[0] = False
                return '', 'invalid output', dict(finish_reason='stop')
            return '', json.dumps(dict(answer='yes', evidence_status='supported', reason='Synthetic test.')), dict(finish_reason='stop')
        return complete

    class Session:
        def __init__(self, registry, work, **kwargs):
            self.work = work
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def submit(self, payload):
            try:
                return await self.work(client, payload)
            except OutputValidationError:
                return 'ERROR: exhausted test request'

    monkeypatch.setattr(labeling, 'load_tokenizer', lambda *a, **kw: Tokenizer())
    monkeypatch.setattr(labeling, 'resolve_sampling_defaults', lambda *a: {})
    monkeypatch.setattr(labeling, 'resolve_parser_name', lambda *a: 'test')
    monkeypatch.setattr(labeling, 'resolve_chat_template_kwargs', lambda *a: {})
    monkeypatch.setattr(labeling, 'make_completion_work_fn', completion_factory)
    monkeypatch.setattr(labeling, 'build_registry_from_args', lambda *a: None)
    monkeypatch.setattr(labeling, 'RequestSession', Session)
    args = SimpleNamespace(corpus=tmp_path / 'corpus', candidates=tmp_path / 'one_pair.jsonl',
        output=tmp_path / 'teacher', teacher_model='test', teacher_tokenizer='test', cache_dir=None,
        request_timeout=5, max_tokens=100, teacher_context=4096, max_attempts=1, pending_requests=2)
    with pytest.raises(RuntimeError, match='1 unresolved'):
        asyncio.run(labeling.label(args))
    assert len(prompts) == 5
    assert all(prompt.count('Question:') == 1 for prompt in prompts)
    assert len(read_jsonl(args.output / 'labels.jsonl')) == 4
    results = asyncio.run(labeling.label(args))
    assert len(prompts) == 6 and len(results) == 5
    assert {r['task'] for r in results} == set(TASKS)
    assert all(r['messages'] and r['rendered_prompt'] and 'reasoning' in r for r in results)
    assert len(list((args.output / 'responses').glob('*/*.json'))) == 5
    args.max_tokens += 1
    with pytest.raises(ValueError, match='identity changed'):
        asyncio.run(labeling.label(args))
    assert len(prompts) == 6
