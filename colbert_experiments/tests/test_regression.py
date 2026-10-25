import asyncio
import json
from types import SimpleNamespace

import pytest
import torch

from colbert_experiments import regression as r
from colbert_experiments.contracts import TASKS, contract_hash, judge_messages
from colbert_experiments.data import digest, load_corpus, read_jsonl, write_json, write_jsonl
from colbert_experiments.fixtures import create_fixture
from colbert_experiments.labeling import label_key, questions


def judgments(pair, eligible='yes'):
    return [dict(**pair, task=task, key=label_key(dict(**pair, task=task)),
                 answer=eligible if task == r.GATE else 'yes', evidence_status='supported',
                 reason='Fabricated evidence.', contract=contract_hash()) for task in TASKS]


def test_gating_skips_four_calls_and_does_not_impute_failed_judgments():
    pair = dict(patient_id='p', cohort_id='c', split='train')
    calls = []
    async def ask(item):
        calls.append(item['task'])
        return item
    rows = judgments(pair, 'no')
    selected = asyncio.run(r.gated_judgments(rows, ask))
    score = r.aggregate_pair(pair, selected)
    assert calls == [r.GATE] and score['score'] == 0
    assert all(score['answers'][task] is None for task in r.SPECIFICITY)
    assert r.aggregate_pair(pair, rows) == score  # Discard old specificity answers after no.
    calls.clear()
    rows = judgments(pair)
    assert len(asyncio.run(r.gated_judgments(rows, ask))) == 5
    assert calls == list(TASKS)
    assert r.aggregate_pair(pair, rows)['score'] == 5
    rows[-1]['answer'] = 'no'
    assert r.aggregate_pair(pair, rows)['normalized_score'] == .8
    with pytest.raises(ValueError, match='Missing required'):
        r.aggregate_pair(pair, rows[:1])
    async def failed(item):
        calls.append(item['task'])
        return None
    calls.clear()
    assert asyncio.run(r.gated_judgments(rows, failed)) is None
    assert calls == [r.GATE]


def test_numeric_loss_and_ranking_metrics_are_distinct():
    scores = torch.tensor([.1, .7, .9], requires_grad=True)
    loss = r.regression_loss(scores, [0, 3, 5])
    assert loss.item() == pytest.approx(.01)
    loss.backward()
    assert scores.grad[0] > 0 and scores.grad[1] > 0 and scores.grad[2] < 0
    linked = torch.tensor([.4, .6], requires_grad=True)
    r.regression_loss(linked, [0, 5], link='sigmoid').backward()
    assert linked.grad[0] > 0 and linked.grad[1] < 0
    perfect = r.score_statistics(torch.tensor([0., .6, 1.]), [0, 3, 5])
    assert perfect['loss'] < 1e-12 and perfect['pairwise_accuracy'] == 1
    assert perfect['ndcg_at_10'] == pytest.approx(1)
    tied = r.score_statistics(torch.tensor([.5, .5]), [0, 5])
    assert tied['pairwise_accuracy'] == .5 and tied['ndcg_at_10'] < 1
    constant = r.score_statistics(torch.tensor([.1, .2]), [0, 0])
    assert constant['pairwise_accuracy'] is None and constant['ndcg_at_10'] is None
    with pytest.raises(ValueError, match='aligned targets'):
        r.regression_loss(scores, [0, 6, 5])


def test_group_contract_preserves_both_directions_and_constant_groups():
    patients = [dict(id='p', text='Full patient history', split='train')]
    cohorts = [dict(id='c', text='Full eligibility criteria', split='train')]
    pair = dict(patient_id='p', cohort_id='c', split='train')
    score = r.aggregate_pair(pair, judgments(pair, 'no'))
    groups, report = r.build_groups(patients, cohorts, [score])
    assert len(groups) == 2 and report['groups_with_score_variation'] == 0
    assert all(g['labels'] == [0] for g in groups)
    assert all(r.INSTRUCTION in g['query'] for g in groups)
    for update in [dict(score=True), dict(score=5), dict(split='val'), dict(contract='old')]:
        with pytest.raises(ValueError):
            r.build_groups(patients, cohorts, [{**score, **update}])
    assert r.make_probes(groups) == r.make_probes(groups)


def test_verified_import_checks_full_text_and_records_reuse(tmp_path):
    corpus, teacher, output = tmp_path / 'corpus', tmp_path / 'teacher', tmp_path / 'regression'
    create_fixture(corpus)
    patients, cohorts = load_corpus(corpus)
    pairs = read_jsonl(corpus / 'all_pairs.jsonl')
    labels = read_jsonl(corpus / 'reference_labels.jsonl')
    pmap, cmap = {p['id']: p for p in patients}, {c['id']: c for c in cohorts}
    for row in labels:
        row['key'] = label_key(row)
        row['prompt_sha256'] = digest(judge_messages(row['task'], pmap[row['patient_id']]['text'], cmap[row['cohort_id']]['text']))
    write_json(teacher / 'labeling_manifest.json', dict(contract=contract_hash(),
        patients=digest(patients), cohorts=digest(cohorts), pairs=digest(pairs)))
    write_jsonl(teacher / 'labels.jsonl', labels)
    report = r.import_verified_judgments(corpus, corpus / 'all_pairs.jsonl', teacher, output)
    assert report['pairs'] == len(pairs) and report['skipped_specificity_calls'] > 0
    provenance = json.loads((output / 'import_manifest.json').read_text())
    assert provenance['new_llm_calls'] == 0
    assert provenance['reused_question_answers'] == report['required_question_answers']
    groups, _ = r.build_groups(patients, cohorts, read_jsonl(output / 'scores.jsonl'))
    assert groups
    assert r.import_verified_judgments(corpus, corpus / 'all_pairs.jsonl', teacher, output) == report
    labels[0]['prompt_sha256'] = 'altered-text'
    write_jsonl(teacher / 'labels.jsonl', labels)
    with pytest.raises(ValueError, match='complete clinical text'):
        r.import_verified_judgments(corpus, corpus / 'all_pairs.jsonl', teacher, output)


def test_gated_labeler_resumes_failed_specificity_without_reasking_gate(tmp_path, monkeypatch):
    from colbert_experiments import labeling
    create_fixture(tmp_path / 'corpus')
    pairs = read_jsonl(tmp_path / 'corpus/train_pairs.jsonl')[:2]
    write_jsonl(tmp_path / 'pairs.jsonl', pairs)
    patients, cohorts = load_corpus(tmp_path / 'corpus')
    by_key = {item['key']: item for item in questions(pairs, patients, cohorts)}
    calls, fail_once = [], [True]

    class Tokenizer:
        chat_template = 'test'
        def apply_chat_template(self, messages, **kwargs):
            return '\n'.join(m['content'] for m in messages)
        def encode(self, prompt, **kwargs):
            return prompt.split()

    class Session:
        def __init__(self, *args, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def submit(self, payload):
            assert payload['prompt'].count('Question:') == 1
            item = by_key[payload['key']]
            calls.append((item['cohort_id'], item['task']))
            if item['task'] == 'biomarker' and fail_once[0]:
                fail_once[0] = False
                return 'ERROR: failed request'
            return dict(answer='no' if item['cohort_id'] == pairs[1]['cohort_id'] else 'yes',
                        evidence_status='supported', reason='Fabricated response.')

    monkeypatch.setattr(labeling, 'load_tokenizer', lambda *a, **kw: Tokenizer())
    monkeypatch.setattr(labeling, 'resolve_sampling_defaults', lambda *a: {})
    monkeypatch.setattr(labeling, 'resolve_parser_name', lambda *a: 'test')
    monkeypatch.setattr(labeling, 'resolve_chat_template_kwargs', lambda *a: {})
    monkeypatch.setattr(labeling, 'make_completion_work_fn', lambda *a: None)
    monkeypatch.setattr(labeling, 'build_registry_from_args', lambda *a: None)
    monkeypatch.setattr(labeling, 'RequestSession', Session)
    args = SimpleNamespace(corpus=tmp_path / 'corpus', candidates=tmp_path / 'pairs.jsonl',
        output=tmp_path / 'teacher', teacher_model='test', teacher_tokenizer='test', cache_dir=None,
        request_timeout=5, max_tokens=100, teacher_context=4096, max_attempts=1, pending_requests=2,
        gated_regression=True)
    with pytest.raises(RuntimeError, match='1 unresolved'):
        asyncio.run(labeling.label(args))
    assert len(calls) == 6
    partial = read_jsonl(args.output / 'scores.jsonl')
    assert len(partial) == 1 and partial[0]['score'] == 0
    asyncio.run(labeling.label(args))
    assert len(calls) == 7 and calls[-1][1] == 'biomarker'
    assert sorted(s['score'] for s in read_jsonl(args.output / 'scores.jsonl')) == [0, 5]
    progress = json.loads((args.output / 'status.json').read_text())
    assert progress['scored_pairs'] == 2 and progress['total_pairs'] == 2
    assert progress['completed'] == 6
    assert progress['score_distribution']['0'] == progress['score_distribution']['5'] == 1
    assert progress['updated_at'] >= progress['started_at']
    args.gated_regression = False
    with pytest.raises(ValueError, match='identity changed'):
        asyncio.run(labeling.label(args))


def test_regression_round_wires_gating_and_numeric_labels(tmp_path, monkeypatch):
    import sys
    from colbert_experiments import run_round
    calls = []
    monkeypatch.setattr(sys, 'argv', ['run_round', '--corpus', str(tmp_path / 'corpus'),
        '--output', str(tmp_path / 'round'), '--gpus', '2', '--teacher-url', 'http://unused.test/v1',
        '--teacher-model', 'test', '--teacher-tokenizer', 'test', '--objective', 'gated_regression'])
    monkeypatch.setattr(run_round, 'busy_training_gpus', lambda g: [])
    monkeypatch.setattr(run_round.subprocess, 'run', lambda command, **kw: calls.append(command))
    run_round.main()
    mine, label, train = calls
    assert mine[mine.index('--objective') + 1] == 'gated_regression'
    assert '--gated-regression' in label
    assert train[train.index('--labels') + 1].endswith('teacher/scores.jsonl')
    assert train[train.index('--objective') + 1] == 'gated_regression'


def test_regression_mining_uses_aggregate_instruction_and_rejects_wrong_contract(tmp_path):
    from colbert_experiments.retrieval import mine
    create_fixture(tmp_path / 'corpus')
    class Model:
        device = torch.device('cpu')
        folder = tmp_path / 'model'
        def fingerprint(self):
            return dict(test=True)
        def encode_batch(self, texts, is_query):
            if is_query:
                assert len(texts) == 1 and texts[0].startswith(r.INSTRUCTION)
            return [torch.ones(2, 2) for text in texts]
    model = Model()
    write_json(model.folder / 'regression_scoring.json', dict(contract=r.score_contract()))
    with pytest.raises(ValueError, match='matching gated_regression'):
        mine(model, tmp_path / 'corpus', tmp_path / 'wrong')
    pairs = mine(model, tmp_path / 'corpus', tmp_path / 'mined', objective='gated_regression',
                 splits=['train'], cohort_query_limit=1, candidate_limit=2, top_k=1, lexical_k=1, random_k=1)
    assert pairs
    rankings = read_jsonl(tmp_path / 'mined/rankings.jsonl')
    assert {row['task'] for row in rankings} == {'gated_score'}
    assert {row['direction'] for row in rankings} == set(r.DIRECTIONS)
    assert mine(model, tmp_path / 'corpus', tmp_path / 'mined', objective='gated_regression',
                splits=['train'], cohort_query_limit=1, candidate_limit=2, top_k=1, lexical_k=1, random_k=1) == pairs


def test_fixed_probes_preserve_ordering_and_report_calibration_separately():
    groups = [dict(direction=direction, query_id='q', query='full query',
                   documents=[dict(id=str(i), text='full criteria') for i in range(12)], labels=[0] * 11 + [5])
              for direction in r.DIRECTIONS]
    probes = r.make_probes(groups, documents_per_class=4)
    assert all(set(p['labels']) == {0, 5} for p in probes)
    class Model(torch.nn.Module):
        device = torch.device('cpu')
        def forward(self, query, documents):
            assert query == 'full query' and len(documents) == 4
            return torch.full((4,), .5)
    model = Model()
    report, rows = r.evaluate_probes(model, probes)
    assert model.training
    assert report['groups'] == report['ordering_groups'] == report['ndcg_groups'] == 2
    assert report['pairwise_accuracy'] == .5 and report['rmse'] == 2.5
    assert report['ordering_comparisons'] == 6
    baseline = r.constant_baselines(groups, {'val': probes})
    assert baseline['train_mean_score'] == pytest.approx(5 / 12)
    assert baseline['zero']['val']['pairwise_accuracy'] == .5
    assert baseline['zero']['val']['loss'] == .25
    assert baseline['train_mean']['val']['loss'] < baseline['zero']['val']['loss']
