"""The teacher's label contracts: five match labels in one request per pair, boilerplate and per-criterion labels."""
import json

import pytest

from trialspace.contracts import judge_messages, parse_judgment
from trialspace.labeling import questions, validate_row
from trialspace.regression import MAX_SCORE, aggregate_pair, score_contract, validate_score
from trialspace.contracts import contract_hash

LABELS = ('reasonableness', 'cancer_type', 'burden', 'treatment_history', 'biomarker')


def answer(**yes):
    return json.dumps({label: {'answer': 'yes' if yes.get(label) else 'no', 'evidence_status': 'supported',
                               'reason': 'Fabricated.'} for label in LABELS})


def corpus():
    patients = [dict(id='p1', text='Key: metastatic lung cancer.', boilerplate='Dialysis.', split='train')]
    cohorts = [dict(id='c1', trial_id='T1', text='Key: metastatic NSCLC.', boilerplate='No dialysis.', split='train'),
               dict(id='c2', trial_id='T2', text='Key: sarcoma.', boilerplate='', split='train')]
    pairs = [dict(patient_id='p1', cohort_id=c, split='train') for c in ('c1', 'c2')]
    return patients, cohorts, pairs


def test_match_labels_are_one_request_per_pair_on_key_text_only():
    patients, cohorts, pairs = corpus()
    items = list(questions(pairs, patients, cohorts, task='match'))
    assert [(i['cohort_id'], i['task']) for i in items] == [('c1', 'match'), ('c2', 'match')]
    user = items[0]['messages'][1]['content']
    assert 'metastatic lung cancer' in user and 'metastatic NSCLC' in user
    assert 'Dialysis' not in user and 'dialysis' not in user
    assert all(f'"{label}"' in items[0]['messages'][0]['content'] for label in LABELS)


def test_boilerplate_labels_read_boilerplate_text_and_skip_cohorts_without_it():
    patients, cohorts, pairs = corpus()
    items = list(questions(pairs, patients, cohorts, task='boilerplate'))
    assert [i['cohort_id'] for i in items] == ['c1']
    user = items[0]['messages'][1]['content']
    assert 'Dialysis.' in user and 'No dialysis.' in user and 'metastatic' not in user
    assert parse_judgment('boilerplate', 'Criterion 1 is met.\nYes!') == dict(excluded=True)
    assert parse_judgment('boilerplate', 'No!') == dict(excluded=False)
    with pytest.raises(ValueError):
        parse_judgment('boilerplate', 'Probably excluded.')
    validate_row('boilerplate', dict(excluded=False))
    with pytest.raises(ValueError):
        validate_row('boilerplate', dict(excluded='no'))


def test_five_answers_give_one_validated_zero_to_six_score():
    patients, cohorts, pairs = corpus()
    item = next(questions(pairs, patients, cohorts, task='match'))
    row = dict(item, contract=contract_hash('match'), **parse_judgment('match', answer(
        reasonableness=True, cancer_type=True, biomarker=True)))
    validate_row('match', row)
    score = aggregate_pair(item, row)
    assert score['score'] == 4 and score['normalized_score'] == 4 / MAX_SCORE and MAX_SCORE == 6
    assert score['answers'] == dict(reasonableness='yes', cancer_type='yes', burden='no', treatment_history='no',
                                    biomarker='yes')
    assert score['question_keys'] == {'match': item['key']} and score['contract'] == score_contract()
    pmap, cmap = {p['id']: p for p in patients}, {c['id']: c for c in cohorts}
    validate_score(score, pmap, cmap, contract_hash('match'), score_contract())
    # Specificity without reasonableness scores zero, but every answer is kept for its head.
    zero = aggregate_pair(item, dict(row, **parse_judgment('match', answer(cancer_type=True, burden=True))))
    assert zero['score'] == 0 and zero['answers']['cancer_type'] == 'yes'
    with pytest.raises(ValueError):
        validate_score(dict(score, score=5), pmap, cmap, contract_hash('match'), score_contract())
    with pytest.raises(ValueError):
        parse_judgment('match', json.dumps({label: {'answer': 'yes'} for label in LABELS}))
    assert judge_messages('match', 'P', 'C')[1]['content'].endswith('COHORT KEY ELIGIBILITY CRITERIA\nC')


def test_boilerplate_labels_cover_each_unique_boilerplate_combination_of_every_round_once(tmp_path):
    from trialspace.data import read_jsonl, write_jsonl
    from trialspace_training import unique_boilerplate_pairs
    patients, cohorts, _ = corpus()
    cohorts.append(dict(id='c3', trial_id='T1', text='Key: other arm.', boilerplate='No dialysis.', split='train'))
    write_jsonl(tmp_path / 'corpus' / 'patients.jsonl', patients)
    write_jsonl(tmp_path / 'corpus' / 'cohorts.jsonl', cohorts)
    pair = lambda c: dict(patient_id='p1', cohort_id=c, split='train', sources=[])
    write_jsonl(tmp_path / 'round1.jsonl', [pair('c1'), pair('c2')])
    write_jsonl(tmp_path / 'round3.jsonl', [pair('c1'), pair('c3')])
    path = unique_boilerplate_pairs(tmp_path / 'corpus', [tmp_path / 'round3.jsonl', tmp_path / 'round1.jsonl'], tmp_path / 'out')
    # c1 and c3 share their boilerplate criteria; c2 has none: one combination to label.
    assert [p['cohort_id'] for p in read_jsonl(path)] == ['c1']
    # A later round skips the combinations an earlier round's directory labeled.
    again = unique_boilerplate_pairs(tmp_path / 'corpus', [tmp_path / 'round3.jsonl'], tmp_path / 'out2', skip=[tmp_path / 'out'])
    assert read_jsonl(again) == []


ITEMS = [dict(number=1, heading='Inclusion criteria:', text='Age >= 18.', criterion_type='inclusion', boilerplate=False),
         dict(number=2, heading='Exclusion criteria:', text='Dialysis.', criterion_type='exclusion', boilerplate=True),
         dict(number=3, heading='Exclusion criteria:', text='Unclassified.', criterion_type=None, boilerplate=False)]


def test_criteria_labels_are_one_request_per_tagged_line_of_sampled_pairs(tmp_path):
    from trialspace.data import read_jsonl, write_jsonl
    from trialspace_training import criteria_line_candidates
    patients, cohorts, _ = corpus()
    for cohort in cohorts:
        cohort['items'] = ITEMS
    write_jsonl(tmp_path / 'corpus' / 'patients.jsonl', patients)
    write_jsonl(tmp_path / 'corpus' / 'cohorts.jsonl', cohorts)
    write_jsonl(tmp_path / 'round1.jsonl', [dict(patient_id='p1', cohort_id=c, split='train', sources=[]) for c in ('c1', 'c2')])
    path = criteria_line_candidates(tmp_path / 'corpus', [tmp_path / 'round1.jsonl'], tmp_path / 'out', fraction=1, seed=42)
    rows = read_jsonl(path)
    # c1 and c2 have the same lines: each tagged line of the patient once; the untyped line is skipped.
    assert [(r['cohort_id'], r['number'], r['criterion_type']) for r in rows] == [('c1', 1, 'inclusion'), ('c1', 2, 'exclusion')]
    assert read_jsonl(criteria_line_candidates(tmp_path / 'corpus', [tmp_path / 'round1.jsonl'], tmp_path / 'out2',
                                               fraction=1, seed=42, skip=[tmp_path / 'out'])) == []
    item = list(questions(rows, patients, cohorts, task='criteria'))[1]
    user = item['messages'][0]['content']
    assert 'Key: metastatic lung cancer.\n\nOther relevant history:\nDialysis.' in user  # the complete summary
    assert '[2] Dialysis.' in user and user.split('OUTPUT FORMAT')[0].rstrip().endswith('[2] Exclusion criterion: Dialysis.')
    yes = json.dumps(dict(answer='yes', criterion_status='insufficient_information', reason='Not documented.'))
    fails = json.dumps(dict(answer='no', criterion_status='met', reason='On dialysis.'))
    assert parse_judgment('criteria', fails, 'exclusion')['criterion']['answer'] == 'no'
    assert parse_judgment('criteria', yes, 'inclusion')['criterion']['criterion_status'] == 'insufficient_information'
    with pytest.raises(ValueError):  # "no" (clearly fails) an exclusion line needs the exclusion to be met
        parse_judgment('criteria', json.dumps(dict(answer='no', criterion_status='not_met', reason='x')), 'exclusion')
    validate_row('criteria', dict(criterion=json.loads(fails), criterion_type='exclusion'))
    assert contract_hash('criteria') != contract_hash('match')
