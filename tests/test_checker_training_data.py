"""Checker training examples: the inference templates on key or boilerplate text, and their targets."""
import json

from trialspace.data import write_jsonl
import train_checkers


def test_trial_checker_has_five_targets_on_key_text_and_boilerplate_checker_its_own(tmp_path):
    corpus = tmp_path / 'corpus'
    write_jsonl(corpus / 'patients.jsonl', [dict(id='p1', text='Lung key.', boilerplate='Dialysis.', split='train')])
    write_jsonl(corpus / 'cohorts.jsonl', [dict(id='c1', trial_id='T1', text='NSCLC key.', boilerplate='No dialysis.',
                                                split='train')])
    answers = dict(reasonableness='yes', cancer_type='yes', burden='no', treatment_history='no', biomarker='yes')
    for round_number, burden in ((1, 'no'), (2, 'yes')):  # a later round wins a conflict
        write_jsonl(tmp_path / f'r{round_number}' / 'scores.jsonl',
                    [dict(patient_id='p1', cohort_id='c1', answers=dict(answers, burden=burden))])
    texts, targets, counts = train_checkers.trial_examples(corpus, [tmp_path / 'r1', tmp_path / 'r2'])
    assert texts == ['NSCLC key.\nNow here is the patient summary:Lung key.']
    assert targets == [[1.0, 1.0, 1.0, 0.0, 1.0]] and counts['conflicting_rounds'] == 1
    write_jsonl(tmp_path / 'b' / 'labels.jsonl', [dict(patient_id='p1', cohort_id='c1', task='boilerplate', excluded=True)])
    texts, targets, counts = train_checkers.boilerplate_examples(corpus, [tmp_path / 'b'])
    assert texts == ['Patient history: Dialysis.\nTrial exclusions:No dialysis.'] and targets == [1]
