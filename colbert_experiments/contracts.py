"""Versioned clinical tasks and short query instructions for the experiment."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
VERSION = 'cohort-colbert-five-tasks-v2'
TASKS = {
    'potentially_eligible': 'Find matches with no clear evidence of ineligibility across all eligibility criteria and patient history.',
    'cancer_type': 'Find matches with a specific matching cancer type and histology.',
    'burden': 'Find matches with a specific matching cancer stage, burden, and disease status.',
    'prior_treatment': 'Find matches satisfying a specific prior-treatment and response requirement.',
    'biomarker': 'Find matches requiring a biomarker documented in the patient or tumor.',
}
DIRECTIONS = ('patient_to_cohort', 'cohort_to_patient')


def query_text(task, direction, text):
    if direction not in DIRECTIONS:
        raise ValueError('Unknown retrieval direction')
    subject = 'Patient history' if direction == 'patient_to_cohort' else 'Cohort eligibility criteria'
    return f'{TASKS[task]}\n\n{subject}:\n{text}'


def prompt_sources():
    return {path.name: path.read_text() for path in sorted((HERE / 'prompts').glob('*.txt'))}


def contract_hash(sources=None):
    return hashlib.sha256(json.dumps(dict(version=VERSION, tasks=TASKS, prompts=prompt_sources() if sources is None else sources),
                                    sort_keys=True).encode()).hexdigest()


def judge_messages(task, patient, cohort, *, sources=None):
    if task not in TASKS:
        raise ValueError('Unknown task')
    sources = prompt_sources() if sources is None else sources
    return [dict(role='system', content=sources['shared.txt'] + '\n\n' + sources[task + '.txt']),
            dict(role='user', content=f'PATIENT HISTORY\n{patient}\n\nCOMPLETE COHORT ELIGIBILITY CRITERIA\n{cohort}')]


def parse_judgment(answer):
    answer = answer.strip()
    if answer.startswith('```') and answer.endswith('```'):
        answer = answer.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    result = json.loads(answer)
    if not isinstance(result, dict) or set(result) != {'answer', 'evidence_status', 'reason'}:
        raise ValueError('Return exactly answer, evidence_status, and reason')
    if result['answer'] not in ('yes', 'no'):
        raise ValueError('answer must be yes or no')
    if result['evidence_status'] not in ('supported', 'contradicted', 'insufficient_information', 'not_applicable'):
        raise ValueError('Invalid evidence_status')
    if not isinstance(result['reason'], str) or not result['reason'].strip():
        raise ValueError('A brief reason is required')
    return result
