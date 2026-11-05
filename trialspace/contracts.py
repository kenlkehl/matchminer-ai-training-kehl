"""The teacher's label contracts, both owned by the inference package.

- ``match``: the five match labels of a patient–cohort pair (reasonableness and the cancer type,
  burden, treatment history and biomarker specificity answers), asked in ONE prompt per pair
  (``matchminer_ai.matching.match_labels``; user instruction 2026-11-04). The patient text is the
  summary's key section and the cohort text its key criteria. TrialSpace and the multi-head
  TrialChecker train on these labels.
- ``boilerplate``: whether the patient clearly fails any of the cohort's boilerplate criteria, from
  the summary's boilerplate section (``llm_exclusion_criteria.user.txt``, the BoilerplateChecker
  prompt of ``exclusion_criteria_check_with_llm``). BoilerplateChecker trains on these labels.
"""
import hashlib
import json

from matchminer_ai.embedding.trialspace_contract import DIRECTIONS
from matchminer_ai.llm.prompts import load_prompt_text
from matchminer_ai.matching.match_labels import (
    LABELS, MATCH_LABELS_PROMPT, build_match_label_messages, match_labels_contract, parse_match_labels,
)

VERSION = 'trialspace-teacher-contracts-v1'
TASKS = ('match', 'boilerplate')
BOILERPLATE_PROMPT = 'llm_exclusion_criteria.user.txt'


def prompt_sources():
    return {name: load_prompt_text(name) for name in (MATCH_LABELS_PROMPT, BOILERPLATE_PROMPT)}


def contract_hash(task='match', sources=None):
    sources = prompt_sources() if sources is None else sources
    detail = match_labels_contract() if task == 'match' else dict(prompt=BOILERPLATE_PROMPT, answer='Yes!/No!')
    return hashlib.sha256(json.dumps(dict(version=VERSION, task=task, detail=detail, prompts=sources),
                                     sort_keys=True).encode()).hexdigest()


def judge_messages(task, patient, cohort, *, sources=None):
    """Messages for one pair: key texts for ``match``, boilerplate texts for ``boilerplate``."""
    sources = prompt_sources() if sources is None else sources
    if task == 'match':
        return build_match_label_messages(patient, cohort, sources[MATCH_LABELS_PROMPT])
    if task == 'boilerplate':
        return [dict(role='system', content=''),
                dict(role='user', content=sources[BOILERPLATE_PROMPT].format(
                    patient_boilerplate=patient, trial_boilerplate=cohort))]
    raise ValueError('Unknown task')


def parse_boilerplate(answer):
    """``{"excluded": bool}`` from the final Yes!/No!; anything else is invalid, never a guess."""
    tail = answer.strip()[-12:]
    if tail.endswith('Yes!'):
        return dict(excluded=True)
    if tail.endswith('No!'):
        return dict(excluded=False)
    raise ValueError('The answer must end with "Yes!" or "No!"')


def parse_judgment(task, answer):
    if task == 'match':
        return dict(labels=parse_match_labels(answer))
    if task == 'boilerplate':
        return parse_boilerplate(answer)
    raise ValueError('Unknown task')


__all__ = ['BOILERPLATE_PROMPT', 'DIRECTIONS', 'LABELS', 'TASKS', 'VERSION', 'contract_hash', 'judge_messages',
           'parse_boilerplate', 'parse_judgment', 'prompt_sources']
