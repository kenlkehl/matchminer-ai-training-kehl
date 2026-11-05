"""EmbeddingGemma formatting around the shared clinical retrieval instruction.

The ``trialspace`` profile (production) gives patients and cohorts their own prompts, saved with
the model under the names inference uses (matchminer_ai.embedding.trialspace_contract); both carry
the shared INSTRUCTION.
"""
from matchminer_ai.embedding.trialspace_contract import COHORT_PROMPT, INSTRUCTION, PATIENT_PROMPT

SYMMETRIC_PREFIX = 'task: sentence similarity | query: '
PROFILES = ('sentence_similarity', 'trialspace')


def model_prompts(profile='sentence_similarity'):
    if profile not in PROFILES:
        raise ValueError('Unknown TrialSpace prompt profile')
    prefix = SYMMETRIC_PREFIX
    if profile == 'trialspace':
        prefix += INSTRUCTION + '\n\n'
    prompts = {name: prefix for name in ('SentenceSimilarity', 'query', 'document')}
    if profile == 'trialspace':
        prompts.update({PATIENT_PROMPT: prefix + 'Patient history:\n',
                        COHORT_PROMPT: prefix + 'Cohort eligibility criteria:\n'})
    return prompts


def training_prompts(profile='sentence_similarity'):
    prompts = model_prompts(profile)
    return ({name: prompts[name] for name in (PATIENT_PROMPT, COHORT_PROMPT)}
            if profile == 'trialspace' else SYMMETRIC_PREFIX)
