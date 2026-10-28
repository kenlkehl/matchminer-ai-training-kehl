"""EmbeddingGemma formatting around the shared clinical retrieval instruction."""
from matchminer_ai.embedding.colbert_contract import INSTRUCTION

SYMMETRIC_PREFIX = 'task: sentence similarity | query: '
PROFILES = ('sentence_similarity', 'colbert_matching')


def model_prompts(profile='sentence_similarity'):
    if profile not in PROFILES:
        raise ValueError('Unknown TrialSpace prompt profile')
    prefix = SYMMETRIC_PREFIX
    if profile == 'colbert_matching':
        prefix += INSTRUCTION + '\n\n'
    prompts = {name: prefix for name in ('SentenceSimilarity', 'query', 'document')}
    if profile == 'colbert_matching':
        prompts.update(patient_summary=prefix + 'Patient history:\n',
                       this_space=prefix + 'Cohort eligibility criteria:\n')
    return prompts


def training_prompts(profile='sentence_similarity'):
    prompts = model_prompts(profile)
    return ({name: prompts[name] for name in ('patient_summary', 'this_space')}
            if profile == 'colbert_matching' else SYMMETRIC_PREFIX)
