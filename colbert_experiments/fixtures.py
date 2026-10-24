"""Minimal fabricated challenge cases, never production patients or trial records."""
from pathlib import Path

from .contracts import TASKS, contract_hash
from .data import write_json, write_jsonl


def create_fixture(output):
    output = Path(output)
    patients, cohorts, labels, candidates = [], [], [], []
    for split, cancer_a, cancer_b, gene_a, gene_b, therapy_a, therapy_b in [
        ('train', 'lung adenocarcinoma', 'colorectal adenocarcinoma', 'EGFR L858R', 'KRAS G12D', 'docetaxel', 'oxaliplatin'),
        ('val', 'cutaneous melanoma', 'gastrointestinal stromal tumor', 'BRAF V600E', 'KIT exon 11 mutation', 'dabrafenib', 'imatinib'),
    ]:
        prefix = 'FABRICATED_' + split.upper()
        definitions = [
            ('A', f'Adults with metastatic {cancer_a}. Progression must have occurred while receiving {therapy_a}. '
             f'The tumor must have BOTH {gene_a} AND a TP53 mutation. ECOG 0 or 1. Exclude brain metastases.'),
            ('B', f'Adults with metastatic {cancer_b}, with progression while receiving {therapy_b}. '
             f'Tumor must have {gene_b}. ECOG 0 or 1. Absolute neutrophil count must be at least 1500 cells/uL.'),
            ('C', 'Adults with any solid tumor, of any stage. No required prior systemic treatment or biomarker. ECOG 0 through 2.'),
            ('D', f'Adults with localized {cancer_a}, without distant metastases. ECOG 0 or 1. '
             'No prior-treatment or biomarker requirement.'),
        ]
        for name, text in definitions:
            cohorts.append(dict(id=f'{prefix}_{name}-0', trial_id=f'{prefix}_{name}', text=text, split=split))
        histories = [
            ('a_good', f'Metastatic {cancer_a}. Progression occurred while receiving {therapy_a}. '
             f'Tumor has {gene_a} and TP53 mutation. No brain metastases. ECOG 1.', 'A', True, True, True),
            ('a_unknown_biomarker', f'Metastatic {cancer_a}. Progression occurred while receiving {therapy_a}. '
             'Molecular testing results are unavailable. No brain metastases. ECOG 1.', 'A', True, None, True),
            ('a_timing_and', f'Metastatic {cancer_a}. Treatment with {therapy_a} stopped during response; '
             f'progression occurred six months AFTER treatment ended, not on therapy. Tumor has {gene_a}; '
             'TP53 testing was negative. No brain metastases. ECOG 1.', 'A', False, False, True),
            ('b_good', f'Metastatic {cancer_b}, progressed while receiving {therapy_b}. '
             f'Tumor has {gene_b}. ECOG 1. Absolute neutrophil count 1600 cells/uL.', 'B', True, True, True),
            ('b_threshold', f'Metastatic {cancer_b}, progressed while receiving {therapy_b}. '
             f'Tumor has {gene_b}. ECOG 1. Absolute neutrophil count 1400 cells/uL.', 'B', True, True, False),
            ('a_localized', f'Localized {cancer_a}, completely resected. No metastatic disease. '
             'No systemic cancer therapy. Biomarker testing explicitly negative for the mutations required by these cohorts. ECOG 1.',
             'D', False, False, True),
            ('a_brain', f'Metastatic {cancer_a}. Progression occurred while receiving {therapy_a}. '
             f'Tumor has {gene_a} and TP53 mutation. Active brain metastases. ECOG 1.', 'A', True, True, False),
        ]
        for name, summary, best, prior, biomarker, other in histories:
            p = dict(id=f'{prefix}_{name}', text='Cancer history:\nAge 60. ' + summary +
                     '\n\nOther relevant history:\nNo other relevant comorbidities.', split=split)
            patients.append(p)
            for c in [r for r in cohorts if r['split'] == split]:
                code = c['trial_id'].rsplit('_', 1)[-1]
                same_type = (best in ('A', 'D') and code in ('A', 'D')) or (best == code == 'B')
                same_burden = (best in ('A', 'B') and code in ('A', 'B')) or best == code == 'D'
                prior_match = prior and best == code and code in ('A', 'B')
                biomarker_match = biomarker is True and best == code and code in ('A', 'B')
                cancer_requirements_met_or_unknown = code == 'C' or (
                    best == code and (code == 'D' or prior_match and biomarker is not False))
                eligible = cancer_requirements_met_or_unknown and (code == 'C' or other)
                truth = dict(potentially_eligible=eligible, cancer_type=same_type, burden=same_burden,
                             prior_treatment=prior_match, biomarker=biomarker_match)
                pair = dict(patient_id=p['id'], cohort_id=c['id'], trial_id=c['trial_id'], split=split,
                            sources=[dict(source='fabricated_challenge')], tasks=list(TASKS))
                candidates.append(pair)
                for task in TASKS:
                    answer = truth[task]
                    unknown = biomarker is None and best == code and task in ('potentially_eligible', 'biomarker')
                    labels.append(dict(**pair, task=task, answer='yes' if answer else 'no',
                        evidence_status='insufficient_information' if unknown else 'supported' if answer else 'contradicted',
                        reason='Hand-specified fabricated challenge expectation; see fixture definition.',
                        contract=contract_hash(), label_source='fabricated_reference'))
    write_jsonl(output / 'patients.jsonl', patients)
    write_jsonl(output / 'cohorts.jsonl', cohorts)
    write_jsonl(output / 'reference_labels.jsonl', labels)
    write_jsonl(output / 'all_pairs.jsonl', candidates)
    write_jsonl(output / 'train_pairs.jsonl', [r for r in candidates if r['split'] == 'train'])
    write_json(output / 'corpus.json', dict(source='fabricated challenge fixture', patients=len(patients), cohorts=len(cohorts)))
