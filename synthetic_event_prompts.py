"""Stage framing; clinical instructions remain editable in the 1a/1b scripts."""


SEED_FORMAT_EXAMPLE = (
    'Marisol Vega, female, was diagnosed with sigmoid colon adenocarcinoma at '
    '54 years and 19 weeks: site sigmoid colon (C18.7), histology adenocarcinoma, '
    'NOS (8140/3), grade 2, tumor size 2.8 cm, pT2 pN0 cM0, stage I, summary '
    'stage localized, 18 regional lymph nodes examined with none positive, '
    'lymphovascular and perineural invasion absent, margins negative; '
    'microsatellite stable, KRAS p.G12D mutation detected, BRAF wild-type.'
)


def seed_messages(*, system, criteria, patient_class, diagnosis, diversity,
                  consistency, second_primary=''):
    population = ('could meet all of these criteria at the end of their histories'
                  if patient_class == 'positive' else
                  'would narrowly miss at least one of these criteria at the end of their histories')
    content = (
        f'Create five diverse patient seeds for patients with cancer who {population}.\n'
        'Eligibility criteria:\n' + criteria + '\n\n'
        'This is the seed stage only. Output exactly five patients as five short '
        'prose paragraphs, separated by <new_patient> on its own line. Each '
        'paragraph must give the patient\'s name, sex, diagnoses, and age at each '
        'diagnosis. Use no event tags, headings, numbering, or commentary.\n'
        'Express each age at diagnosis as whole years plus additional weeks, '
        'for example "50 years and 21 weeks". Use nonnegative integer years and '
        'integer weeks from 0 through 51; include both units even when weeks is '
        'zero. Give ages only for diagnoses; the later event-generation call '
        'will determine the age at the beginning of the complete history. '
        'Include relevant cancer site, histology, stage, and biomarkers in the '
        'prose. If a prior unrelated cancer is included, describe its diagnosis '
        'and its own age at diagnosis. Keep each seed brief and limited to these '
        'facts. Treatment, progression, and report events will be generated in a '
        'separate call for each patient; do not generate them here.\n'
        + diagnosis + diversity + consistency
    )
    if second_primary:
        content += '\n' + second_primary + '\n'
    content += (
        '\nKeep the five identities and their clinical facts distinct. Biological '
        'and anatomical consistency take precedence over diversity. '
        + ('For these positive examples, all eligibility criteria take precedence '
           'over diversity requests: omit a prior unrelated cancer when it would '
           'violate the supplied exclusions.\n' if patient_class == 'positive' else
           'These are near-miss examples; preserve a plausible clinical history '
           'even when choosing a characteristic that does not meet the criteria.\n')
        + '\nThe following is one hypothetical prose seed, provided only to '
        'illustrate formatting. Do not copy this example into your output or '
        'reuse its patient identity. Generate five new patients. '
        'It is not an additional patient or a template for the required cancer '
        'type or biomarkers; use the supplied criteria for those.\n'
        '(Beginning of seed prose format example)\n' + SEED_FORMAT_EXAMPLE
        + '\n(End of seed prose format example)\n'
        'Return only five new short prose seeds separated by <new_patient>.'
    )
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': content}]


def patient_messages(*, system, seed, instructions, consistency):
    content = (
        'Generate the complete event chronology for exactly one patient based on the assigned prose seed.\n'
        + instructions + consistency
        + '\nThe assigned patient prose seed is:\n<patient_seed>\n' + seed
        + '\n</patient_seed>\n\n'
        'Expand only this patient. Use the prose seed as the factual basis for '
        'the complete tagged event list. Generate the wording of every event, '
        'including demographics and diagnoses. Preserve the seed\'s name, sex, '
        'diagnoses, ages at diagnosis, cancer sites, histologies, and biomarker '
        'facts. Include each seed diagnosis as a <diagnosis> event at its assigned '
        'age, in its correct chronological position. '
        'Begin the output with exactly one <demographics> line giving name, sex, '
        'and age at the earliest event in the generated history. Keep this line '
        'first. Express that age as whole years plus additional weeks, for '
        'example "50 years and 21 weeks". '
        'The first clinical event does not have to be a diagnosis. Include '
        'clinically relevant symptoms, imaging, biopsy, or surgery before the '
        'diagnosis when appropriate, while preserving the seed diagnosis age. '
        'Place the workup establishing a diagnosis before its diagnosis entry, '
        'including events at the same age in causal order. Report findings '
        'only after the examination, procedure, or test that establishes them; '
        'for example, do not report surgical margins, lymph-node counts, or '
        'pathologic staging before the relevant surgery and pathology. '
        'Later recurrences or metastases may be added consistently with the seed.\n'
        'Start each event line with its event tag. Immediately after the tag, '
        'every non-demographic event must begin with "At age N years and W weeks," '
        'using nonnegative integer years and integer weeks from 0 through 51. '
        'Include both units even when weeks is zero. Use weeks to distinguish '
        'events within the same year; do not round every event to a whole year. '
        'Output events in nondecreasing age order, comparing both years and weeks. Include '
        'all diagnoses and other events in one complete history of about 20–30 '
        'events. Do not output <new_patient>, another patient, headings, closing '
        'tags, or an explanation. Do not describe trial screening, eligibility '
        'decisions, study entry, or enrollment in any event.'
    )
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': content}]
