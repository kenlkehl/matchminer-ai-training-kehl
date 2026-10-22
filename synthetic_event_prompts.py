"""Stage framing; clinical instructions remain editable in the 1a/1b scripts."""


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
        + 'Return only five short prose seeds separated by <new_patient>.'
    )
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': content}]


def patient_messages(*, system, criteria, patient_class, seed, instructions, consistency):
    population = ('could meet all supplied criteria at the end of the history'
                  if patient_class == 'positive' else
                  'narrowly misses at least one supplied criterion at the end of the history')
    content = (
        f'Generate the complete event chronology for exactly one patient who {population}.\n'
        'Eligibility criteria:\n' + criteria + '\n\n'
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
        'decisions, study entry, or enrollment in any event. '
        'Record ordinary clinical facts that establish the intended fit or mismatch.'
    )
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': content}]
