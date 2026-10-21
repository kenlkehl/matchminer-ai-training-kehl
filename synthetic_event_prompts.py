"""Stage framing; clinical instructions remain editable in the 1a/1b scripts."""


def seed_messages(*, system, criteria, patient_class, diagnosis, diversity,
                  consistency, second_primary=''):
    population = ('could meet all of these criteria at the end of their histories'
                  if patient_class == 'positive' else
                  'would narrowly miss at least one of these criteria at the end of their histories')
    content = (
        f'Create five diverse patient seeds for patients with cancer who {population}.\n'
        'Eligibility criteria:\n' + criteria + '\n\n'
        'This is the seed stage only. Output exactly five patients, separated by '
        '<new_patient> on its own line. For each patient, output exactly one '
        '<demographics> line and one or more <diagnosis> lines. Output no other '
        'event types, commentary, headings, or closing tags.\n'
        'The demographics line must give the name, sex, and age at the beginning '
        'of the history. Each diagnosis line must begin with "At age N years," '
        'and describe a diagnosis that will anchor this patient\'s chronology. '
        'Include relevant cancer site, histology, stage, and biomarkers in the '
        'diagnosis lines. A prior unrelated cancer, if included, needs its own '
        'diagnosis line. Later treatment, progression, and report events will be '
        'generated in a separate call for each patient; do not generate them here.\n'
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
        + 'Return only the five tagged seeds.'
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
        + '\nThe assigned patient seed is:\n<patient_seed>\n' + seed
        + '\n</patient_seed>\n\n'
        'Expand only this patient. Copy the supplied demographics and diagnosis '
        'lines verbatim, exactly once each, placing diagnosis lines at their correct '
        'chronological positions among the new events. Do not change the name, '
        'sex, ages, cancer sites, histologies, or biomarker facts in the seed. '
        'The demographics age is the age at the beginning of the history. '
        'Later recurrences or metastases may be added consistently with the seed.\n'
        'Every non-demographic event must begin with "At age N years," using a '
        'numeric age. Output events in nondecreasing age order. Include the '
        'seed lines and the new events in one complete history of about 20–30 '
        'events. Do not output <new_patient>, another patient, headings, closing '
        'tags, or an explanation. Do not describe trial screening, eligibility '
        'decisions, study entry, or enrollment in any event. '
        'Record ordinary clinical facts that establish the intended fit or mismatch.'
    )
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': content}]
