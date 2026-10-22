"""Positive-patient prompt definitions imported by generate_synthetic_data.py.

Edit the clinical instruction blocks here. Generation, validation, and
checkpointing belong to the production driver, not this prompt module.
"""
import random


SYSTEM_PROMPT = (
    "I'm a medical oncologist and data scientist. Your job is to help me create synthetic clinical data "
    "for cancer research."
)


DIAGNOSIS_INSTRUCTIONS = (
    'Diagnosis events should include TNM stage, summary stage, site description and code, histology description '
    'and code, and all relevant site-specific data elements that a cancer registrar would annotate.\n'
)


DIVERSITY_INSTRUCTIONS = (
    'To ensure diversity in the generated data, vary patient age, gender, name, stage at diagnosis, biomarkers, '
    'treatment approaches, and disease course (e.g., stable disease, progression, remission, recurrence).\n'
)


CONSISTENCY_INSTRUCTIONS = (
    'VERY IMPORTANT: All events MUST be in chronological order and must be internally consistent in that context. '
    'For example, an event from a younger age must not follow an event from an older age, and an organ that has '
    'been removed must not later be described as present.\n'
    'CRITICAL: Do not add findings, including claims that disease is absent, for organs that would not have been '
    'visualized by the specified scan type and field of view; for example, a CT of the chest, abdomen, and '
    'pelvis cannot establish the presence or absence of brain metastases.\n'
    'CRITICAL: Do not mention screening or enrollment on the trial itself anywhere in your chronology!! That would '
    'contaminate the synthetic data and render them useless, since we will be using the synthetic data to derive '
    'clinical history models to match patients to clinical trials on which they have not yet enrolled!\n'
)


EVENT_INSTRUCTIONS = (
    'Use everything you know about cancer and clinical oncology.\n'
    'The "Prior treatment requirements" and "Biomarker requirements" fields each include both positive and '
    'negative conditions. Preserve each clause, its negation, AND/OR logic, and exceptions when interpreting '
    'eligibility.\n'
    'Types of events might include a diagnosis as recorded in a cancer registry; initiation of a systemic therapy; '
    'a surgery; a radiation treatment; an adverse event; a clinical progress note; an imaging report; a pathology '
    'report; and a next-generation sequencing (NGS) report. For progress notes, imaging reports, pathology '
    'reports, and NGS reports, include findings documented in the report in the description of the event. \n'
    'NGS reports should be very detailed; they should include both any key actionable alterations if present, and '
    'comutations/fusions/copy number alterations; most reports should describe alterations in many genes, even '
    'though only some of those will be clinically relevant. \n'
    'CRITICAL: Genomic findings should make sense based on known mutation and comutation patterns. For example, '
    'remember that EGFR mutant lung cancers almost never have KRAS co-mutations. \n'
    'There should be one event per line of text in your output, and each event should be formatted as a sentence. \n'
    'Most patients will have many events along their disease trajectories (20-30). \n'
    'Tag each event with an event type at the beginning of the line. Acceptable event types include '
    '<demographics>, <diagnosis>, <systemic>, <surgery>, <radiation>, <adverse_event>, <clinical_note>, '
    '<imaging_report>, <pathology_report>, and <ngs_report>.\n'
    'Each event should correspond only to one point in time, and each report should correspond only to one report '
    'that could have been written at that time.\n'
    + DIAGNOSIS_INSTRUCTIONS
    + (
        'Imaging report events must describe only one radiographic study and should specify the type of study. Imaging '
        'report events must also indicate whether cancer was present on the scan; if so, whether it was responding, '
        'progressing, or neither; and what metastatic sites were involved.\n'
        'Oncologist note events must indicate whether cancer was present at the time; and if so, whether it was '
        'responding, progressing, or neither. \n'
        'NGS report events should indicate the diagnosis, specimen site, and genomic findings.\n'
        'Here is an example of what your output should look like. This is hypothetical, just to illustrate the '
        "formatting. Don't use this text in your output.\n"
        'Do adhere closely to the formatting.\n'
        '(Beginning of example)\n'
        '<demographics> The patient is a male named John Smith, aged 70 years and 0 weeks at the beginning of the history.\n'
        '<diagnosis>At age 70 years and 0 weeks, the patient had a diagnosis of stage 1E (Best AJCC) LYMPHOMA, MALIG, DIFFUSE, '
        'NOS, (MALIGNANT, PRIMARY) of the BASAL GANGLIA. Relevant biomarkers included B Symptoms: 0: No B symptoms '
        '(asymptomatic) ; Classified as A by physician when asymptomatic. Other relevant diagnostic information '
        'included Confirmation: POSITIVE HISTOLOGY; Tumor Size: 15 mm.\n'
        '<pathology_report>At age 70 years and 0 weeks, the patient had a pathology result of type ANATOMIC PATHOLOGY.\n'
        '<imaging_report>At age 70 years and 0 weeks, the patient had a CT HEAD, which showed no cancer.\n'
        '<imaging_report>At age 70 years and 0 weeks, the patient had a CT HEAD, which showed no cancer.\n'
        '<imaging_report>At age 70 years and 0 weeks, the patient had a NM PET CT SCALP TO TOES, which showed no cancer.\n'
        '<imaging_report>At age 70 years and 0 weeks, the patient had a MRI LUMBAR SPINE, which showed no cancer.\n'
        '<imaging_report>At age 70 years and 0 weeks, the patient had a MRI THORACIC SPINE, which showed no cancer.\n'
        '<imaging_report>At age 70 years and 0 weeks, the patient had a MRI CERVICAL SPINE, which showed no cancer.\n'
        '<systemic>At age 70 years and 1 week, the patient received curative-intent methotrexate/temozolomide/rituximab.\n'
        '<pathology_report>At age 70 years and 2 weeks, the patient had a pathology result of type CLINICAL ONCOPANEL.\n'
        '<clinical_note>At age 70 years and 2 weeks, the patient had an oncologist office assessment, which showed cancer.\n'
        '<clinical_note>At age 70 years and 4 weeks, the patient had an oncologist office assessment, which showed cancer. There was '
        'response to therapy.\n'
        '<clinical_note>At age 70 years and 6 weeks, the patient had an oncologist office assessment, which showed cancer.\n'
        '<ngs_report>At age 70 years and 6 weeks, the patient had next generation sequencing performed for diffuse large b-cell '
        'lymphoma based on a specimen obtained from a unspecified site (cns/brain), which showed a CD79B p.Y196F '
        'mutation, a CDKN2A loss, a CDKN2A p.W110* mutation, a ETV6 p.X11_splice mutation, a MYD88 p.L265P mutation, '
        'and a ERBB2 gain.\n'
        '<imaging_report>At age 71 years and 0 weeks, the patient had a MRI BRAIN, which showed cancer. There was response to therapy.\n'
        '<clinical_note>At age 71 years and 4 weeks, the patient had an oncologist office assessment, which showed cancer. There was '
        'response to therapy.\n'
        '<clinical_note>At age 71 years and 12 weeks, the patient had an oncologist office assessment, which showed cancer. There was '
        'progression of disease.\n'
        '<imaging_report>At age 71 years and 12 weeks, the patient had a MRI ABDOMEN, which showed no cancer.\n'
        '(End of example)\n'
        '\n'
    )
)


SECOND_PRIMARY_INSTRUCTION = (
    "In this particular group of patients, each patient have had a second primary prior unrelated cancer history before "
    "the cancer relevant to the trial. Incorporate events for the prior cancer as well as the current cancer into your output."
)


def build_seed_messages(criteria_text: str, rng: random.Random):
    from synthetic_event_prompts import seed_messages
    return seed_messages(system=SYSTEM_PROMPT, criteria=criteria_text,
                         patient_class='positive', diagnosis=DIAGNOSIS_INSTRUCTIONS,
                         diversity=DIVERSITY_INSTRUCTIONS, consistency=CONSISTENCY_INSTRUCTIONS,
                         second_primary=SECOND_PRIMARY_INSTRUCTION if rng.randrange(10) == 5 else '')


def build_patient_messages(criteria_text: str, seed_text: str):
    from synthetic_event_prompts import patient_messages
    return patient_messages(system=SYSTEM_PROMPT, criteria=criteria_text,
                            patient_class='positive', seed=seed_text,
                            instructions=EVENT_INSTRUCTIONS, consistency=CONSISTENCY_INSTRUCTIONS)


if __name__ == "__main__":
    raise SystemExit(
        "This is an import-only prompt module. Run generate_synthetic_data.py "
        "or train_all.sh --start-at generate-synthetic instead."
    )
