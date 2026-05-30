import type { PromptKey, PromptTemplate } from "../types";

export const DEFAULT_PROMPTS: Record<PromptKey, PromptTemplate> = {
  patientSummary: {
    key: "patientSummary",
    label: "Patient summary",
    description: "Summarizes a single patient's oncology history.",
    variables: ["prior_summary", "first_date", "last_date", "record_segment"],
    defaultValue: `You are an experienced clinical oncology history summarization bot.

You are maintaining a running summary of the history of a patient's active cancer(s) in their electronic health record.
You will be given:
1. A PRIOR SUMMARY of the patient's history, which may be empty.
2. THE NEXT SEGMENT of the patient's clinical record, which may contain multiple dated notes.

Update the summary to incorporate new relevant information from this segment. If the segment contains no information that would change the summary, output the prior summary exactly as-is. If the patient may not yet have a cancer diagnosis, state "No cancer diagnosis documented as of [date]" and summarize relevant medical history.

Document only these sections:
Age:
Sex:
Cancer type:
Histology:
Current extent:
Biomarkers:
Treatment history:

Boilerplate conditions:

The "Boilerplate conditions:" section should include conditions that might meet common clinical-trial exclusions, such as uncontrolled brain metastases, poor performance status, lack of measurable disease, congestive heart failure, pneumonitis, renal dysfunction, liver dysfunction, HIV or hepatitis infection, or prior unrelated cancers.

Do not include the patient's name. Include relevant dates when documented. Use generic cancer drug names when you know them. Output plain text only. Do not output markdown or tables.

PRIOR SUMMARY:
{prior_summary}

NEXT CLINICAL RECORD SEGMENT covering {first_date} to {last_date}:
{record_segment}

Write the updated summary now.`
  },
  trialSpaceExtraction: {
    key: "trialSpaceExtraction",
    label: "Trial-space extraction",
    description: "Extracts trial spaces from ClinicalTrials.gov text.",
    variables: ["trial_text"],
    defaultValue: `You are an expert clinical oncologist with broad knowledge of cancer and its treatments.

Review the clinical trial document and extract structured clinical spaces eligible for the trial. A clinical space is a unique combination of age range, sex, cancer type, histology, cancer burden, required prior treatments, excluded prior treatments, required biomarkers, and excluded biomarkers.

Output one space per line using this exact pattern:
1. Age range allowed: <age_range_allowed>. Sex allowed: <sex_allowed>. Cancer type allowed: <cancer_type_allowed>. Histology allowed: <histology_allowed>. Cancer burden allowed: <cancer_burden_allowed>. Prior treatment required: <prior_treatments_required>. Prior treatment excluded: <prior_treatments_excluded>. Biomarkers required: <biomarkers_required>. Biomarkers excluded: <biomarkers_excluded>.

Then output a line containing "Boilerplate exclusions:" and list generic exclusion criteria that are not part of the trial-space definitions.

Ignore treatment washout criteria. Do not output markdown, tables, or introductory text.

Clinical trial document:
{trial_text}`
  },
  trialDeepScreen: {
    key: "trialDeepScreen",
    label: "Trial deep screen",
    description: "LLM-based trial reasonableness screen.",
    variables: ["patient_summary", "trial_summary"],
    defaultValue: `You are a brilliant oncologist with encyclopedic knowledge about cancer and its treatment.

Evaluate whether a clinical trial is a reasonable consideration for a patient, given a clinical trial summary and a patient summary. Score how targeted the trial is for this specific patient.

Clinical trial summary:
{trial_summary}

Patient summary:
{patient_summary}

Base your judgment on age, sex, cancer type, cancer burden, prior treatments, and biomarkers. Ignore washout periods. Do not decide final eligibility; decide whether this is reasonable for the patient's oncologist to consider further.

Score 0 to 5:
0 means not reasonable.
1 means at least reasonable.
Add 1 point each for matching cancer type specificity, burden/stage specificity, prior treatment specificity, and biomarker specificity.

End with exactly:
Final score: X`
  },
  boilerplateDeepScreen: {
    key: "boilerplateDeepScreen",
    label: "Boilerplate deep screen",
    description: "LLM-based common exclusion screen.",
    variables: ["patient_boilerplate", "trial_boilerplate"],
    defaultValue: `You are a brilliant oncologist with encyclopedic knowledge about cancer and its treatment.

Evaluate whether a patient clearly has an underlying medical condition that would exclude them from a specific clinical trial.

Patient history:
{patient_boilerplate}

Trial exclusions:
{trial_boilerplate}

Do not evaluate exclusions other than those listed for this trial. Give the patient the benefit of the doubt when evidence is unclear. A mild, resolved, or historical condition should not exclude the patient unless the trial exclusion clearly applies.

End with one word exactly, including the exclamation point:
Yes!
or
No!`
  }
};

export function getDefaultPromptValues(): Record<PromptKey, string> {
  return Object.fromEntries(
    Object.entries(DEFAULT_PROMPTS).map(([key, value]) => [key, value.defaultValue])
  ) as Record<PromptKey, string>;
}
