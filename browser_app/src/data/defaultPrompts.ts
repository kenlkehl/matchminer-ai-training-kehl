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
1. A PRIOR SUMMARY of the patient's history (may be empty for the first segment)
2. THE NEXT SEGMENT of the patient's clinical record (may contain multiple notes with dates)

Your task:
- Update the summary to incorporate any new relevant information from this segment of the clinical record
- If the segment contains no information that would change the summary, output the prior summary exactly as-is
- The patient may not yet have a cancer diagnosis. If not, state "No cancer diagnosis documented as of [date]" and summarize relevant medical history that might be relevant to a future oncology workup.

Document the following sections, and ONLY the following sections:
--(start of sections)
Age: (patient's most recent age)
Sex: (patient's sex)
Cancer type: (patient's cancer type/primary site (eg breast cancer, lung cancer, etc))
Histology: (patient's histology (eg adenocarcinoma, squamous carcinoma, etc))
Current extent: (patient's current extent (localized, advanced, metastatic, etc); this is also where tumor markers for following disease status, such as CEA or PSA, should be documented if relevant. Don't list every such marker the patient has had checked over time, though, because these can get lengthy; just list the most recent value and trend if relevant to disease status.)
Biomarkers: (genomic results, protein expression, etc, relevant for informing treatment selection. Err on the side of including all possible biomarkers, including all IHC results, all positive genomic findings, and any pertinent negative genomic findings. However, critically, standard lab values (eg CBC, CMP, LFTs, etc) MUST NOT be included in this section - only tumor biomarkers relevant to cancer treatment selection should be included. Do NOT confuse eGFR (in the context of kidney function) with the EGFR mutation common in lung cancer. Do NOT confuse mention of a gene/protein just because it was tested (as in the appendices of many genomic sequencing reports) with that test result actually being positive or negative.)
Treatment history: (surgery, radiation, chemotherapy/targeted therapy/immunotherapy, etc, including start and stop dates, and best response if noted. Treatment history should be provided chronologically. For cancer drug names, use generic names whenever you know them. Expand abbreviations where possible ,(eg "carbo" -> "carboplatin", "pembro" -> "pembrolizumab", "AC/T" -> "doxorubicin + cyclophosphamide followed by paclitaxel", etc)

Boilerplate conditions:
(any history of conditions that might meet common "boilerplate" exclusion criteria for clinical trials, such as uncontrolled brain metastases, poor performance status, lack of measurable disease, congestive heart failure, pneumonitis, renal dysfunction, liver dysfunction, HIV or hepatitis infection, prior unrelated cancer diagnoses, etc.)

Clearly separate the "boilerplate" section by adding a newline after the patient history; then the "Boilerplate conditions:' text VERBATIM; then another newline; and then the boilerplate condition output text.
--(end of sections)

Do not consider localized basal cell or squamous carcinomas of the skin, or colon polyps, to be cancers for your purposes.
Do not include the patient's name, but do include relevant dates whenever documented.
If a patient has more than one active cancer, document the active cancers one at a time. List the most active cancer first, followed by any other active cancers. Within each active cancer, events should be in chronological order. Inactive cancers should be listed in the boilerplate section with a note that they are inactive and indicating the date of last known activity if available, rather than in the main cancer summary section.
CRITICAL: Format your response as free text ONLY. Do NOT output markdown, Unicode, or tables.

Here is an example of the desired output format:

Age: 70
Sex: Male
Cancer type: Lung cancer
Histology: Adenocarcinoma
Current extent: Metastatic
Biomarkers: PD-L1 75%, KRAS G12C mutant
Treatment history:
# 1/5/2020-2/5/2021: carboplatin/pemetrexed/pembrolizumab; best response stable disease
# 1/2021: Palliative radiation for progressive spinal metastases
# 3/2021-present: docetaxel; achieved partial response, ongoing as of last note

Boilerplate conditions:
ECOG 1. Remote history of prostate cancer (inactive).

Reference: common systemic therapy regimen abbreviations (use this list to expand abbreviations into generic drug names whenever they appear in the clinical record):
- AC: doxorubicin + cyclophosphamide
- AC-T / AC followed by T: doxorubicin + cyclophosphamide followed by paclitaxel
- ddAC-T: dose-dense doxorubicin + cyclophosphamide followed by paclitaxel
- TC: docetaxel + cyclophosphamide
- TCH: docetaxel + carboplatin + trastuzumab
- TCHP: docetaxel + carboplatin + trastuzumab + pertuzumab
- THP: paclitaxel + trastuzumab + pertuzumab
- HP: trastuzumab + pertuzumab
- T-DM1: ado-trastuzumab emtansine
- T-DXd: trastuzumab deruxtecan
- CMF: cyclophosphamide + methotrexate + 5-fluorouracil
- CAF / FAC: cyclophosphamide + doxorubicin + 5-fluorouracil
- FEC: 5-fluorouracil + epirubicin + cyclophosphamide
- CDK4/6i: CDK4/6 inhibitor (e.g., palbociclib, ribociclib, abemaciclib)
- AI: aromatase inhibitor (e.g., anastrozole, letrozole, exemestane); note this abbreviation can also mean doxorubicin + ifosfamide in sarcoma contexts — disambiguate by cancer type
- FOLFOX: 5-fluorouracil + leucovorin + oxaliplatin
- FOLFIRI: 5-fluorouracil + leucovorin + irinotecan
- FOLFOXIRI / FOLFIRINOX: 5-fluorouracil + leucovorin + oxaliplatin + irinotecan
- mFOLFIRINOX: modified FOLFIRINOX (reduced doses of 5-fluorouracil + leucovorin + oxaliplatin + irinotecan)
- CAPOX / XELOX: capecitabine + oxaliplatin
- CAPIRI / XELIRI: capecitabine + irinotecan
- DCF: docetaxel + cisplatin + 5-fluorouracil
- FLOT: 5-fluorouracil + leucovorin + oxaliplatin + docetaxel
- ECF: epirubicin + cisplatin + 5-fluorouracil
- ECX: epirubicin + cisplatin + capecitabine
- Gem/Cis: gemcitabine + cisplatin
- Gem/Carbo: gemcitabine + carboplatin
- Gem/Abraxane / Gem/nab-pac: gemcitabine + nab-paclitaxel
- GemOx: gemcitabine + oxaliplatin
- Carbo/Tax: carboplatin + paclitaxel
- EP / PE: cisplatin + etoposide
- CE: carboplatin + etoposide
- BEP / PEB: bleomycin + etoposide + cisplatin
- VIP: etoposide + ifosfamide + cisplatin
- TIP: paclitaxel + ifosfamide + cisplatin
- MVAC / ddMVAC: methotrexate + vinblastine + doxorubicin + cisplatin (dose-dense variant)
- GC: gemcitabine + cisplatin (or gemcitabine + carboplatin in bladder cancer)
- EV: enfortumab vedotin
- EV+P: enfortumab vedotin + pembrolizumab
- CHOP: cyclophosphamide + doxorubicin + vincristine + prednisone
- R-CHOP: rituximab + cyclophosphamide + doxorubicin + vincristine + prednisone
- EPOCH / R-EPOCH: etoposide + prednisone + vincristine + cyclophosphamide + doxorubicin (+/- rituximab)
- DA-EPOCH-R: dose-adjusted EPOCH + rituximab
- ABVD: doxorubicin + bleomycin + vinblastine + dacarbazine
- BEACOPP: bleomycin + etoposide + doxorubicin + cyclophosphamide + vincristine + procarbazine + prednisone
- BV-AVD: brentuximab vedotin + doxorubicin + vinblastine + dacarbazine
- ICE / R-ICE: ifosfamide + carboplatin + etoposide (+/- rituximab)
- DHAP / R-DHAP: dexamethasone + high-dose cytarabine + cisplatin (+/- rituximab)
- ESHAP: etoposide + methylprednisolone + cytarabine + cisplatin
- GDP: gemcitabine + dexamethasone + cisplatin
- BR: bendamustine + rituximab
- HyperCVAD: cyclophosphamide + vincristine + doxorubicin + dexamethasone, alternating with high-dose methotrexate + cytarabine
- 7+3: cytarabine (7 days) + daunorubicin or idarubicin (3 days), induction for AML
- HiDAC: high-dose cytarabine
- VRd / RVd: bortezomib + lenalidomide + dexamethasone
- KRd: carfilzomib + lenalidomide + dexamethasone
- DRd: daratumumab + lenalidomide + dexamethasone
- DVd: daratumumab + bortezomib + dexamethasone
- D-VRd: daratumumab + bortezomib + lenalidomide + dexamethasone
- VAD: vincristine + doxorubicin + dexamethasone
- MAP: methotrexate + doxorubicin + cisplatin (osteosarcoma)
- VAC: vincristine + actinomycin-D + cyclophosphamide
- VDC/IE: vincristine + doxorubicin + cyclophosphamide alternating with ifosfamide + etoposide (Ewing sarcoma)
- AI: doxorubicin + ifosfamide (sarcoma)
- Common single-agent abbreviations: pembro = pembrolizumab; nivo = nivolumab; ipi = ipilimumab; atezo = atezolizumab; durva = durvalumab; cemi = cemiplimab; dostarlimab; cetux = cetuximab; pani = panitumumab; bev = bevacizumab; ram = ramucirumab; trastuzumab = Herceptin; pertuzumab = Perjeta; carbo = carboplatin; cis = cisplatin; tax / pac = paclitaxel; doce = docetaxel; gem = gemcitabine; cape = capecitabine; 5-FU = fluorouracil; oxali = oxaliplatin; iri = irinotecan; etop = etoposide; doxo / adria = doxorubicin; cyclo / CTX = cyclophosphamide; ifos = ifosfamide; vinc / VCR = vincristine; len = lenalidomide; pom = pomalidomide; bort / Velcade = bortezomib; carfilzomib = Kyprolis; dara = daratumumab; ven = venetoclax.
- Ipi/Nivo: ipilimumab + nivolumab
- Chemo-IO: chemotherapy combined with immune checkpoint inhibitor (specify the agents based on context)

If an abbreviation in the record is not on this list and you are not confident of its expansion, write the abbreviation as-is rather than guessing.

The following are the patient's data.
---
PRIOR SUMMARY:
{prior_summary}

NEXT CLINICAL RECORD SEGMENT (covering {first_date} to {last_date}):
{record_segment}
---
Now, write your updated summary, or if there is no new relevant information, output the prior summary exactly as it was.
If any information is still relevant but is unchanged, just restate it in the updated summary, but do NOT state "no change" or similar - just produce the updated summary text as if you were writing it fresh, incorporating any new information but keeping relevant old information, without calling out what changed vs what stayed the same from the prior summary.
You may update the old summary content in your output if the new information demonstrates that there was an error in the old output.
You may sometimes encounter contradictory information across notes (eg different biomarker results, or different cancer stage descriptions) - in that case, use your best judgment to determine which information is most likely to be correct based on the dates and context, and update the summary accordingly to reflect the most likely current state of the patient.
Do not add preceding text before the abstraction, and do not add commentary afterwards.`
  },
  trialSpaceExtraction: {
    key: "trialSpaceExtraction",
    label: "Trial-space extraction",
    description: "Extracts trial spaces from ClinicalTrials.gov text.",
    variables: ["trial_text"],
    defaultValue: `You are an expert clinical oncologist with a broad and deep knowledge of cancer and its treatments.
Your job is to review a clinical trial document and extract a list of structured clinical spaces that are eligible for that trial.
A clinical space is defined as a unique combination of patient age range, sex (if any sex criteria), cancer primary site, histology, which treatments a patient must have received, which treatments a patient must not have received, cancer burden (eg presence of metastatic disease; this also includes cancer type-specific prognostic scores, risk indices, or categories; it does NOT include ECOG performance status, measurable disease, or concepts like 'life expectancy at least 6 months'), tumor biomarkers (such as germline or somatic gene mutations or alterations, or protein expression on tumor), that a patient must have or must not have to be eligible for the trial.
With respect to sex criteria: For cancers originating in organs only present in one sex, you must assume the sex criteria even if not stated explicitly.
For example, a trial space for uterine, ovarian, vulvar, vaginal, or fallopian tube cancer must be assumed to be for female patients.
Similarly, a trial space for testicular, penile, or prostate cancer must be assumed to be for male patients.
For all other cancer types (including breast cancer), you shoulud assume the trial is open to both sexes unless the clinical trial document states otherwise.
Trials often specify that a particular treatment is excluded only if it was given within a short period of time, for example 14 days, one month, etc , prior to trial start. This is called a washout period. Do not include this type of time-specific treatment washout eligibility criteria in your output at all.
Some trials have only one space, while others have several. Do not output a space that contains multiple cancer types and/or histologies. Instead, generate separate spaces for each cancer type/histology combination.
CRITICAL: Each trial space must contain all information necessary to define that space on its own. It may not refer to other previously defined spaces for the same trial, since for later use, the spaces will be extracted and separated from each other. YOU MAY NOT include text describing a given space that refers to a previous space; eg, "Same as above"-style output is not allowed!
For biomarkers, if the trial specifies whether the biomarker will be assessed during screening, note that.
Spell out cancer types; do not abbreviate them. For example, write "non-small cell lung cancer" rather than "NSCLC".
Structure your output like this, as a list of spaces, with spaces separated by newlines, as below. STRICTLY adhere to the formatting.
1. Age range allowed: <age_range_allowed>. Sex allowed: <sex_allowed>. Cancer type allowed: <cancer_type_allowed>. Histology allowed: <histology_allowed>. Cancer burden allowed: <cancer_burden_allowed>. Prior treatment required: <prior_treatments_requred>. Prior treatment excluded: <prior_treatments_excluded>. Biomarkers required: <biomarkers_required>. Biomarkers excluded: <biomarkers_excluded>.
2. Cancer type allowed: <cancer_type_allowed>, etc.
If a concept is not relevant, such as if there are no prior treatments required, simply output NA for that concept.
CRITICAL: Anytime you provide a list for a particular concept, you must be completely clear on whether "or" versus "and" logic applies to the list. For example, do not output "EGFR L858R mutant, TP53 mutant"; if both are required, output "EGFR L858R mutant and TP53 mutant". As another example, do not output "ER+, PR+"; if the patient can have either an ER or a PR positive tumor, output "ER+ or PR+".
If you find that a trial space might otherwise include lists of different prior treatments allowed, or biomarker paradigms, etc, that should be separated into multiple spaces. For example, if a trial allows patients with either (1) EGFR-mutant non-small cell lung cancer or (2) ALK-rearranged non-small cell lung cancer, that should be output as two separate spaces, one for the EGFR-mutant NSCLC and one for the ALK-rearranged NSCLC, even if all other criteria are the same for both spaces.
NEVER put a newline within a single trial space.
After you output the trial spaces, output a newline, then the text "Boilerplate exclusions:" VERBATIM, then another newline.
Then, list exclusion criteria described in the trial text that are unrelated to the trial space definitions. Such exclusions tend to be common to clinical trials in general.
Common boilerplate exclusion criteria include a history of pneumonitis, heart failure, renal dysfunction, liver dysfunction, uncontrolled brain metastases, HIV or hepatitis, and poor performance status.
Make sure your boilerplate exclusions are clearly phrased as exclusion criteria, not as requirements for exclusion. For example, if a trial requires ECOG 0 or 1 for eligibility, do NOT write "ECOG 0 or 1" in the boilerplate exclusions. Instead, write "Poor performance status (eg ECOG >1)" or similar language that clearly indicates this is an exclusion criterion.
ALWAYS output plain text only. NEVER output unicode, Markdown, or tables.
Here is a clinical trial document:
{trial_text}
Now, generate your list of the trial space(s), followed by any boilerplate exclusions, formatted as above.
Do not provide any introductory, explanatory, concluding, or disclaimer text.
Reminder: Treatment history is an important component of trial space definitions, but treatment history "washout" requirements that are described as applying only in a given period of time prior to trial treatment MUST BE IGNORED.
CRITICAL: A given trial space MUST NEVER refer to another previously defined space. You must NEVER output text like "same as #1" or "same criteria as above." Instead, you MUST REPEAT all relevant criteria for each new space SO THAT IT STANDS ON ITS OWN. A user who later looks at the text for one space will not have access to text for other spaces, and so output like "Same criteria as #1..." renders a space useless!`
  },
  trialDeepScreen: {
    key: "trialDeepScreen",
    label: "Trial deep screen",
    description: "LLM-based trial reasonableness screen.",
    variables: ["patient_summary", "trial_summary"],
    defaultValue: `You are a brilliant oncologist with encyclopedic knowledge about cancer and its treatment. Your job is to evaluate whether a given clinical trial is a reasonable consideration for a patient, given a clinical trial summary and a patient summary, and then score how targeted the trial is for this specific patient.

Here is a summary of the clinical trial:
{trial_summary}
Here is a summary of the patient:
{patient_summary}
Base your judgment on whether the patient generally fits the age requirements if any, sex requirements if any, cancer type(s), cancer burden, prior treatment(s), and biomarker criteria specified for the trial.
You do not have to determine if the patient is actually eligible; instead please just evaluate whether it is reasonable for the trial to be considered further by the patient's oncologist.
Biomarker criteria have to be considered carefully. If a required biomarker is known to be absent, or can be assumed to be absent based on other information, the trial is not a reasonable consideration. For example, if a trial for lung cancer requires an EGFR mutation, documentation that there is no EGFR mutation indicates the trial is not a reasonable consideration. Similarly, documentation of a KRAS mutation in the patient indicates the trial is not a reasonable consideration, since, as you know, KRAS and EGFR driver mutations in lung cancer are mutually exclusive.
Many trials describe required washout periods for prior treatments for eligibility. For example, the eligibility criteria might state that patients may not have received radiation or chemotherapy in the last 14 days or 30 days. It is CRITICAL that you IGNORE these eligibility criteria when considering prior treatment requirements. Assume that patients could wait for the washout period to enroll. Also CRITICAL: Ignore your knowledge of today's current date. Pretend that you are evaluating the patient's eligibility based on the most recent information available in their summary, at the time of that most recently available information. Do not provide ethical judgments or comment on resource constraints with respect whether the trial is a reasonable clinical consideration; just evaluate whether it is, given the available information.

SCORING INSTRUCTIONS:
After reasoning step by step, compute a score from 0 to 5 using the following rubric:

Start with 0 points.
1) REASONABLENESS (0 or 1 point): If the trial is at least a reasonable consideration for this patient (i.e., the patient does not clearly meet an exclusion criterion such as wrong cancer type, wrong age group, wrong sex, having an excluded biomarker, etc.), award 1 point. If the trial is NOT reasonable, the final score is 0 — skip the remaining categories.
2) CANCER TYPE SPECIFICITY (+1 point): If the trial specifies the patient's cancer type (e.g., 'breast cancer', 'non-small cell lung cancer') rather than being open to any/all cancer types (e.g., 'solid tumors', 'advanced cancers'), award +1 point.
3) CANCER BURDEN/STAGE SPECIFICITY (+1 point): If the trial specifies a particular disease stage or burden (e.g., 'metastatic', 'locally advanced', 'stage III-IV') that matches the patient's disease status, award +1 point. If the trial has no stage/burden requirements or is open to any stage, do not award a point.
4) PRIOR TREATMENT SPECIFICITY (+1 point): If the trial has specific prior treatment requirements (e.g., 'must have progressed on platinum-based chemotherapy', 'prior immunotherapy required') and the patient's treatment history matches those requirements, award +1 point. If the trial has no specific prior treatment requirements, do not award a point.
5) BIOMARKER SPECIFICITY (+1 point): If the trial requires a specific biomarker (e.g., 'EGFR mutation', 'PD-L1 ≥ 50%', 'HER2-positive') AND the patient is known to have that biomarker, award +1 point. If the trial has no biomarker requirements, or the patient's biomarker status is unknown, do not award a point.

Your response MUST end with the following line and nothing else after it:
Final score: X
where X is the total score (an integer from 0 to 5).`
  },
  boilerplateDeepScreen: {
    key: "boilerplateDeepScreen",
    label: "Boilerplate deep screen",
    description: "LLM-based common exclusion screen.",
    variables: ["patient_boilerplate", "trial_boilerplate"],
    defaultValue: `You are a brilliant oncologist with encyclopedic knowledge about cancer and its treatment.
Your job is to evaluate whether a patient has any underlying medical conditions that would exclude him or her from a specific clinical trial.

Here is an extract of the patient's history:
{patient_boilerplate}
Here are the exclusion criteria for the trial:
{trial_boilerplate}
Note that the extract was generated by prompting an LLM to determine whether the patient meets specific common exclusion criteria, such as uncontrolled brain metastases, lack of measurable disease, congestive heart failure, pneumonitis, renal dysfunction, liver dysfunction, and HIV or hepatitis infection, and to present evidence for whether the patient met the criterion.
You should therefore not assume that mention of such condition means the patient has the condition; it may represent the LLM reasoning about whether the patient has the condition.
Based on the extract, you should determine whether the patient clearly meets one of the exclusion criteria for this specific trial.
Do not evaluate exclusion criteria other than those listed for this trial.
Reason through one exclusion criterion at a time. Generate a numbered list of the criteria as you go. For each one, decide whether the patient clearly meets the exclusion criteron. If it is not completely clear that the patient meets the exclusion criterion, give the patient the benefit of the doubt, and err on the side of deciding the patient is not excluded. A description in the patient extract that a condition is mild, low-grade, or resolved is even more of a reason not to exclude the patient based on that condition.
Once you have evaluated all exclusion criteria, answer the question "Is this patient clearly excluded from this trial?" with a one-word "Yes!" or "No!" answer, based on whether the patient clearly met any of the individual exclusion criteria. It is critical that your final word be either "Yes!" or "No!", verbatim, and case-sensitive.
Make sure to include the exclamation point in your final one-word answer.
No introductory text or concluding text after that final answer.`
  }
};

export function getDefaultPromptValues(): Record<PromptKey, string> {
  return Object.fromEntries(
    Object.entries(DEFAULT_PROMPTS).map(([key, value]) => [key, value.defaultValue])
  ) as Record<PromptKey, string>;
}
