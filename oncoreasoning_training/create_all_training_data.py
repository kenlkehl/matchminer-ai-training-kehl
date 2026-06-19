#!/usr/bin/env python3
"""
Consolidated oncoreasoning training data preparation.

Replaces the 4 individual data-creation scripts and the combiner/tokenizer:
  - create_training_data_boilerplate_checks.py
  - create_training_data_summarize_histories.py
  - create_training_data_trialchecks_with_labels.py
  - create_training_data_trialspaces.py
  - prepare_training_data.py

Usage:
  python create_all_training_data.py --max-seq-length 32000
"""

import argparse
import gc
import json
import math
import multiprocessing as mp
import os
import random
import re
import shutil
import struct
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets.arrow_writer import ArrowWriter
from transformers import AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_DATA_DIR = REPO_ROOT.parent / "data" / "no_phi"
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_DIR / "oncoreasoning_training_data"
DEFAULT_WORKER_COUNT = max(1, (os.cpu_count() or 1) - 1)
TEXT_SCHEMA = pa.schema([("text", pa.string())])


# ---------------------------------------------------------------------------
# Constants: prompt templates copied verbatim from the original scripts
# ---------------------------------------------------------------------------

TRIALSPACE_PROMPT_HEADER = (
    "You are an expert clinical oncologist with a broad and deep knowledge of cancer and its treatments.\n"
    "Your job is to review a clinical trial document and extract a list of structured clinical spaces that are eligible for that trial.\n"
    "A clinical space is defined as a unique combination of patient age range, sex (if any sex criteria), cancer primary site, histology, which treatments a patient must have received, "
    "which treatments a patient must not have received, cancer burden (eg presence of metastatic disease; this also includes cancer type-specific prognostic scores, risk indices, or categories; it does NOT include ECOG performance status, measurable disease, or concepts like 'life expectancy at least 6 months'), tumor biomarkers (such as "
    "germline or somatic gene mutations or alterations, or protein expression on tumor), that a patient must have or must not have to "
    "be eligible for the trial. \n"
    "With respect to sex criteria: For cancers originating in organs only present in one sex, you must assume the sex criteria even if not stated explicitly.\n"
    "For example, a trial space for uterine, ovarian, vulvar, vaginal, or fallopian tube cancer must be assumed to be for female patients.\n"
    "Similarly, a trial space for testicular, penile, or prostate cancer must be assumed to be for male patients.\n"
    "For all other cancer types (including breast cancer), you shoulud assume the trial is open to both sexes unless the clinical trial document states otherwise.\n"
    "Trials often specify that a particular treatment is excluded only if it was given within a short period of time, for example 14 days, "
    "one month, etc , prior to trial start. This is called a washout period. Do not include this type of time-specific treatment washout "
    "eligibility criteria in your output at all.\n"
    "Some trials have only one space, while others have several. Do not output a space that contains multiple cancer types and/or histologies. "
    "Instead, generate separate spaces for each cancer type/histology combination.\n"
    "CRITICAL: Each trial space must contain all information necessary to define that space on its own. It may not refer to other previously "
    "defined spaces for the same trial, since for later use, the spaces will be extracted and separated from each other. YOU MAY NOT include "
    "text describing a given space that refers to a previous space; eg, \"Same as above\"-style output is not allowed!\n"
    "For biomarkers, if the trial specifies whether the biomarker will be assessed during screening, note that.\n"
    "Spell out cancer types; do not abbreviate them. For example, write \"non-small cell lung cancer\" rather than \"NSCLC\".\n"
    "Structure your output like this, as a list of spaces, with spaces separated by newlines, as below. STRICTLY adhere to the formatting.\n"
    "1. Age range allowed: <age_range_allowed>. Sex allowed: <sex_allowed>. Cancer type allowed: <cancer_type_allowed>. Histology allowed: <histology_allowed>. Cancer burden allowed: <cancer_burden_allowed>. Prior treatment required: <prior_treatments_requred>. Prior treatment excluded: <prior_treatments_excluded>. Biomarkers required: <biomarkers_required>. Biomarkers excluded: <biomarkers_excluded>. \n"
    "2. Cancer type allowed: <cancer_type_allowed>, etc.\n"
    "If a concept is not relevant, such as if there are no prior treatments required, simply output NA for that concept.\n"
    "CRITICAL: Anytime you provide a list for a particular concept, you must be completely clear on whether \"or\" versus \"and\" logic applies "
    "to the list. For example, do not output \"EGFR L858R mutant, TP53 mutant\"; if both are required, output \"EGFR L858R mutant and TP53 mutant\". "
    "As another example, do not output \"ER+, PR+\"; if the patient can have either an ER or a PR positive tumor, output \"ER+ or PR+\".\n"
    "If you find that a trial space might otherwise include lists of different prior treatments allowed, or biomarker paradigms, etc, that should be separated into multiple spaces. For example, if a trial allows patients with either (1) EGFR-mutant non-small cell lung cancer or (2) ALK-rearranged non-small cell lung cancer, that should be output as two separate spaces, one for the EGFR-mutant NSCLC and one for the ALK-rearranged NSCLC, even if all other criteria are the same for both spaces.\n"
    "NEVER put a newline within a single trial space.\n"
    "After you output the trial spaces, output a newline, then the text \"Boilerplate exclusions:\" VERBATIM, then another newline.\n"
    "Then, list exclusion criteria described in the trial text that are unrelated to the trial space definitions. Such exclusions tend to be common "
    "to clinical trials in general.\n"
    "Common boilerplate exclusion criteria include a history of pneumonitis, heart failure, renal dysfunction, liver dysfunction, uncontrolled brain "
    "metastases, HIV or hepatitis, and poor performance status.\n"
    "Make sure your boilerplate exclusions are clearly phrased as exclusion criteria, not as requirements for exclusion. For example, if a trial requires ECOG 0 or 1 for eligibility, do NOT write \"ECOG 0 or 1\" in the boilerplate exclusions. Instead, write \"Poor performance status (eg ECOG >1)\" or similar language that clearly indicates this is an exclusion criterion.\n"
    "ALWAYS output plain text only. NEVER output unicode, Markdown, or tables.\n"
)

TRIALSPACE_PROMPT_SUFFIX = (
    "Now, generate your list of the trial space(s), followed by any boilerplate exclusions, formatted as above.\n"
    "Do not provide any introductory, explanatory, concluding, or disclaimer text.\n"
    "Reminder: Treatment history is an important component of trial space definitions, but treatment history \"washout\" requirements that are "
    "described as applying only in a given period of time prior to trial treatment MUST BE IGNORED.\n"
    "CRITICAL: A given trial space MUST NEVER refer to another previously defined space. You must NEVER output text like \"same as #1\" or "
    "\"same criteria as above.\" Instead, you MUST REPEAT all relevant criteria for each new space SO THAT IT STANDS ON ITS OWN. A user who later "
    "looks at the text for one space will not have access to text for other spaces, and so output like \"Same criteria as #1...\" renders a space useless!"
)


# ---------------------------------------------------------------------------
# Truncation helpers
# ---------------------------------------------------------------------------

def truncate_field(text: str, max_tokens: int, tokenizer) -> str:
    """Truncate text to max_tokens using head+tail strategy."""
    toks = tokenizer(text, add_special_tokens=False).input_ids
    if len(toks) <= max_tokens:
        return text
    half = max_tokens // 2
    return tokenizer.decode(toks[:half]) + " ... " + tokenizer.decode(toks[-half:])


def token_len(text: str, tokenizer) -> int:
    """Return the number of tokens in text (no special tokens)."""
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def get_optional_text(row, column: str) -> str:
    """Return a string column value, treating missing/NA as empty."""
    if column not in row:
        return ""
    value = row[column]
    if pd.isna(value):
        return ""
    return str(value)


def normalize_source_text(text: str) -> str:
    """Normalize source-model text before target-tokenizer chat templating."""
    return str(text).replace("\r\n", "\n").replace("\r", "\n").strip()


def clean_reasoning_text(reasoning: str) -> str:
    """Remove source-model channel wrappers from a reasoning trace."""
    text = normalize_source_text(reasoning)
    if not text:
        return ""

    text = re.sub(r'^\s*<\|channel\>(thought|analysis|thinking)\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*<channel\|>\s*$', '', text)
    text = re.sub(r'^\s*<think>\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*</think>\s*$', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*assistantfinal\s*$', '', text)
    return text.strip()


def clean_final_text(final_response: str) -> str:
    """Remove source-model final-channel prefixes from a final answer."""
    text = normalize_source_text(final_response)
    if not text:
        return ""

    text = re.sub(r'^\s*assistantfinal\s*', '', text)
    text = re.sub(r'^\s*<\|channel\>final\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\s*<channel\|>\s*', '', text)
    return text.strip()


def build_assistant_target(reasoning: str, final_response: str) -> str:
    """Build target-model assistant content from separate reasoning/final text."""
    reasoning = clean_reasoning_text(reasoning)
    final_response = clean_final_text(final_response)

    if reasoning and final_response:
        return f"<think>\n{reasoning}\n</think>\n{final_response}"
    if reasoning:
        return f"<think>\n{reasoning}\n</think>"
    return final_response


def build_reasoning_response(row, reasoning_col: str, response_col: str) -> str:
    """Build the assistant target from optional reasoning and final response."""
    return build_assistant_target(
        reasoning=get_optional_text(row, reasoning_col),
        final_response=get_optional_text(row, response_col),
    )


def split_combined_reasoning_and_final(combined: str, final_response: str = ""):
    """Split source traces where final output is appended to reasoning.

    Trial-space traces currently store a Gemma-style combined field ending in
    the separate ``space_output_no_reasoning`` value, usually after a
    ``<channel|>`` marker.  Prefer the explicit final column when present, and
    fall back to known source-model separators for older data.
    """
    combined = normalize_source_text(combined)
    final_response = clean_final_text(final_response)

    if not combined:
        return "", final_response

    if final_response:
        combined_rstripped = combined.rstrip()
        final_rstripped = final_response.rstrip()
        if combined_rstripped.endswith(final_rstripped):
            reasoning = combined_rstripped[:-len(final_rstripped)]
            return clean_reasoning_text(reasoning), final_response

        idx = combined.rfind(final_response)
        if idx >= 0:
            return clean_reasoning_text(combined[:idx]), final_response

    for marker in ("assistantfinal", "<|channel>final", "<channel|>"):
        if marker in combined:
            reasoning, parsed_final = combined.rsplit(marker, 1)
            return clean_reasoning_text(reasoning), clean_final_text(parsed_final)

    return clean_reasoning_text(combined), final_response


def build_combined_reasoning_response(row, combined_col: str, final_col: str, fallback_final_col: str = "") -> str:
    """Build assistant target from a combined reasoning+final source column."""
    final_response = get_optional_text(row, final_col)
    if not final_response and fallback_final_col:
        final_response = get_optional_text(row, fallback_final_col)
    reasoning, final_response = split_combined_reasoning_and_final(
        combined=get_optional_text(row, combined_col),
        final_response=final_response,
    )
    return build_assistant_target(reasoning, final_response)


# ---------------------------------------------------------------------------
# Per-task prompt builders (return messages list + truncatable field info)
# ---------------------------------------------------------------------------

def build_boilerplate_messages(row):
    """Build chat messages for a boilerplate check example."""
    patient_bp = row['patient_boilerplate_text']
    trial_bp = row['trial_boilerplate_text']
    answer = build_reasoning_response(
        row,
        'boilerplate_check_llm_reasoning',
        'boilerplate_check_llm_response',
    )

    user_content = (
        "You are a brilliant oncologist with encyclopedic knowledge about cancer and its treatment.\n"
        "Your job is to evaluate whether a patient has any underlying medical conditions that would exclude him or her from a specific clinical trial.\n\n"
        f"Here is an extract of the patient's history:\n{patient_bp}\n"
        f"Here are the exclusion criteria for the trial:\n{trial_bp}\n"
        "Note that the extract was generated by prompting an LLM to determine whether the patient meets specific common exclusion criteria, "
        "such as uncontrolled brain metastases, lack of measurable disease, congestive heart failure, pneumonitis, renal dysfunction, "
        "liver dysfunction, and HIV or hepatitis infection, and to present evidence for whether the patient met the criterion.\n"
        "You should therefore not assume that mention of such condition means the patient has the condition; it may represent the LLM reasoning "
        "about whether the patient has the condition.\n"
        "Based on the extract, you should determine whether the patient clearly meets one of the exclusion criteria for this specific trial.\n"
        "Do not evaluate exclusion criteria other than those listed for this trial.\n"
        "Reason through one exclusion criterion at a time. Generate a numbered list of the criteria as you go. For each one, decide whether the patient clearly "
        "meets the exclusion criteron. If it is not completely clear that the patient meets the exclusion criterion, give the patient the benefit of the doubt, "
        "and err on the side of deciding the patient is not excluded. A description in the patient extract that a condition is mild, low-grade, or resolved is even "
        "more of a reason not to exclude the patient based on that condition.\n"
        'Once you have evaluated all exclusion criteria, answer the question "Is this patient clearly excluded from this trial?" with a one-word "Yes!" or "No!" answer, '
        "based on whether the patient clearly met any of the individual exclusion criteria. It is critical that your final word be either \"Yes!\" or \"No!\", verbatim, and case-sensitive.\n"
        "Make sure to include the exclamation point in your final one-word answer.\n"
        "No introductory text or concluding text after that final answer."
    )

    messages = [
        {'role': 'system', 'content': "Reasoning: high"},
        {'role': 'user', 'content': user_content},
        {'role': 'assistant', 'content': answer},
    ]
    return messages


def build_summarization_messages(row):
    """Build chat messages for a summarization example."""
    prior_summary = row['prior_summary'] if pd.notna(row['prior_summary']) else None
    first_date = str(row['first_date']) if pd.notna(row['first_date']) else "unknown date"
    last_date = str(row['last_date']) if pd.notna(row['last_date']) else "unknown date"
    chunk_text = str(row['chunk_text']) if pd.notna(row['chunk_text']) else ""

    full_response = build_reasoning_response(
        row,
        'new_summary_reasoning',
        'new_summary',
    )

    prior_summary_text = prior_summary if prior_summary else "None - this is the first segment for this patient"

    user_content = f"""You are an experienced clinical oncology history summarization bot.

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
{prior_summary_text}

NEXT CLINICAL RECORD SEGMENT (covering {first_date} to {last_date}):
{chunk_text}
---
Now, write your updated summary, or if there is no new relevant information, output the prior summary exactly as it was.{" "}
If any information is still relevant but is unchanged, just restate it in the updated summary, but do NOT state "no change" or similar - just produce the updated summary text as if you were writing it fresh, incorporating any new information but keeping relevant old information, without calling out what changed vs what stayed the same from the prior summary.{" "}
You may update the old summary content in your output if the new information demonstrates that there was an error in the old output.
You may sometimes encounter contradictory information across notes (eg different biomarker results, or different cancer stage descriptions) - in that case, use your best judgment to determine which information is most likely to be correct based on the dates and context, and update the summary accordingly to reflect the most likely current state of the patient.
Do not add preceding text before the abstraction, and do not add commentary afterwards."""

    messages = [
        {'role': 'system', 'content': 'Reasoning: high'},
        {'role': 'user', 'content': user_content},
        {'role': 'assistant', 'content': full_response},
    ]
    return messages


def build_trialcheck_messages(row):
    """Build chat messages for a trial check example."""
    patient_summary = row['patient_summary']
    trial_summary = row['this_space']
    answer = build_reasoning_response(
        row,
        'trialcheck_llm_reasoning',
        'trialcheck_llm_response',
    )

    user_content = (
        "You are a brilliant oncologist with encyclopedic knowledge about cancer and its treatment. "
        "Your job is to evaluate whether a given clinical trial is a reasonable consideration for a patient, "
        "given a clinical trial summary and a patient summary, and then score how targeted the trial is for "
        "this specific patient.\n\n"
        f"Here is a summary of the clinical trial:\n{trial_summary}\n"
        f"Here is a summary of the patient:\n{patient_summary}\n"
        "Base your judgment on whether the patient generally fits the age requirements if any, sex requirements if any, cancer type(s), cancer burden, prior treatment(s), "
        "and biomarker criteria specified for the trial.\n"
        "You do not have to determine if the patient is actually eligible; instead please just evaluate whether it is reasonable "
        "for the trial to be considered further by the patient's oncologist.\n"
        "Biomarker criteria have to be considered carefully. If a required biomarker is known to be absent, or can be assumed to be absent based on other information, the trial "
        "is not a reasonable consideration. For example, if a trial for lung cancer requires an EGFR mutation, documentation that there "
        "is no EGFR mutation indicates the trial is not a reasonable consideration. Similarly, documentation of a KRAS mutation in the "
        "patient indicates the trial is not a reasonable consideration, since, as you know, KRAS and EGFR driver mutations in lung cancer "
        "are mutually exclusive.\n"
        "Many trials describe required washout periods for prior treatments for eligibility. For example, the eligibility criteria might state "
        "that patients may not have received radiation or chemotherapy in the last 14 days or 30 days. It is CRITICAL that you IGNORE these "
        "eligibility criteria when considering prior treatment requirements. Assume that patients could wait for the washout period to enroll. "
        "Also CRITICAL: Ignore your knowledge of today's current date. Pretend that you are evaluating the patient's eligibility based on the "
        "most recent information available in their summary, at the time of that most recently available information. "
        "Do not provide ethical judgments or comment on resource constraints with respect whether the trial is a reasonable clinical "
        "consideration; just evaluate whether it is, given the available information.\n\n"
        "SCORING INSTRUCTIONS:\n"
        "After reasoning step by step, compute a score from 0 to 5 using the following rubric:\n\n"
        "Start with 0 points.\n"
        "1) REASONABLENESS (0 or 1 point): If the trial is at least a reasonable consideration for this patient "
        "(i.e., the patient does not clearly meet an exclusion criterion such as wrong cancer type, wrong age group, "
        "wrong sex, having an excluded biomarker, etc.), award 1 point. If the trial is NOT reasonable, the final score is 0 — "
        "skip the remaining categories.\n"
        "2) CANCER TYPE SPECIFICITY (+1 point): If the trial specifies the patient's cancer type (e.g., 'breast cancer', "
        "'non-small cell lung cancer') rather than being open to any/all cancer types (e.g., 'solid tumors', 'advanced cancers'), "
        "award +1 point.\n"
        "3) CANCER BURDEN/STAGE SPECIFICITY (+1 point): If the trial specifies a particular disease stage or burden "
        "(e.g., 'metastatic', 'locally advanced', 'stage III-IV') that matches the patient's disease status, award +1 point. "
        "If the trial has no stage/burden requirements or is open to any stage, do not award a point.\n"
        "4) PRIOR TREATMENT SPECIFICITY (+1 point): If the trial has specific prior treatment requirements "
        "(e.g., 'must have progressed on platinum-based chemotherapy', 'prior immunotherapy required') "
        "and the patient's treatment history matches those requirements, award +1 point. "
        "If the trial has no specific prior treatment requirements, do not award a point.\n"
        "5) BIOMARKER SPECIFICITY (+1 point): If the trial requires a specific biomarker (e.g., 'EGFR mutation', "
        "'PD-L1 ≥ 50%', 'HER2-positive') AND the patient is known to have that biomarker, award +1 point. "
        "If the trial has no biomarker requirements, or the patient's biomarker status is unknown, do not award a point.\n\n"
        "Your response MUST end with the following line and nothing else after it:\n"
        "Final score: X\n"
        "where X is the total score (an integer from 0 to 5)."
    )

    messages = [
        {'role': 'system', 'content': "Reasoning: high"},
        {'role': 'user', 'content': user_content},
        {'role': 'assistant', 'content': answer},
    ]
    return messages


def build_trialspace_messages(row):
    """Build chat messages for a trial space extraction example."""
    trial_text = row['trial_text']
    answer = build_combined_reasoning_response(
        row,
        combined_col='space_reasoning_and_output',
        final_col='space_output_no_reasoning',
        fallback_final_col='space_text',
    )

    messages = [
        {'role': 'system', 'content': "Reasoning: high."},
        {
            'role': 'user',
            'content': (
                TRIALSPACE_PROMPT_HEADER
                + "Here is a clinical trial document:\n"
                + str(trial_text)
                + "\n"
                + TRIALSPACE_PROMPT_SUFFIX
            ),
        },
        {'role': 'assistant', 'content': answer},
    ]
    return messages


# ---------------------------------------------------------------------------
# Per-task: extractors for truncatable fields and prompt rebuilders
# ---------------------------------------------------------------------------

# Each task defines:
#   get_truncatable_fields(row) -> dict[field_name -> text]
#   rebuild_messages(row, truncated_fields) -> messages list

def boilerplate_truncatable_fields(row):
    return {
        'patient_boilerplate_text': row['patient_boilerplate_text'],
        'trial_boilerplate_text': row['trial_boilerplate_text'],
    }


def boilerplate_rebuild(row, truncated):
    row = row.copy()
    row['patient_boilerplate_text'] = truncated['patient_boilerplate_text']
    row['trial_boilerplate_text'] = truncated['trial_boilerplate_text']
    return build_boilerplate_messages(row)


def summarization_truncatable_fields(row):
    fields = {'chunk_text': str(row['chunk_text']) if pd.notna(row['chunk_text']) else ""}
    if pd.notna(row['prior_summary']) and row['prior_summary']:
        fields['prior_summary'] = row['prior_summary']
    return fields


def summarization_rebuild(row, truncated):
    row = row.copy()
    for k, v in truncated.items():
        row[k] = v
    return build_summarization_messages(row)


def trialcheck_truncatable_fields(row):
    return {
        'patient_summary': row['patient_summary'],
        'this_space': row['this_space'],
    }


def trialcheck_rebuild(row, truncated):
    row = row.copy()
    row['patient_summary'] = truncated['patient_summary']
    row['this_space'] = truncated['this_space']
    return build_trialcheck_messages(row)


def trialspace_truncatable_fields(row):
    return {'trial_text': row['trial_text']}


def trialspace_rebuild(row, truncated):
    row = row.copy()
    row['trial_text'] = truncated['trial_text']
    return build_trialspace_messages(row)


# ---------------------------------------------------------------------------
# Generic prompt builder with truncation
# ---------------------------------------------------------------------------

def build_prompt_with_truncation(
    row,
    tokenizer,
    max_seq_length,
    build_messages_fn,
    get_truncatable_fn,
    rebuild_fn,
):
    """
    Build a training prompt, truncating if needed to fit max_seq_length.

    Returns (prompt_string, was_truncated) or (None, False) if the example
    cannot fit even after truncation.
    """
    messages = build_messages_fn(row)
    prompt = tokenizer.apply_chat_template(conversation=messages, tokenize=False, enable_thinking=True)
    total_tokens = token_len(prompt, tokenizer)

    if total_tokens <= max_seq_length:
        return prompt, False

    # Need truncation
    trunc_fields = get_truncatable_fn(row)
    if not trunc_fields:
        return None, False  # nothing to truncate

    # Measure token sizes of truncatable fields
    field_tokens = {k: token_len(v, tokenizer) for k, v in trunc_fields.items()}
    total_trunc_tokens = sum(field_tokens.values())

    # overhead = tokens from everything except the truncatable fields
    overhead = total_tokens - total_trunc_tokens
    available = max_seq_length - overhead

    if available <= 0:
        return None, False  # even without truncatable content, prompt is too long

    # Distribute available budget proportionally by original field sizes
    truncated = {}
    for field_name, text in trunc_fields.items():
        if total_trunc_tokens > 0:
            budget = int(available * field_tokens[field_name] / total_trunc_tokens)
        else:
            budget = available // len(trunc_fields)
        budget = max(budget, 1)
        truncated[field_name] = truncate_field(text, budget, tokenizer)

    # Rebuild the prompt with truncated fields
    new_messages = rebuild_fn(row, truncated)
    new_prompt = tokenizer.apply_chat_template(conversation=new_messages, tokenize=False, enable_thinking=True)
    new_total = token_len(new_prompt, tokenizer)

    if new_total <= max_seq_length:
        return new_prompt, True

    # Still over — drop this example
    return None, False


# ---------------------------------------------------------------------------
# Parallel prompt preparation
# ---------------------------------------------------------------------------

_PROMPT_WORKER_TOKENIZER = None
_PROMPT_WORKER_MAX_SEQ_LENGTH = None
_PROMPT_WORKER_TASK = None


def _init_prompt_worker(model_name, max_seq_length, task_name):
    """Load one tokenizer per prompt-prep worker process."""
    global _PROMPT_WORKER_TOKENIZER
    global _PROMPT_WORKER_MAX_SEQ_LENGTH
    global _PROMPT_WORKER_TASK

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    _PROMPT_WORKER_TOKENIZER = tokenizer
    _PROMPT_WORKER_MAX_SEQ_LENGTH = max_seq_length
    _PROMPT_WORKER_TASK = TASKS[task_name]


def _build_prompt_chunk_worker(chunk):
    """Build prompts for one dataframe-record chunk in a worker process."""
    chunk_index, records = chunk
    if _PROMPT_WORKER_TOKENIZER is None or _PROMPT_WORKER_TASK is None:
        raise RuntimeError("Prompt worker was not initialized")

    prompts = []
    truncated_count = 0
    dropped_count = 0

    for row in records:
        prompt, was_truncated = build_prompt_with_truncation(
            row=row,
            tokenizer=_PROMPT_WORKER_TOKENIZER,
            max_seq_length=_PROMPT_WORKER_MAX_SEQ_LENGTH,
            build_messages_fn=_PROMPT_WORKER_TASK['build_messages'],
            get_truncatable_fn=_PROMPT_WORKER_TASK['get_truncatable'],
            rebuild_fn=_PROMPT_WORKER_TASK['rebuild'],
        )
        if prompt is None:
            dropped_count += 1
        else:
            if was_truncated:
                truncated_count += 1
            prompts.append(prompt)

    return chunk_index, prompts, truncated_count, dropped_count, len(records)


def _iter_dataframe_record_chunks(df, chunk_size):
    """Yield dataframe rows as picklable dict chunks."""
    for chunk_index, start in enumerate(range(0, len(df), chunk_size)):
        stop = min(start + chunk_size, len(df))
        yield chunk_index, df.iloc[start:stop].to_dict("records")


def _build_task_prompts_serial(df, task, tokenizer, max_seq_length):
    prompts = []
    truncated_count = 0
    dropped_count = 0
    total_rows = len(df)
    next_log_at = 5_000

    for i in range(total_rows):
        row = df.iloc[i]
        prompt, was_truncated = build_prompt_with_truncation(
            row=row,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            build_messages_fn=task['build_messages'],
            get_truncatable_fn=task['get_truncatable'],
            rebuild_fn=task['rebuild'],
        )
        if prompt is None:
            dropped_count += 1
        else:
            if was_truncated:
                truncated_count += 1
            prompts.append(prompt)

        rows_done = i + 1
        if rows_done == total_rows or rows_done >= next_log_at:
            print(
                f"  Built prompts for {rows_done}/{total_rows} rows "
                f"({len(prompts)} kept)...",
                flush=True,
            )
            next_log_at += 5_000

    return prompts, truncated_count, dropped_count


def build_task_prompts(
    task_name,
    df,
    tokenizer,
    max_seq_length,
    num_workers=1,
    chunk_size=128,
):
    """Build all prompts for one task, optionally in parallel."""
    task = TASKS[task_name]
    total_rows = len(df)
    if total_rows == 0:
        return [], 0, 0

    num_workers = max(1, num_workers)
    chunk_size = max(1, chunk_size)
    num_chunks = math.ceil(total_rows / chunk_size)
    actual_workers = min(num_workers, num_chunks)

    if actual_workers <= 1:
        return _build_task_prompts_serial(df, task, tokenizer, max_seq_length)

    model_name = tokenizer.name_or_path
    print(
        f"  Launching {actual_workers} prompt-prep workers "
        f"({num_chunks} chunks of up to {chunk_size} rows, start_method=spawn)...",
        flush=True,
    )

    prompts = []
    truncated_count = 0
    dropped_count = 0
    rows_done = 0
    next_log_at = 5_000
    max_in_flight = actual_workers * 2
    chunk_iter = _iter_dataframe_record_chunks(df, chunk_size)

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        actual_workers,
        initializer=_init_prompt_worker,
        initargs=(model_name, max_seq_length, task_name),
    ) as pool:
        in_flight = {}
        next_collect = 0

        def submit_one():
            try:
                chunk = next(chunk_iter)
            except StopIteration:
                return False
            chunk_index, _ = chunk
            in_flight[chunk_index] = pool.apply_async(
                _build_prompt_chunk_worker,
                (chunk,),
            )
            return True

        for _ in range(max_in_flight):
            if not submit_one():
                break

        while in_flight:
            chunk_index, chunk_prompts, chunk_truncated, chunk_dropped, chunk_rows = (
                in_flight.pop(next_collect).get()
            )
            if chunk_index != next_collect:
                raise RuntimeError(
                    f"Unexpected prompt chunk order: got {chunk_index}, "
                    f"expected {next_collect}"
                )

            prompts.extend(chunk_prompts)
            truncated_count += chunk_truncated
            dropped_count += chunk_dropped
            rows_done += chunk_rows
            next_collect += 1
            submit_one()

            if rows_done == total_rows or rows_done >= next_log_at:
                print(
                    f"  Built prompts for {rows_done}/{total_rows} rows "
                    f"({len(prompts)} kept)...",
                    flush=True,
                )
                while next_log_at <= rows_done:
                    next_log_at += 5_000

    return prompts, truncated_count, dropped_count


class TextParquetWriter:
    """Small append-style writer for one-column prompt parquet files."""

    def __init__(self, output_path, row_group_size=10_000):
        self.output_path = output_path
        self.row_group_size = row_group_size
        self.rows_written = 0
        self.writer = None

    def __enter__(self):
        output_dir = os.path.dirname(self.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        self.writer = pq.ParquetWriter(self.output_path, TEXT_SCHEMA)
        return self

    def write(self, texts):
        if not texts:
            return
        table = pa.Table.from_arrays(
            [pa.array(texts, type=pa.string())],
            schema=TEXT_SCHEMA,
        )
        self.writer.write_table(table, row_group_size=self.row_group_size)
        self.rows_written += len(texts)

    def write_record_batch(self, record_batch):
        if record_batch.num_rows == 0:
            return
        text_index = record_batch.schema.get_field_index("text")
        if text_index < 0:
            raise ValueError("Expected a 'text' column in record batch")
        table = pa.Table.from_arrays(
            [record_batch.column(text_index).cast(pa.string())],
            schema=TEXT_SCHEMA,
        )
        self.writer.write_table(table, row_group_size=self.row_group_size)
        self.rows_written += table.num_rows

    def __exit__(self, exc_type, exc, tb):
        if self.writer is not None:
            self.writer.close()


def _build_task_prompt_parquet_serial(
    df,
    task,
    tokenizer,
    max_seq_length,
    output_path,
    write_batch_size=128,
):
    truncated_count = 0
    dropped_count = 0
    kept_count = 0
    total_rows = len(df)
    next_log_at = 5_000
    pending_prompts = []

    with TextParquetWriter(output_path) as writer:
        for i in range(total_rows):
            row = df.iloc[i]
            prompt, was_truncated = build_prompt_with_truncation(
                row=row,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                build_messages_fn=task['build_messages'],
                get_truncatable_fn=task['get_truncatable'],
                rebuild_fn=task['rebuild'],
            )
            if prompt is None:
                dropped_count += 1
            else:
                if was_truncated:
                    truncated_count += 1
                pending_prompts.append(prompt)
                kept_count += 1

                if len(pending_prompts) >= write_batch_size:
                    writer.write(pending_prompts)
                    pending_prompts.clear()

            rows_done = i + 1
            if rows_done == total_rows or rows_done >= next_log_at:
                print(
                    f"  Built prompts for {rows_done}/{total_rows} rows "
                    f"({kept_count} kept)...",
                    flush=True,
                )
                next_log_at += 5_000

        writer.write(pending_prompts)
        pending_prompts.clear()

    return kept_count, truncated_count, dropped_count


def build_task_prompt_parquet(
    task_name,
    df,
    tokenizer,
    max_seq_length,
    output_path,
    num_workers=1,
    chunk_size=128,
    write_batch_size=128,
):
    """Build prompts for one task and write them directly to a parquet file."""
    task = TASKS[task_name]
    total_rows = len(df)
    if total_rows == 0:
        with TextParquetWriter(output_path):
            pass
        return 0, 0, 0

    num_workers = max(1, num_workers)
    chunk_size = max(1, chunk_size)
    write_batch_size = max(1, write_batch_size)
    num_chunks = math.ceil(total_rows / chunk_size)
    actual_workers = min(num_workers, num_chunks)

    if actual_workers <= 1:
        return _build_task_prompt_parquet_serial(
            df=df,
            task=task,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            output_path=output_path,
            write_batch_size=write_batch_size,
        )

    model_name = tokenizer.name_or_path
    print(
        f"  Launching {actual_workers} prompt-prep workers "
        f"({num_chunks} chunks of up to {chunk_size} rows, start_method=spawn)...",
        flush=True,
    )

    truncated_count = 0
    dropped_count = 0
    kept_count = 0
    rows_done = 0
    next_log_at = 5_000
    max_in_flight = actual_workers * 2
    chunk_iter = _iter_dataframe_record_chunks(df, chunk_size)
    pending_prompts = []

    ctx = mp.get_context("spawn")
    with TextParquetWriter(output_path) as writer:
        with ctx.Pool(
            actual_workers,
            initializer=_init_prompt_worker,
            initargs=(model_name, max_seq_length, task_name),
        ) as pool:
            in_flight = {}
            next_collect = 0

            def submit_one():
                try:
                    chunk = next(chunk_iter)
                except StopIteration:
                    return False
                chunk_index, _ = chunk
                in_flight[chunk_index] = pool.apply_async(
                    _build_prompt_chunk_worker,
                    (chunk,),
                )
                return True

            for _ in range(max_in_flight):
                if not submit_one():
                    break

            while in_flight:
                chunk_index, chunk_prompts, chunk_truncated, chunk_dropped, chunk_rows = (
                    in_flight.pop(next_collect).get()
                )
                if chunk_index != next_collect:
                    raise RuntimeError(
                        f"Unexpected prompt chunk order: got {chunk_index}, "
                        f"expected {next_collect}"
                    )

                pending_prompts.extend(chunk_prompts)
                truncated_count += chunk_truncated
                dropped_count += chunk_dropped
                kept_count += len(chunk_prompts)
                rows_done += chunk_rows
                next_collect += 1
                submit_one()

                if len(pending_prompts) >= write_batch_size:
                    writer.write(pending_prompts)
                    pending_prompts.clear()

                if rows_done == total_rows or rows_done >= next_log_at:
                    print(
                        f"  Built prompts for {rows_done}/{total_rows} rows "
                        f"({kept_count} kept)...",
                        flush=True,
                    )
                    while next_log_at <= rows_done:
                        next_log_at += 5_000

        writer.write(pending_prompts)
        pending_prompts.clear()

    return kept_count, truncated_count, dropped_count


def sample_task_rows(df, sample_rows, seed, task_name):
    """Optionally sample rows from one task dataframe."""
    if sample_rows is None:
        return df

    total_rows = len(df)
    if total_rows <= sample_rows:
        print(
            f"  Sample rows requested for {task_name}: {sample_rows}; "
            f"using all {total_rows} rows",
            flush=True,
        )
        return df.reset_index(drop=True)

    print(
        f"  Sampling {sample_rows}/{total_rows} rows for {task_name} "
        f"(seed={seed})...",
        flush=True,
    )
    return df.sample(n=sample_rows, random_state=seed).reset_index(drop=True)


def parquet_row_count(path):
    """Return the number of rows in a parquet file without reading its data."""
    return pq.ParquetFile(path).metadata.num_rows


def balance_task_parquet(input_path, output_path, target, batch_size=10_000):
    """Balance one task parquet and write the result without keeping other tasks."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    raw_count = parquet_row_count(input_path)

    if raw_count == 0:
        with TextParquetWriter(output_path):
            pass
        return raw_count, 0, 0.0

    rows_remaining = target
    replication = 1.0 if raw_count >= target else math.ceil(target / raw_count)

    with TextParquetWriter(output_path) as writer:
        while rows_remaining > 0:
            wrote_this_pass = 0
            pf = pq.ParquetFile(input_path)
            for record_batch in pf.iter_batches(
                batch_size=batch_size,
                columns=["text"],
            ):
                if rows_remaining <= 0:
                    break
                if record_batch.num_rows > rows_remaining:
                    record_batch = record_batch.slice(0, rows_remaining)
                writer.write_record_batch(record_batch)
                rows_remaining -= record_batch.num_rows
                wrote_this_pass += record_batch.num_rows

            if wrote_this_pass == 0:
                break

    final_count = target - rows_remaining
    gc.collect()
    return raw_count, final_count, replication


def combine_and_shuffle_task_parquets(
    task_paths,
    combined_path,
    seed,
    bucket_count=256,
    batch_size=10_000,
):
    """Combine task parquets with a disk-backed row shuffle."""
    bucket_count = max(1, bucket_count)
    batch_size = max(1, batch_size)
    bucket_dir = f"{combined_path}.shuffle_buckets"
    if os.path.exists(bucket_dir):
        shutil.rmtree(bucket_dir)
    os.makedirs(bucket_dir, exist_ok=True)

    bucket_paths = [
        os.path.join(bucket_dir, f"bucket-{idx:05d}.parquet")
        for idx in range(bucket_count)
    ]
    rng = random.Random(seed)
    total_rows = 0

    print(
        f"Partitioning rows into {bucket_count} shuffle buckets "
        f"(seed={seed})..."
    )
    bucket_writers = []
    try:
        for bucket_path in bucket_paths:
            bucket_writers.append(TextParquetWriter(bucket_path).__enter__())

        for task_name, path in task_paths:
            print(f"  Streaming balanced {task_name} prompts from {path}...")
            task_rows = 0
            pf = pq.ParquetFile(path)
            for record_batch in pf.iter_batches(
                batch_size=batch_size,
                columns=["text"],
            ):
                text_index = record_batch.schema.get_field_index("text")
                texts = record_batch.column(text_index).to_pylist()
                bucketed_texts = [[] for _ in range(bucket_count)]
                for text in texts:
                    bucketed_texts[rng.randrange(bucket_count)].append(text)

                for bucket_idx, bucket_texts in enumerate(bucketed_texts):
                    bucket_writers[bucket_idx].write(bucket_texts)

                task_rows += len(texts)
                total_rows += len(texts)
                if task_rows % (batch_size * 10) < batch_size:
                    print(
                        f"    Partitioned {task_rows} {task_name} rows...",
                        flush=True,
                    )
            print(f"    Partitioned {task_rows} {task_name} rows total")
    finally:
        for writer in bucket_writers:
            writer.__exit__(None, None, None)

    print(f"Combined dataset: {total_rows} rows")
    print(f"Writing shuffled combined parquet to {combined_path}...")
    bucket_order = list(range(bucket_count))
    random.Random(seed + 1).shuffle(bucket_order)

    rows_written = 0
    with TextParquetWriter(combined_path) as writer:
        for bucket_idx in bucket_order:
            bucket_path = bucket_paths[bucket_idx]
            bucket_rows = parquet_row_count(bucket_path)
            if bucket_rows == 0:
                continue

            table = pq.read_table(bucket_path, columns=["text"])
            row_order = list(range(table.num_rows))
            random.Random(seed + 10_000 + bucket_idx).shuffle(row_order)
            shuffled_table = table.take(pa.array(row_order, type=pa.int64()))
            writer.writer.write_table(
                shuffled_table.cast(TEXT_SCHEMA),
                row_group_size=writer.row_group_size,
            )
            writer.rows_written += shuffled_table.num_rows
            rows_written += shuffled_table.num_rows

            del table, shuffled_table, row_order
            gc.collect()

            if rows_written % (batch_size * 10) < bucket_rows:
                print(
                    f"  Wrote {rows_written}/{total_rows} shuffled rows...",
                    flush=True,
                )

    shutil.rmtree(bucket_dir)
    gc.collect()
    return rows_written


def positive_int(value):
    """Parse a positive integer CLI argument."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


# ---------------------------------------------------------------------------
# Streaming tokenization — writes directly to Arrow on disk
# ---------------------------------------------------------------------------

def _find_last_subsequence(seq, subseq):
    """Return the start index of the last occurrence of subseq in seq, or -1.

    Uses struct packing + bytes.rfind for C-level search speed instead of
    a pure-Python reverse linear scan.
    """
    if not subseq:
        return len(seq)
    seq_bytes = struct.pack(f'{len(seq)}I', *seq)
    sub_bytes = struct.pack(f'{len(subseq)}I', *subseq)
    pos = seq_bytes.rfind(sub_bytes)
    if pos < 0:
        return -1
    return pos // 4


def get_assistant_header_ids(tokenizer):
    """Return token ids for the assistant-response header in this tokenizer.

    The exact chat-template marker is model-specific.  Derive it from
    ``add_generation_prompt`` instead of hard-coding a Llama-style header.
    """
    probe_prefix = [
        {"role": "system", "content": "__mask_probe_system__"},
        {"role": "user", "content": "__mask_probe_user__"},
    ]
    probe_full = probe_prefix + [
        {"role": "assistant", "content": "__mask_probe_assistant__"},
    ]

    def render(messages, **kwargs):
        return tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=False,
            enable_thinking=True,
            **kwargs,
        )

    without_header = render(probe_prefix, add_generation_prompt=False)
    with_header = render(probe_prefix, add_generation_prompt=True)

    candidates = []
    if with_header.startswith(without_header):
        candidates.append(with_header[len(without_header):])

    # Fallbacks for common chat templates, in case a tokenizer's generation
    # prompt is not a strict text suffix of the non-generation rendering.
    candidates.extend([
        "<|im_start|>assistant\n",
        "<|start_header_id|>assistant<|end_header_id|>\n\n",
    ])

    full_ids = tokenizer(render(probe_full), add_special_tokens=False).input_ids
    for header_text in candidates:
        if not header_text:
            continue
        header_ids = tokenizer(header_text, add_special_tokens=False).input_ids
        if header_ids and _find_last_subsequence(full_ids, header_ids) >= 0:
            return header_ids, header_text

    raise ValueError(
        "Could not derive assistant header token ids from tokenizer chat template"
    )


def _tokenize_range_worker(args):
    """Multiprocessing worker: tokenize row groups from a parquet file.

    Each worker opens the source parquet independently and reads only its
    assigned row groups, so no text data is loaded into the parent process.
    """
    (source_parquet, row_group_indices, model_name, max_seq_length,
     arrow_path, header_ids, batch_size, worker_idx, num_workers) = args
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pf = pq.ParquetFile(source_parquet)
    header_len = len(header_ids)
    writer = ArrowWriter(path=arrow_path)
    masked_count = 0
    unmasked_count = 0
    total = sum(pf.metadata.row_group(i).num_rows for i in row_group_indices)
    rows_done = 0
    log_prefix = f"[Worker {worker_idx + 1}/{num_workers}] "

    for rg_idx in row_group_indices:
        texts = pf.read_row_group(rg_idx, columns=["text"]).column("text").to_pylist()

        for batch_start in range(0, len(texts), batch_size):
            batch_texts = texts[batch_start:batch_start + batch_size]
            tokenized = tokenizer(batch_texts, max_length=max_seq_length, truncation=True)

            batch_labels = []
            for input_ids in tokenized["input_ids"]:
                labels = list(input_ids)
                idx = _find_last_subsequence(input_ids, header_ids)
                if idx >= 0:
                    mask_end = idx + header_len
                    labels[:mask_end] = [-100] * mask_end
                    masked_count += 1
                else:
                    unmasked_count += 1
                batch_labels.append(labels)

            writer.write_batch({
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "labels": batch_labels,
            })

            rows_done += len(batch_texts)
            if rows_done % (batch_size * 10) < batch_size:
                print(f"  {log_prefix}Tokenized {rows_done}/{total} examples...", flush=True)

    num_examples, num_bytes = writer.finalize()
    print(
        f"  {log_prefix}Wrote {num_examples} examples ({num_bytes / 1e6:.1f} MB)",
        flush=True,
    )
    return worker_idx, num_examples, num_bytes, masked_count, unmasked_count


def streaming_tokenize(source_parquet, tokenizer, max_seq_length, output_path,
                       batch_size=256, num_workers=1):
    """
    Tokenize texts from *source_parquet* to Arrow file(s) on disk.

    Creates a ``labels`` column with prompt tokens masked to -100 so the loss
    is computed only on the assistant response (proper SFT behaviour).

    When *num_workers* > 1, the parquet file's row groups are distributed
    across workers.  Each worker opens the file independently and reads only
    its assigned row groups — no text data is loaded into the parent process.
    The resulting multi-file dataset is compatible with
    ``Dataset.load_from_disk()``.
    """
    os.makedirs(output_path, exist_ok=True)

    # Mark the dataset incomplete until all shards and metadata are rewritten.
    for metadata_name in ("state.json", "dataset_info.json"):
        metadata_path = os.path.join(output_path, metadata_name)
        if os.path.exists(metadata_path):
            os.remove(metadata_path)

    # Token sequence that marks the start of the assistant's response.
    # Everything up to and including this header is masked in labels.
    header_ids, assistant_header = get_assistant_header_ids(tokenizer)
    print(f"  Assistant header for masking: {assistant_header!r}")

    pf = pq.ParquetFile(source_parquet)
    total_rows = pf.metadata.num_rows
    num_workers = max(1, min(num_workers, total_rows))

    if num_workers <= 1:
        # ---- Single-process path: stream from parquet, tokenize in chunks ----
        arrow_path = os.path.join(output_path, "data-00000-of-00001.arrow")
        header_len = len(header_ids)
        writer = ArrowWriter(path=arrow_path)
        masked_count = 0
        unmasked_count = 0
        rows_done = 0

        for record_batch in pf.iter_batches(batch_size=batch_size, columns=["text"]):
            texts = record_batch.column("text").to_pylist()
            tokenized = tokenizer(texts, max_length=max_seq_length, truncation=True)

            batch_labels = []
            for input_ids in tokenized["input_ids"]:
                labels = list(input_ids)
                idx = _find_last_subsequence(input_ids, header_ids)
                if idx >= 0:
                    mask_end = idx + header_len
                    labels[:mask_end] = [-100] * mask_end
                    masked_count += 1
                else:
                    unmasked_count += 1
                batch_labels.append(labels)

            writer.write_batch({
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "labels": batch_labels,
            })

            rows_done += len(texts)
            if rows_done % (batch_size * 10) < batch_size:
                print(f"  Tokenized {rows_done}/{total_rows} examples...")

        num_examples, num_bytes = writer.finalize()
        data_files = [{"filename": "data-00000-of-00001.arrow"}]
    else:
        # ---- Multi-process path ----
        # Distribute parquet row groups across workers.  Each worker opens
        # the file independently and reads only its assigned row groups,
        # so no text data is loaded into the parent process.
        model_name = tokenizer.name_or_path
        num_row_groups = pf.metadata.num_row_groups

        if num_row_groups < num_workers:
            print(f"  Warning: only {num_row_groups} row groups in parquet, "
                  f"capping workers from {num_workers} to {num_row_groups}")
            num_workers = num_row_groups

        # Round-robin assignment of row groups to workers
        rg_assignments = [[] for _ in range(num_workers)]
        for rg_idx in range(num_row_groups):
            rg_assignments[rg_idx % num_workers].append(rg_idx)

        worker_args = []
        for i in range(num_workers):
            if not rg_assignments[i]:
                break
            arrow_path = os.path.join(
                output_path, f"data-{i:05d}-of-{num_workers:05d}.arrow",
            )
            worker_args.append((
                source_parquet, rg_assignments[i], model_name,
                max_seq_length, arrow_path, header_ids, batch_size,
                i, num_workers,
            ))

        actual_workers = len(worker_args)
        print(
            f"  Launching {actual_workers} tokenization workers "
            f"({num_row_groups} row groups, start_method=spawn)...",
            flush=True,
        )

        # Spawn avoids inheriting tokenizer/native-thread state from the parent.
        ctx = mp.get_context("spawn")
        results = []
        num_examples = 0
        num_bytes = 0
        masked_count = 0
        unmasked_count = 0

        with ctx.Pool(actual_workers) as pool:
            for result in pool.imap_unordered(
                _tokenize_range_worker, worker_args, chunksize=1,
            ):
                worker_idx, worker_examples, worker_bytes, worker_masked, worker_unmasked = result
                results.append(result)
                num_examples += worker_examples
                num_bytes += worker_bytes
                masked_count += worker_masked
                unmasked_count += worker_unmasked
                print(
                    f"  Collected worker {worker_idx + 1}/{num_workers} "
                    f"({len(results)}/{actual_workers}); "
                    f"aggregate {num_examples} examples ({num_bytes / 1e6:.1f} MB)",
                    flush=True,
                )

        data_files = [
            {"filename": f"data-{i:05d}-of-{num_workers:05d}.arrow"}
            for i in range(actual_workers)
        ]

    print(f"  Wrote {num_examples} examples ({num_bytes / 1e6:.1f} MB)")
    print(f"  Prompt-masked: {masked_count}, unmasked (fallback): {unmasked_count}")

    # Write minimal metadata so Dataset.load_from_disk() works
    with open(os.path.join(output_path, "state.json"), "w") as f:
        json.dump({
            "_data_files": data_files,
            "_fingerprint": "streaming_tokenized",
            "_format_columns": None,
            "_format_kwargs": {},
            "_format_type": None,
            "_output_all_columns": False,
            "_split": None,
        }, f, indent=2)

    with open(os.path.join(output_path, "dataset_info.json"), "w") as f:
        json.dump({}, f)

    return num_examples


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_boilerplate(data_dir):
    path = os.path.join(data_dir, 'boilerplate_checks', 'final_boilerplate_checks.parquet')
    print(f"  Loading {path}...")
    df = pd.read_parquet(path)
    print(f"  Loaded {len(df)} boilerplate records")
    return df


def load_summarization(data_dir):
    sum_path = os.path.join(data_dir, 'patient_serial_summaries.parquet')
    chunk_path = os.path.join(data_dir, 'summary_shards', 'prepared_chunks.parquet')
    print(f"  Loading {sum_path}...")
    serial_summaries = pd.read_parquet(sum_path)
    print(f"  Loaded {len(serial_summaries)} summary records")

    print(f"  Loading {chunk_path}...")
    prepared_chunks = pd.read_parquet(chunk_path)
    print(f"  Loaded {len(prepared_chunks)} chunk records")

    prepared_chunks = prepared_chunks.rename(columns={
        'patient_id': 'pseudo_mrn',
        'local_idx': 'chunk_index',
    })
    serial_summaries = serial_summaries.merge(
        prepared_chunks[['pseudo_mrn', 'chunk_index', 'chunk_text']],
        on=['pseudo_mrn', 'chunk_index'],
        how='left',
    )
    serial_summaries = serial_summaries.sort_values(['pseudo_mrn', 'chunk_index']).reset_index(drop=True)
    print(f"  After merge: {len(serial_summaries)} records ({serial_summaries['chunk_text'].notna().sum()} with chunk text)")
    return serial_summaries


def load_trialchecks(data_dir):
    KEEP_COLS = ['patient_summary', 'this_space', 'trialcheck_llm_reasoning',
                 'trialcheck_llm_response',
                 'eligibility_result', 'eligibility_verdict']
    rename = {
        'trialcheck_llama_reasoning': 'trialcheck_llm_reasoning',
        'trialcheck_llama_response': 'trialcheck_llm_response',
        'llama_reasoning': 'trialcheck_llm_reasoning',
        'llama_response': 'trialcheck_llm_response',
    }

    def load_and_select(path, rename_cols=None):
        df = pd.read_parquet(path)
        if rename_cols:
            df = df.rename(columns=rename_cols)
        available = [c for c in KEEP_COLS if c in df.columns]
        return df[available]

    files = [
        ('space_specific_eligibility_checks.parquet', None),
        ('round1_patientcentric_checks/top_cohorts_checked_round1.parquet', rename),
        ('round2_patientcentric_checks/top_cohorts_checked_round2.parquet', rename),
        ('round3_patientcentric_checks/top_cohorts_checked_round3.parquet', rename),
        ('round1_trialcentric_checks/top_patients_checked_round1.parquet', rename),
        ('round2_trialcentric_checks/top_patients_checked_round2.parquet', rename),
        ('round3_trialcentric_checks/top_patients_checked_round3.parquet', rename),
    ]

    frames = []
    for rel_path, ren in files:
        path = os.path.join(data_dir, rel_path)
        print(f"  Loading {path}...")
        df = load_and_select(path, ren)
        print(f"    {len(df)} records")
        frames.append(df)

    allchecks = pd.concat(frames, ignore_index=True)
    print(f"  Total trial check records: {len(allchecks)}")

    firstchecks = allchecks.groupby(['patient_summary', 'this_space']).first().reset_index()
    print(f"  After deduplication: {len(firstchecks)} records")
    return firstchecks


def load_trialspaces(data_dir):
    path = os.path.join(data_dir, 'trial_space_lineitems.csv')
    print(f"  Loading {path}...")
    df = pd.read_csv(path)
    print(f"  Loaded {len(df)} trial space records")
    return df


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASKS = {
    'boilerplate': {
        'loader': load_boilerplate,
        'build_messages': build_boilerplate_messages,
        'get_truncatable': boilerplate_truncatable_fields,
        'rebuild': boilerplate_rebuild,
    },
    'summarization': {
        'loader': load_summarization,
        'build_messages': build_summarization_messages,
        'get_truncatable': summarization_truncatable_fields,
        'rebuild': summarization_rebuild,
    },
    'trialchecks': {
        'loader': load_trialchecks,
        'build_messages': build_trialcheck_messages,
        'get_truncatable': trialcheck_truncatable_fields,
        'rebuild': trialcheck_rebuild,
    },
    'trialspaces': {
        'loader': load_trialspaces,
        'build_messages': build_trialspace_messages,
        'get_truncatable': trialspace_truncatable_fields,
        'rebuild': trialspace_rebuild,
    },
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Consolidated oncoreasoning training data preparation"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help=f"Base data directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        required=True,
        help="Max sequence length for fine-tuning (required)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="google/gemma-4-e2b-it",
        help="tokenizer. default: google/gemma-4-e2b)",
    )
    parser.add_argument(
        "--balance-target",
        type=str,
        default="max",
        help='Target example count per task for balancing. "max" = match largest task, or an integer (default: max)',
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Shuffle seed (default: 42)",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=list(TASKS.keys()),
        default=list(TASKS.keys()),
        help="Which tasks to include (default: all four)",
    )
    parser.add_argument(
        "--sample-rows",
        type=positive_int,
        default=None,
        help="Randomly sample up to this many rows from each selected task dataset before prompt preparation (default: all rows)",
    )
    parser.add_argument(
        "--writer-batch-size",
        type=int,
        default=1000,
        help="Writer batch size for tokenization to reduce memory usage (default: 1000)",
    )
    parser.add_argument(
        "--prep-workers",
        type=int,
        default=None,
        help="Number of parallel workers for prompt/parquet preparation before tokenization (default: --num-workers)",
    )
    parser.add_argument(
        "--prep-chunk-size",
        type=int,
        default=128,
        help="Rows per prompt-prep worker task (default: 128)",
    )
    parser.add_argument(
        "--balance-batch-size",
        type=positive_int,
        default=10_000,
        help="Rows per streaming balance batch (default: 10000)",
    )
    parser.add_argument(
        "--shuffle-batch-size",
        type=positive_int,
        default=10_000,
        help="Rows per streaming shuffle partition batch (default: 10000)",
    )
    parser.add_argument(
        "--shuffle-buckets",
        type=positive_int,
        default=256,
        help="Number of disk-backed shuffle buckets (default: 256)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_WORKER_COUNT,
        help="Number of parallel workers for tokenization (default: CPU count minus one)",
    )
    parser.add_argument(
        "--change-to-think",
        action="store_true",
        default=False,
        help="Deprecated no-op; reasoning/final outputs are normalized before chat templating",
    )
    parser.add_argument(
        "--retokenize",
        action="store_true",
        default=False,
        help="Reuse existing all_training_data.parquet and regenerate only the tokenized dataset",
    )
    parser.add_argument(
        "--reuse-balanced-task-parquets",
        action="store_true",
        default=False,
        help="Skip prompt building/balancing and combine existing balanced task parquets",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.change_to_think:
        print("Note: --change-to-think is deprecated and no longer changes output.")

    prep_workers = args.prep_workers if args.prep_workers is not None else args.num_workers
    prep_workers = max(1, prep_workers)
    prep_chunk_size = max(1, args.prep_chunk_size)

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    combined_path = os.path.join(args.output_dir, 'all_training_data.parquet')
    tokenized_path = os.path.join(args.output_dir, 'tokenized_training_data.dataset')
    task_parquet_dir = os.path.join(args.output_dir, 'task_parquets')
    balanced_task_parquet_dir = os.path.join(args.output_dir, 'balanced_task_parquets')

    # Retokenization is explicit-only so stale prompt parquet files are not
    # silently reused after prompt-construction changes.
    if args.retokenize:
        if args.sample_rows is not None:
            print("Note: --sample-rows is ignored with --retokenize.")
        if not os.path.exists(combined_path):
            raise FileNotFoundError(f"Cannot retokenize; missing {combined_path}")
        print(f"\nRetokenizing from existing {combined_path}.")
        print("Skipping data loading/building — jumping straight to tokenization.")
        row_count = pq.ParquetFile(combined_path).metadata.num_rows
        print(f"Source parquet has {row_count} rows")

        print(f"\nTokenizing with max_length={args.max_seq_length} (streaming to disk)...")
        num_examples = streaming_tokenize(
            source_parquet=combined_path,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            output_path=tokenized_path,
            batch_size=args.writer_batch_size,
            num_workers=args.num_workers,
        )
        print(f"Total examples: {num_examples}")
        print("Done!")
        return

    task_parquet_paths = {}
    counts = {}
    balanced_task_paths = []
    if args.reuse_balanced_task_parquets:
        print(f"\n{'='*60}")
        print("Reusing balanced task parquets")
        print(f"{'='*60}")
        if args.sample_rows is not None:
            print("Note: --sample-rows is ignored with --reuse-balanced-task-parquets.")

        for name in args.tasks:
            balanced_path = os.path.join(balanced_task_parquet_dir, f'{name}.parquet')
            if not os.path.exists(balanced_path):
                raise FileNotFoundError(
                    f"Cannot reuse balanced task parquets; missing {balanced_path}"
                )
            row_count = parquet_row_count(balanced_path)
            print(f"  {name}: {row_count} rows from {balanced_path}")
            if row_count > 0:
                balanced_task_paths.append((name, balanced_path))

        if not balanced_task_paths:
            raise ValueError("No reusable balanced task parquets had any rows.")
    else:
        # ---- Step 1 & 2: Load data and build prompts with truncation ----
        for task_name in args.tasks:
            task = TASKS[task_name]
            task_parquet_path = os.path.join(task_parquet_dir, f'{task_name}.parquet')
            print(f"\n{'='*60}")
            print(f"Task: {task_name}")
            print(f"{'='*60}")

            print("Loading data...")
            df = task['loader'](args.data_dir)
            df = sample_task_rows(
                df=df,
                sample_rows=args.sample_rows,
                seed=args.seed,
                task_name=task_name,
            )

            print(f"Building prompts (max_seq_length={args.max_seq_length})...")
            kept_count, truncated_count, dropped_count = build_task_prompt_parquet(
                task_name=task_name,
                df=df,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                output_path=task_parquet_path,
                num_workers=prep_workers,
                chunk_size=prep_chunk_size,
            )

            print(f"  Built {kept_count} prompts")
            print(f"  Truncated: {truncated_count}")
            print(f"  Dropped (too long): {dropped_count}")
            print(f"  Saved task parquet: {task_parquet_path}")
            task_parquet_paths[task_name] = task_parquet_path

            del df
            gc.collect()

        # ---- Step 3: Balance tasks ----
        print(f"\n{'='*60}")
        print("Balancing tasks")
        print(f"{'='*60}")

        counts = {
            name: parquet_row_count(path)
            for name, path in task_parquet_paths.items()
        }
        if not counts or max(counts.values()) == 0:
            raise ValueError("No prompts were generated for the selected tasks.")

        if args.balance_target == "max":
            target = max(counts.values())
            print(f"Balance target: max = {target}")
        else:
            target = int(args.balance_target)
            print(f"Balance target: {target}")

        for name in args.tasks:
            raw_path = task_parquet_paths[name]
            raw_count = counts[name]
            if raw_count == 0:
                print(f"  {name}: 0 examples - skipping")
                continue

            balanced_path = os.path.join(balanced_task_parquet_dir, f'{name}.parquet')
            raw_count, final_count, replication = balance_task_parquet(
                input_path=raw_path,
                output_path=balanced_path,
                target=target,
                batch_size=args.balance_batch_size,
            )
            balanced_task_paths.append((name, balanced_path))
            print(f"  {name}: {raw_count} raw -> x{replication} -> {final_count} final")

    # ---- Step 4: Combine, shuffle, tokenize, save ----
    print(f"\n{'='*60}")
    print("Combining, shuffling, and saving")
    print(f"{'='*60}")

    rows_written = combine_and_shuffle_task_parquets(
        task_paths=balanced_task_paths,
        combined_path=combined_path,
        seed=args.seed,
        bucket_count=args.shuffle_buckets,
        batch_size=args.shuffle_batch_size,
    )
    print(f"Saved {rows_written} rows")

    # Tokenize — stream directly from the parquet we just saved
    print(f"\nTokenizing with max_length={args.max_seq_length} (streaming to disk)...")
    del balanced_task_paths, task_parquet_paths, counts
    gc.collect()

    num_examples = streaming_tokenize(
        source_parquet=combined_path,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        output_path=tokenized_path,
        batch_size=args.writer_batch_size,
        num_workers=args.num_workers,
    )

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Max sequence length: {args.max_seq_length}")
    print(f"Total examples: {num_examples}")
    print(f"Output parquet: {combined_path}")
    print(f"Tokenized dataset: {tokenized_path}")
    print("Done!")


if __name__ == "__main__":
    main()
