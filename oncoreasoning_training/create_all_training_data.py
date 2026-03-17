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
import json
import math
import multiprocessing as mp
import os
import re
import struct

import pandas as pd
import pyarrow.parquet as pq
from datasets.arrow_writer import ArrowWriter
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Constants: prompt templates copied verbatim from the original scripts
# ---------------------------------------------------------------------------

TRIALSPACE_PROMPT_HEADER = (
    "You are an expert clinical oncologist with an encyclopedic knowledge of cancer and its treatments.\n"
    "Your job is to review a clinical trial document and extract a list of structured clinical spaces that are eligible for that trial.\n"
    "A clinical space is defined as a unique combination of patient age range, sex (if any sex criteria), cancer primary site, histology, which treatments a patient must have received, "
    "which treatments a patient must not have received, cancer burden (eg presence of metastatic disease; this also includes cancer type-specific prognostic scores, risk indices, or categories), tumor biomarkers (such as "
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
    "If a concept is not relevant, such as if there are no prior treatents required, simply output NA for that concept.\n"
    "CRITICAL: Anytime you provide a list for a particular concept, you must be completely clear on whether \"or\" versus \"and\" logic applies "
    "to the list. For example, do not output \"EGFR L858R mutant, TP53 mutant\"; if both are required, output \"EGFR L858R mutant and TP53 mutant\". "
    "As another example, do not output \"ER+, PR+\"; if the patient can have either an ER or a PR positive tumor, output \"ER+ or PR+\".\n"
    "NEVER put a newline within a single trial space.\n"
    "After you output the trial spaces, output a newline, then the text \"Boilerplate exclusions:\" VERBATIM, then another newline.\n"
    "Then, list exclusion criteria described in the trial text that are unrelated to the trial space definitions. Such exclusions tend to be common "
    "to clinical trials in general.\n"
    "Common boilerplate exclusion criteria include a history of pneumonitis, heart failure, renal dysfunction, liver dysfunction, uncontrolled brain "
    "metastases, HIV or hepatitis, and poor performance status.\n"
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


def apply_think_tags(text: str) -> str:
    """Replace gpt-oss-120b reasoning markers with <think>/<​/think> tags.

    Replaces 'assistantfinal' -> '</think>' first, then
    'assistant' (not inside chat template headers) -> '<think>'.
    """
    text = text.replace('assistantfinal', '</think>')
    text = re.sub(r'(?<!\|start_header_id\|>)assistant(?!final)', '<think>', text)
    return text


# ---------------------------------------------------------------------------
# Per-task prompt builders (return messages list + truncatable field info)
# ---------------------------------------------------------------------------

def build_boilerplate_messages(row):
    """Build chat messages for a boilerplate check example."""
    patient_bp = row['patient_boilerplate_text']
    trial_bp = row['trial_boilerplate_text']
    answer = row['boilerplate_check_llm_response']

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

    reasoning = row['new_summary_reasoning']
    summary = row['new_summary']
    full_response = reasoning + "assistantfinal" + summary

    prior_summary_text = prior_summary if prior_summary else "None - this is the first segment for this patient"

    user_content = f"""You are an experienced clinical oncology history summarization bot.

You are maintaining a running summary of a patient's cancer history based on their electronic health record.
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
Current extent: (patient's current extent (localized, advanced, metastatic, etc); this is also where tumor markers for following disease status over time, such as CEA or PSA, should be documented if relevant)
Biomarkers: (genomic results, protein expression, etc, relevant or potentially relevant for informing treatment selection. Err on the side of including all possible biomarkers, including all IHC results, all positive genomic findings, and any pertinent negative genomic findings)
Treatment history: (surgery, radiation, chemotherapy/targeted therapy/immunotherapy, etc, including start and stop dates, and best response if noted. Treatment history should be provided chronologically.)
Boilerplate: (any history of conditions that might meet common "boilerplate" exclusion criteria for clinical trials, such as uncontrolled brain metastases, lack of measurable disease, congestive heart failure, pneumonitis, renal dysfunction, liver dysfunction, HIV or hepatitis infection, etc)
Clearly separate the "boilerplate" section by labeling it "Boilerplate: " before describing any such conditions.
--(end of sections)

Do not consider localized basal cell or squamous carcinomas of the skin, or colon polyps, to be cancers for your purposes.
Do not include the patient's name, but do include relevant dates whenever documented.
If a patient has a history of more than one cancer, document the cancers one at a time. List the currently or most recently active cancer first, followed by any prior cancers. Within each cancer, events should be in chronological order.
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
Boilerplate:
No evidence of common boilerplate exclusion criteria

The following are the patient's data.
---
PRIOR SUMMARY:
{prior_summary_text}

NEXT CLINICAL RECORD SEGMENT (covering {first_date} to {last_date}):
{chunk_text}
---
Now, write your updated summary, or if there is no new relevant information, output the prior summary exactly as it was.
If any information is still relevant but is unchanged, just restate it in the updated summary, but do NOT state "no change" or similar - just produce the updated summary text as if you were writing it fresh, incorporating any new information but keeping relevant old information, without calling out what changed vs what stayed the same from the prior summary.
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
    answer = row['trialcheck_llm_response']

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
    answer = row['space_reasoning_and_output']

    messages = [
        {'role': 'system', 'content': """
        Reasoning: high.
        """},
        {'role': 'user', 'content': TRIALSPACE_PROMPT_HEADER + "\n" + trial_text + "\n" + TRIALSPACE_PROMPT_SUFFIX},
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
    prompt = tokenizer.apply_chat_template(conversation=messages, tokenize=False)
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
    new_prompt = tokenizer.apply_chat_template(conversation=new_messages, tokenize=False)
    new_total = token_len(new_prompt, tokenizer)

    if new_total <= max_seq_length:
        return new_prompt, True

    # Still over — drop this example
    return None, False


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
                print(f"  {log_prefix}Tokenized {rows_done}/{total} examples...")

    num_examples, num_bytes = writer.finalize()
    print(f"  {log_prefix}Wrote {num_examples} examples ({num_bytes / 1e6:.1f} MB)")
    return num_examples, num_bytes, masked_count, unmasked_count


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

    # Token sequence that marks the start of the assistant's response.
    # Everything up to and including this header is masked in labels.
    assistant_header = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    header_ids = tokenizer.encode(assistant_header, add_special_tokens=False)

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
        print(f"  Launching {actual_workers} tokenization workers "
              f"({num_row_groups} row groups)...")
        with mp.Pool(actual_workers) as pool:
            results = pool.map(_tokenize_range_worker, worker_args)

        num_examples = sum(r[0] for r in results)
        num_bytes = sum(r[1] for r in results)
        masked_count = sum(r[2] for r in results)
        unmasked_count = sum(r[3] for r in results)

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
    sum_path = os.path.join(data_dir, 'patient_serial_summaries_20K_chunks.parquet')
    chunk_path = os.path.join(data_dir, 'summary_shards_20K_chunks', 'prepared_chunks.parquet')
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
    KEEP_COLS = ['patient_summary', 'this_space', 'trialcheck_llm_response',
                 'eligibility_result', 'eligibility_verdict']
    rename = {'trialcheck_llama_response': 'trialcheck_llm_response'}

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
        default="../../data/no_phi",
        help="Base data directory (default: ../../data/no_phi)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../../data/no_phi/oncoreasoning_training_data",
        help="Output directory (default: ../../data/no_phi/oncoreasoning_training_data)",
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
        default="LiquidAI/LFM2.5-1.2B-Thinking",
        help="Tokenizer model (default: LiquidAI/LFM-2.5-1.2B-Thinking)",
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
        "--writer-batch-size",
        type=int,
        default=1000,
        help="Writer batch size for tokenization to reduce memory usage (default: 1000)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of parallel workers for tokenization (default: all CPUs)",
    )
    parser.add_argument(
        "--change-to-think",
        action="store_true",
        default=False,
        help="Replace reasoning markers (assistant/assistantfinal) with <think>/</think> tags",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    combined_path = os.path.join(args.output_dir, 'all_training_data.parquet')
    tokenized_path = os.path.join(args.output_dir, 'tokenized_training_data.dataset')

    # If the final parquet exists but the tokenized dataset doesn't, skip to tokenization
    if os.path.exists(combined_path) and not os.path.exists(tokenized_path):
        print(f"\nFound existing {combined_path} but no tokenized dataset.")
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

    # ---- Step 1 & 2: Load data and build prompts with truncation ----
    task_prompts = {}
    for task_name in args.tasks:
        task = TASKS[task_name]
        print(f"\n{'='*60}")
        print(f"Task: {task_name}")
        print(f"{'='*60}")

        print("Loading data...")
        df = task['loader'](args.data_dir)

        print(f"Building prompts (max_seq_length={args.max_seq_length})...")
        prompts = []
        truncated_count = 0
        dropped_count = 0

        for i in range(len(df)):
            row = df.iloc[i]
            prompt, was_truncated = build_prompt_with_truncation(
                row=row,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                build_messages_fn=task['build_messages'],
                get_truncatable_fn=task['get_truncatable'],
                rebuild_fn=task['rebuild'],
            )
            if prompt is None:
                dropped_count += 1
            else:
                if was_truncated:
                    truncated_count += 1
                if args.change_to_think:
                    prompt = apply_think_tags(prompt)
                prompts.append(prompt)

        print(f"  Built {len(prompts)} prompts")
        print(f"  Truncated: {truncated_count}")
        print(f"  Dropped (too long): {dropped_count}")
        task_prompts[task_name] = prompts

    # ---- Step 3: Balance tasks ----
    print(f"\n{'='*60}")
    print("Balancing tasks")
    print(f"{'='*60}")

    counts = {name: len(p) for name, p in task_prompts.items()}
    if args.balance_target == "max":
        target = max(counts.values())
        print(f"Balance target: max = {target}")
    else:
        target = int(args.balance_target)
        print(f"Balance target: {target}")

    balanced_prompts = {}
    for name, prompts in task_prompts.items():
        raw_count = len(prompts)
        if raw_count == 0:
            print(f"  {name}: 0 examples — skipping")
            continue

        if raw_count >= target:
            # Downsample if larger (just take target count)
            balanced = prompts[:target]
            replication = 1.0
        else:
            reps = math.ceil(target / raw_count)
            balanced = (prompts * reps)[:target]
            replication = reps

        balanced_prompts[name] = balanced
        print(f"  {name}: {raw_count} raw -> x{replication} -> {len(balanced)} final")

    # ---- Step 4: Combine, shuffle, tokenize, save ----
    print(f"\n{'='*60}")
    print("Combining, shuffling, and saving")
    print(f"{'='*60}")

    all_prompts = []
    for name in args.tasks:
        if name in balanced_prompts:
            all_prompts.extend(balanced_prompts[name])

    combined_df = pd.DataFrame({'text': all_prompts})
    print(f"Combined dataset: {len(combined_df)} rows")

    print(f"Shuffling with seed={args.seed}...")
    combined_df = combined_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    print(f"Saving combined parquet to {combined_path}...")
    combined_df.to_parquet(combined_path, row_group_size=10_000)
    print(f"Saved {len(combined_df)} rows")

    # Tokenize — stream directly from the parquet we just saved
    print(f"\nTokenizing with max_length={args.max_seq_length} (streaming to disk)...")
    del combined_df, all_prompts, balanced_prompts, task_prompts

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
