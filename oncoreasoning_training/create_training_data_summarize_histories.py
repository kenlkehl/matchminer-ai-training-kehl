#!/usr/bin/env python3
"""
Generate serial patient summarization training data.

Reads patient_serial_summaries.parquet (summaries/reasoning) and
prepared_chunks.parquet (raw chunk text) from ../../data and produces
summarized_patient_histories.parquet with formatted prompts for training.

This uses the chunk-based serial summarization approach: each prompt includes
the prior summary and a new chunk of clinical notes (which may contain multiple
notes with dates), and the model outputs an updated summary.
"""

import pandas as pd
from transformers import AutoTokenizer


def build_prompt_text(
    tokenizer,
    prior_summary: str,
    first_date: str,
    last_date: str,
    chunk_text: str,
    max_model_len: int = 120000,
    margin_tokens: int = 5000
) -> str:
    """
    Build a single prompt for iterative summarization.
    Truncates chunk_text if too long, keeping head & tail.
    """
    threshold = max(1024, max_model_len - margin_tokens)

    # Truncate chunk_text if needed
    toks = tokenizer(chunk_text, add_special_tokens=False).input_ids
    if len(toks) > threshold:
        half = threshold // 2
        first_part = toks[:half]
        last_part = toks[-half:]
        chunk_text = tokenizer.decode(first_part) + " ... " + tokenizer.decode(last_part)

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

Document the patient's most recent age; sex; cancer type/primary site (eg breast cancer, lung cancer, etc); histology (eg adenocarcinoma, squamous carcinoma, etc); current extent (localized, advanced, metastatic, etc); biomarkers (genomic results, protein expression, etc); and treatment history (surgery, radiation, chemotherapy/targeted therapy/immunotherapy, etc, including start and stop dates and best response if known).
Do not consider localized basal cell or squamous carcinomas of the skin, or colon polyps, to be cancers for your purposes.
Do not include the patient's name, but do include relevant dates whenever documented.
If a patient has a history of more than one cancer, document the cancers one at a time. List the currently or most recently active cancer first, followed by any prior cancers. Within each cancer, events should be in chronological order.
CRITICAL: Format your response as free text ONLY. Do NOT output markdown, Unicode, or tables.

Also document any history of conditions that might meet "boilerplate" exclusion criteria for clinical trials, including uncontrolled brain metastases, lack of measurable disease, congestive heart failure, pneumonitis, renal dysfunction, liver dysfunction, lack of measurable disease,and HIV or hepatitis infection.
Clearly separate the "boilerplate" section by labeling it "Boilerplate: " before describing any such conditions.

Here is an example of the desired output format:

Age: 70
Sex: Male
Cancer type: Lung cancer
Histology: Adenocarcinoma
Current extent: Metastatic
Biomarkers: PD-L1 75%, KRAS G12C mutant
Treatment history:
# 1/5/2020-2/5/2021: carboplatin/pemetrexed/pembrolizumab
# 1/2021: Palliative radiation to progressive spinal metastases
# 3/2021-present: docetaxel
Boilerplate:
No evidence of common boilerplate exclusion criteria

---
PRIOR SUMMARY:
{prior_summary_text}

NEXT CLINICAL RECORD SEGMENT (covering {first_date} to {last_date}):
{chunk_text}
---
Now, write your updated summary. Do not add preceding text before the abstraction, and do not add commentary afterwards."""

    return user_content


def create_training_prompts(df: pd.DataFrame, tokenizer) -> list:
    """
    Create training prompts from chunk-based serial summarization data.

    Expects a DataFrame with columns: pseudo_mrn, chunk_index, first_date,
    last_date, prior_summary, new_summary_reasoning, new_summary, chunk_text.
    """
    # Sort by patient and chunk index
    df = df.sort_values(['pseudo_mrn', 'chunk_index']).reset_index(drop=True)

    prompts = []
    for i in range(len(df)):
        row = df.iloc[i]

        prior_summary = row['prior_summary'] if pd.notna(row['prior_summary']) else None
        first_date = str(row['first_date']) if pd.notna(row['first_date']) else "unknown date"
        last_date = str(row['last_date']) if pd.notna(row['last_date']) else "unknown date"
        chunk_text = str(row['chunk_text']) if pd.notna(row['chunk_text']) else ""

        # Combine reasoning + "assistantfinal" + summary for full assistant response
        reasoning = row['new_summary_reasoning']
        summary = row['new_summary']
        full_response = reasoning + "assistantfinal" + summary

        # Build user content using chunk-based prompt
        user_content = build_prompt_text(tokenizer, prior_summary, first_date, last_date, chunk_text)

        # Build the full message
        messages = [
            {'role': 'system', 'content': 'Reasoning: high'},
            {'role': 'user', 'content': user_content},
            {'role': 'assistant', 'content': full_response}
        ]

        prompt = tokenizer.apply_chat_template(conversation=messages, tokenize=False)
        prompts.append(prompt)

    return prompts


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/llama-3.2-3B-Instruct')

    # Load summaries (output of 6_summarize_patients.py)
    print("Loading patient serial summaries from ../../data/no_phi/patient_serial_summaries.parquet...")
    serial_summaries = pd.read_parquet('../../data/no_phi/patient_serial_summaries.parquet')
    print(f"Loaded {len(serial_summaries)} summary records")

    # Load prepared chunks (contains the raw chunk text needed for prompts)
    print("Loading prepared chunks from ../../data/no_phi/summary_shards/prepared_chunks.parquet...")
    prepared_chunks = pd.read_parquet('../../data/no_phi/summary_shards/prepared_chunks.parquet')
    print(f"Loaded {len(prepared_chunks)} chunk records")

    # Join summaries with chunk text on (patient_id/pseudo_mrn, local_idx/chunk_index)
    prepared_chunks = prepared_chunks.rename(columns={
        'patient_id': 'pseudo_mrn',
        'local_idx': 'chunk_index',
    })
    serial_summaries = serial_summaries.merge(
        prepared_chunks[['pseudo_mrn', 'chunk_index', 'chunk_text']],
        on=['pseudo_mrn', 'chunk_index'],
        how='left',
    )
    print(f"After merge: {len(serial_summaries)} records ({serial_summaries['chunk_text'].notna().sum()} with chunk text)")

    print("Generating training prompts...")
    prompts = create_training_prompts(serial_summaries, tokenizer)

    print("Saving to ../../data/no_phi/oncoreasoning_training_data/summarized_patient_histories.parquet...")
    pd.DataFrame(prompts, columns=['text']).to_parquet('../../data/no_phi/oncoreasoning_training_data/summarized_patient_histories.parquet')
    print(f"Done! Saved {len(prompts)} records")


if __name__ == '__main__':
    main()
