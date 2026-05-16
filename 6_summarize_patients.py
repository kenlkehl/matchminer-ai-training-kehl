#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Serial patient summarization with chunk-based updates using vLLM server(s).

For each patient, all clinical notes are sorted by date and concatenated into
a single text, then chunked into token-length segments. A running summary is
maintained across chunks. Work is scheduled in rounds (Round N = Nth chunk from
each patient) to maximize GPU utilization.

Multiple vLLM servers can be launched in parallel to increase throughput.
The number of servers is determined by: n_servers = len(gpu_ids) // gpus_per_server.
Within each round, prompts are distributed across servers round-robin.

Examples
--------
# Single server on 2 GPUs (tensor_parallel_size=2)
python 6_summarize_patients.py \
  --input_parquet ../data/no_phi/all_synthetic_notes.parquet \
  --output_parquet ../data/no_phi/patient_serial_summaries.parquet \
  --shard_dir ../data/no_phi/summary_shards \
  --model openai/gpt-oss-120b \
  --download_dir /data1/ken/models \
  --gpu_ids 0,1,2,3 \
  --gpus_per_server 1 \
  --max_model_len 120000 \
  --chunk_size 50000 \
  --chunk_overlap 500 \
  --generate_dates \
  --synthetic_start_date 2017-01-01 \
  --synthetic_min_days 7 \
  --synthetic_max_days 90 \
  --max_patients 10


# shorter chunks
python 6_summarize_patients.py \
  --input_parquet ../data/no_phi/all_synthetic_notes.parquet \
  --output_parquet ../data/no_phi/patient_serial_summaries_20K_chunks.parquet \
  --shard_dir ../data/no_phi/summary_shards_20K_chunks \
  --model openai/gpt-oss-120b \
  --download_dir ~/models \
  --gpu_ids 0,1,2,3,4,5,6,7 \
  --gpus_per_server 1 \
  --max_model_len 50000 \
  --chunk_size 20000 \
  --chunk_overlap 500 \
  --generate_dates \
  --synthetic_start_date 2017-01-01 \
  --synthetic_min_days 7 \
  --synthetic_max_days 90 \


# Two servers, 2 GPUs each (4 GPUs total, tensor_parallel_size=2 per server)
python 6_summarize_patients.py \
  --input_parquet ../data/no_phi/all_synthetic_notes.parquet \
  --output_parquet ../data/no_phi/patient_serial_summaries.parquet \
  --shard_dir ../data/no_phi/summary_shards \
  --model openai/gpt-oss-120b \
  --download_dir /data1/ken/models \
  --gpu_ids 0,1,2,3 \
  --gpus_per_server 2 \
  --base_port 8000 \
  --max_model_len 120000 \
  --chunk_size 10000 \
  --chunk_overlap 500 \
  --max_concurrent_requests 100


# Pass extra flags through to the vLLM server (quoted, shell-style; split with shlex)
python 6_summarize_patients.py \
  --input_parquet ../data/no_phi/all_synthetic_notes.parquet \
  --output_parquet ../data/no_phi/patient_serial_summaries.parquet \
  --shard_dir ../data/no_phi/summary_shards \
  --model openai/gpt-oss-120b \
  --download_dir /data1/ken/models \
  --gpu_ids 0,1,2,3 \
  --gpus_per_server 1 \
  --max_model_len 120000 \
  --chunk_size 50000 \
  --additional_vllm_args "--quantization nvfp4 --kv-cache-dtype fp8"
"""

import argparse
import asyncio
import glob
import os
import random
import re
import shlex
import signal
import subprocess
import sys
import time
import warnings
from datetime import datetime, timedelta
from multiprocessing import Pool
from typing import List, Dict, Tuple, Optional

import pandas as pd
import requests
from openai import AsyncOpenAI


# -------------------------
# Utilities
# -------------------------

def generate_synthetic_dates(
    df: pd.DataFrame,
    patient_id_col: str,
    date_col: str,
    start_date_str: str,
    min_days: int,
    max_days: int
) -> pd.DataFrame:
    """
    Generate synthetic dates for notes when no date column exists.

    For each patient, assigns dates starting from start_date, with random
    intervals between min_days and max_days for each subsequent note.
    Notes are assumed to be in their original order within each patient group.

    Args:
        df: Input DataFrame
        patient_id_col: Column name for patient ID
        date_col: Column name to create for dates
        start_date_str: Start date for first note (YYYY-MM-DD format)
        min_days: Minimum days between consecutive notes
        max_days: Maximum days between consecutive notes

    Returns:
        DataFrame with new date column added
    """
    df = df.copy()
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")

    # Generate dates for each patient
    dates = []
    for idx in range(len(df)):
        dates.append(None)  # Placeholder

    # Group by patient and assign dates
    current_patient = None
    current_date = start_date

    for idx, row in df.iterrows():
        pid = row[patient_id_col]

        if pid != current_patient:
            # New patient - reset to start date
            current_patient = pid
            current_date = start_date
        else:
            # Same patient - add random interval
            days_to_add = random.randint(min_days, max_days)
            current_date = current_date + timedelta(days=days_to_add)

        dates[idx] = current_date

    df[date_col] = dates
    return df


def concatenate_and_chunk_notes(
    notes: List[Tuple[str, str]],
    tokenizer,
    chunk_size: int = 10000,
    chunk_overlap: int = 500,
) -> List[Tuple[str, str, str]]:
    """
    Concatenate all notes for a patient with date headers, then chunk into
    token-length segments with overlap.

    Args:
        notes: List of (date_str, note_text) sorted chronologically
        tokenizer: HuggingFace tokenizer for token counting
        chunk_size: Maximum tokens per chunk
        chunk_overlap: Token overlap between consecutive chunks

    Returns:
        List of (chunk_text, first_date_in_chunk, last_date_in_chunk) tuples
    """
    assert chunk_overlap < chunk_size, "chunk_overlap must be less than chunk_size"

    if not notes:
        return []

    # Concatenate all notes with date headers
    blocks = []
    for date_str, note_text in notes:
        blocks.append(f"=== Clinical Note dated {date_str} ===\n{note_text}\n")
    full_text = "\n".join(blocks)

    all_dates = [date_str for date_str, _ in notes]

    # Tokenize the full concatenated text
    all_tokens = tokenizer(full_text, add_special_tokens=False).input_ids

    # If it fits in a single chunk, return as-is
    if len(all_tokens) <= chunk_size:
        return [(full_text, all_dates[0], all_dates[-1])]

    # Chunk with overlap
    stride = chunk_size - chunk_overlap
    date_header_pattern = re.compile(r"=== Clinical Note dated (.+?) ===")
    chunks = []

    start = 0
    while start < len(all_tokens):
        end = min(start + chunk_size, len(all_tokens))
        chunk_tokens = all_tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)

        # Extract dates present in this chunk
        found_dates = date_header_pattern.findall(chunk_text)
        if found_dates:
            first_date = found_dates[0]
            last_date = found_dates[-1]
        else:
            # Chunk falls within a single note with no header visible
            # Use the dates from the nearest preceding chunk or overall dates
            if chunks:
                first_date = chunks[-1][2]  # last_date of previous chunk
                last_date = first_date
            else:
                first_date = all_dates[0]
                last_date = all_dates[0]

        chunks.append((chunk_text, first_date, last_date))

        # Stop if we've reached the end
        if end >= len(all_tokens):
            break
        start += stride

    return chunks


def build_prompt_text(
    tokenizer,
    prior_summary: Optional[str],
    first_date: str,
    last_date: str,
    chunk_text: str,
    max_model_len: int,
    margin_tokens: int = 5000,
    model_name: str = ""
) -> str:
    """
    Build a single prompt for iterative summarization.
    Truncates chunk_text if too long, keeping head & tail.
    """
    threshold = max(1024, max_model_len - margin_tokens)

    # Truncate chunk_text if needed (safety net; chunks should already be sized)
    toks = tokenizer(chunk_text, add_special_tokens=False).input_ids
    if len(toks) > threshold:
        half = threshold // 2
        first_part = toks[:half]
        last_part = toks[-half:]
        chunk_text = tokenizer.decode(first_part) + " ... " + tokenizer.decode(last_part)

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
Now, write your updated summary, or if there is no new relevant information, output the prior summary exactly as it was. 
If any information is still relevant but is unchanged, just restate it in the updated summary, but do NOT state "no change" or similar - just produce the updated summary text as if you were writing it fresh, incorporating any new information but keeping relevant old information, without calling out what changed vs what stayed the same from the prior summary. 
You may update the old summary content in your output if the new information demonstrates that there was an error in the old output.
You may sometimes encounter contradictory information across notes (eg different biomarker results, or different cancer stage descriptions) - in that case, use your best judgment to determine which information is most likely to be correct based on the dates and context, and update the summary accordingly to reflect the most likely current state of the patient.
Do not add preceding text before the abstraction, and do not add commentary afterwards."""

    system_content = 'Reasoning: high' if 'gpt-oss' in model_name.lower() else ''
    messages = [
        {'role': 'system', 'content': system_content},
        {'role': 'user', 'content': user_content}
    ]

    prompt = tokenizer.apply_chat_template(
        conversation=messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=True,
    )
    return prompt


# Worker functions for parallel prompt building via multiprocessing.Pool.
# Each worker process loads its own tokenizer to avoid pickling issues.
_worker_tokenizer = None
_worker_model_name = None


def _init_prompt_worker(model_name, download_dir):
    """Initialize tokenizer in each worker process."""
    global _worker_tokenizer, _worker_model_name
    from transformers import AutoTokenizer
    _worker_tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=download_dir, trust_remote_code=True
    )
    _worker_model_name = model_name


def _build_prompt_worker(item):
    """Build a single prompt in a worker process. Returns (chunk_idx, prompt, prompt_token_count)."""
    chunk_idx, prior_summary, first_date, last_date, chunk_text, max_model_len = item
    prompt = build_prompt_text(
        _worker_tokenizer, prior_summary, first_date, last_date, chunk_text, max_model_len,
        model_name=_worker_model_name
    )
    prompt_token_count = len(_worker_tokenizer(prompt, add_special_tokens=False).input_ids)
    return (chunk_idx, prompt, prompt_token_count)


def postprocess_output(raw_text: str, parser_name: str, tokenizer) -> Tuple[str, str]:
    """Split output into (reasoning, summary) via the vLLM reasoning parser."""
    from vllm_reasoning_utils import parse_reasoning_output
    return parse_reasoning_output(raw_text, parser_name, tokenizer)


# -------------------------
# Work Planning
# -------------------------

_sentence_split_re = re.compile(r'(?<=[.!?])\s+')


def deduplicate_patient_notes(
    notes: List[Tuple[str, str]],
) -> List[Tuple[str, str]]:
    """
    Remove duplicate sentences across a patient's notes (keeping first occurrence).
    Notes must already be sorted chronologically.
    """
    seen: set = set()
    deduped_notes = []
    for date_str, note_text in notes:
        sentences = _sentence_split_re.split(note_text)
        kept = []
        for s in sentences:
            normalized = s.strip()
            if not normalized:
                continue
            if normalized not in seen:
                seen.add(normalized)
                kept.append(s)
        if kept:
            deduped_notes.append((date_str, " ".join(kept)))
    return deduped_notes


def prepare_rounds(
    df: pd.DataFrame,
    patient_id_col: str,
    date_col: str,
    text_col: str,
    tokenizer,
    chunk_size: int = 10000,
    chunk_overlap: int = 500,
) -> Tuple[List[List[Tuple[str, int, str, str, str]]], Dict[str, List[int]], Dict[str, str]]:
    """
    Organize work into rounds for parallel processing using chunk-based approach.

    For each patient, all notes are concatenated chronologically and split into
    token-length chunks. Rounds are built so that Round N contains the Nth chunk
    from each patient.

    Returns:
        rounds: List of rounds, each containing list of (patient_id, chunk_idx, first_date, last_date, chunk_text)
        patient_chunk_order: Dict mapping patient_id -> list of chunk indices
        patient_last_dates: Dict mapping patient_id -> last note date string
    """
    # Sort by patient and date
    df = df.sort_values([patient_id_col, date_col]).reset_index(drop=True)

    patient_chunks: Dict[str, List[Tuple[str, str, str]]] = {}  # pid -> [(chunk_text, first_date, last_date)]
    patient_chunk_order: Dict[str, List[int]] = {}  # pid -> [chunk_idx, ...]
    patient_last_dates: Dict[str, str] = {}  # pid -> last_note_date

    chunk_idx_counter = 0

    for pid, group in df.groupby(patient_id_col, sort=False):
        pid = str(pid)

        # Collect notes sorted by date for this patient
        notes = []
        last_date_str = "unknown date"
        for _, row in group.iterrows():
            date_val = row[date_col]
            date_str = str(date_val) if pd.notna(date_val) else "unknown date"
            note_text = str(row[text_col])
            notes.append((date_str, note_text))
            last_date_str = date_str

        patient_last_dates[pid] = last_date_str

        # Deduplicate sentences across this patient's notes
        notes = deduplicate_patient_notes(notes)

        # Concatenate and chunk
        chunks = concatenate_and_chunk_notes(notes, tokenizer, chunk_size, chunk_overlap)
        patient_chunks[pid] = chunks

        patient_chunk_order[pid] = []
        for _ in chunks:
            patient_chunk_order[pid].append(chunk_idx_counter)
            chunk_idx_counter += 1

    # Build rounds: round i contains the (i+1)th chunk for each patient that has one
    max_chunks = max(len(clist) for clist in patient_chunk_order.values()) if patient_chunk_order else 0

    rounds: List[List[Tuple[str, int, str, str, str]]] = []
    for round_idx in range(max_chunks):
        round_items = []
        for pid in patient_chunk_order:
            chunk_indices = patient_chunk_order[pid]
            if round_idx < len(chunk_indices):
                cidx = chunk_indices[round_idx]
                chunk_text, first_date, last_date = patient_chunks[pid][round_idx]
                round_items.append((pid, cidx, first_date, last_date, chunk_text))
        if round_items:
            rounds.append(round_items)

    return rounds, patient_chunk_order, patient_last_dates


def save_prepared_chunks(
    shard_dir: str,
    rounds: List[List[Tuple[str, int, str, str, str]]],
    patient_chunk_order: Dict[str, List[int]],
    patient_last_dates: Dict[str, str],
):
    """Save prepared chunks to parquet for fast resume."""
    os.makedirs(shard_dir, exist_ok=True)
    rows = []
    # Build local_idx lookup: for each patient, chunk_order position
    patient_local_idx: Dict[str, int] = {}
    for rnd in rounds:
        for pid, chunk_idx, first_date, last_date, chunk_text in rnd:
            local_idx = patient_local_idx.get(pid, 0)
            patient_local_idx[pid] = local_idx + 1
            rows.append({
                "patient_id": pid,
                "chunk_idx": chunk_idx,
                "local_idx": local_idx,
                "first_date": first_date,
                "last_date": last_date,
                "chunk_text": chunk_text,
                "patient_last_date": patient_last_dates.get(pid, ""),
            })
    chunk_df = pd.DataFrame(rows)
    path = os.path.join(shard_dir, "prepared_chunks.parquet")
    chunk_df.to_parquet(path, index=False)
    print(f"Saved prepared chunks: {path} ({len(chunk_df)} chunks)")


def load_prepared_chunks(
    shard_dir: str,
) -> Optional[Tuple[List[List[Tuple[str, int, str, str, str]]], Dict[str, List[int]], Dict[str, str]]]:
    """
    Load prepared chunks from parquet if available.

    Returns None if no cached file exists, otherwise returns
    (rounds, patient_chunk_order, patient_last_dates).
    """
    path = os.path.join(shard_dir, "prepared_chunks.parquet")
    if not os.path.exists(path):
        return None

    print(f"Loading cached prepared chunks from {path}...")
    chunk_df = pd.read_parquet(path)

    # Reconstruct patient_chunk_order and patient_last_dates
    patient_chunk_order: Dict[str, List[int]] = {}
    patient_last_dates: Dict[str, str] = {}

    for _, row in chunk_df.iterrows():
        pid = str(row["patient_id"])
        chunk_idx = int(row["chunk_idx"])
        if pid not in patient_chunk_order:
            patient_chunk_order[pid] = []
            patient_last_dates[pid] = str(row["patient_last_date"])
        patient_chunk_order[pid].append(chunk_idx)

    # Reconstruct rounds: group by local_idx (round number)
    max_local = int(chunk_df["local_idx"].max()) + 1 if len(chunk_df) > 0 else 0
    rounds: List[List[Tuple[str, int, str, str, str]]] = []
    for round_idx in range(max_local):
        round_rows = chunk_df[chunk_df["local_idx"] == round_idx]
        round_items = []
        for _, row in round_rows.iterrows():
            round_items.append((
                str(row["patient_id"]),
                int(row["chunk_idx"]),
                str(row["first_date"]),
                str(row["last_date"]),
                str(row["chunk_text"]),
            ))
        if round_items:
            rounds.append(round_items)

    print(f"Loaded {len(chunk_df)} cached chunks ({len(rounds)} rounds, {len(patient_chunk_order)} patients)")
    return rounds, patient_chunk_order, patient_last_dates


def load_existing_shards(shard_dir: str) -> Tuple[int, Dict[int, Tuple[str, str, str]]]:
    """
    Load existing shard files to enable resume.

    Returns:
        completed_rounds: Number of completed rounds
        results: Dict mapping row_idx -> (reasoning, summary, prior_summary)
    """
    results: Dict[int, Tuple[str, str, str]] = {}
    completed_rounds = 0

    if not os.path.exists(shard_dir):
        return completed_rounds, results

    shard_files = sorted(glob.glob(os.path.join(shard_dir, "round_*.parquet")))
    for shard_file in shard_files:
        # Extract round number from filename
        basename = os.path.basename(shard_file)
        match = re.match(r"round_(\d+)\.parquet", basename)
        if match:
            round_num = int(match.group(1))
            completed_rounds = max(completed_rounds, round_num + 1)

            shard_df = pd.read_parquet(shard_file)
            for _, row in shard_df.iterrows():
                row_idx = int(row["row_idx"])
                reasoning = str(row["reasoning"]) if pd.notna(row["reasoning"]) else ""
                summary = str(row["summary"]) if pd.notna(row["summary"]) else ""
                # Handle older shards that may not have prior_summary
                prior_summary = str(row["prior_summary"]) if "prior_summary" in row and pd.notna(row["prior_summary"]) else ""
                results[row_idx] = (reasoning, summary, prior_summary)

    return completed_rounds, results


def save_round_shard(
    shard_dir: str,
    round_idx: int,
    round_results: List[Tuple[int, str, str, str]]
):
    """Save results from a round to a shard file (includes prior_summary)."""
    os.makedirs(shard_dir, exist_ok=True)
    shard_path = os.path.join(shard_dir, f"round_{round_idx:04d}.parquet")

    shard_df = pd.DataFrame(round_results, columns=["row_idx", "reasoning", "summary", "prior_summary"])
    shard_df.to_parquet(shard_path, index=False)
    print(f"Saved shard: {shard_path} ({len(round_results)} records)")


# -------------------------
# vLLM Server Management
# -------------------------

def start_vllm_server(
    model: str,
    download_dir: str,
    gpu_ids: str,
    tensor_parallel_size: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    reasoning_parser: str,
    max_num_seqs: int = 900,
    port: int = 8000,
    log_file: Optional[str] = None,
    enforce_eager: bool = False,
    additional_vllm_args: Optional[List[str]] = None,
) -> subprocess.Popen:
    """Start vLLM server as a subprocess."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_ids

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--download-dir", download_dir,
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--max-model-len", str(max_model_len),
        "--max-num-seqs", str(max_num_seqs),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--port", str(port),
        "--reasoning-parser", reasoning_parser,
    ]
    if enforce_eager:
        cmd.append("--enforce-eager")
    if additional_vllm_args:
        cmd.extend(additional_vllm_args)

    print(f"Starting vLLM server: {' '.join(cmd)}")
    print(f"Using GPUs: {gpu_ids}")

    # Start process - write logs to file if specified, otherwise to console
    if log_file:
        print(f"vLLM server logs will be written to: {log_file}")
        log_handle = open(log_file, "w")
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        process._log_handle = log_handle  # Store for cleanup
    else:
        # Let vLLM output go directly to console for debugging
        print("vLLM server logs will be shown in console")
        process = subprocess.Popen(
            cmd,
            env=env,
            # No stdout/stderr redirection - goes to console
        )

    return process


def wait_for_server_ready(port: int, timeout: int = 600, poll_interval: float = 5.0) -> bool:
    """
    Poll health endpoint until server is ready.

    Args:
        port: Server port
        timeout: Maximum seconds to wait
        poll_interval: Seconds between health checks

    Returns:
        True if server is ready, False if timeout exceeded
    """
    health_url = f"http://localhost:{port}/health"
    start_time = time.time()

    print(f"Waiting for vLLM server to be ready at {health_url}...")

    while time.time() - start_time < timeout:
        try:
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                print(f"vLLM server is ready (took {time.time() - start_time:.1f}s)")
                return True
        except requests.exceptions.RequestException:
            pass

        time.sleep(poll_interval)
        elapsed = time.time() - start_time
        print(f"  Still waiting... ({elapsed:.0f}s / {timeout}s)")

    print(f"Timeout waiting for vLLM server after {timeout}s")
    return False


def check_server_health(port: int) -> bool:
    """Check if vLLM server is still responding."""
    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def shutdown_server(process: subprocess.Popen, timeout: int = 30):
    """Gracefully terminate the server subprocess."""
    if process is None:
        return

    print("Shutting down vLLM server...")

    # Close log file handle if present
    if hasattr(process, '_log_handle') and process._log_handle:
        try:
            process._log_handle.close()
        except Exception:
            pass

    # Try graceful termination first
    process.terminate()

    try:
        process.wait(timeout=timeout)
        print("vLLM server stopped gracefully.")
    except subprocess.TimeoutExpired:
        print("Server did not stop gracefully, forcing kill...")
        process.kill()
        process.wait()
        print("vLLM server killed.")


# -------------------------
# Async Inference
# -------------------------

async def single_inference_request(
    client: AsyncOpenAI,
    row_idx: int,
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    top_k: int,
    parser_name: str,
    tokenizer,
    top_p: float = 1.0,
    presence_penalty: float = 0.0,
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
    max_retries: int = 6,
    base_timeout: float = 600.0,
) -> Tuple[int, str, str]:
    """
    Send a single inference request with retry logic.
    Returns (row_idx, reasoning, summary).
    """
    for attempt in range(max_retries):
        try:
            extra = {
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "skip_special_tokens": False,
            }
            if min_p > 0.0:
                extra["min_p"] = min_p
            response = await asyncio.wait_for(
                client.completions.create(
                    model=model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    presence_penalty=presence_penalty,
                    extra_body=extra,
                ),
                timeout=base_timeout
            )
            raw_text = response.choices[0].text
            reasoning, summary = postprocess_output(raw_text, parser_name, tokenizer)
            return (row_idx, reasoning, summary)

        except asyncio.TimeoutError:
            wait_time = min((2 ** attempt) * 15, 600)  # 15s, 30s, 60s, 120s, 240s, 480s, 600s, ...
            if attempt < max_retries - 1:
                print(f"  Row {row_idx}: timeout (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                print(f"  Row {row_idx}: all retries exhausted (timeout)")
                return (row_idx, "", "ERROR: timeout after all retries")

        except Exception as e:
            wait_time = min((2 ** attempt) * 10, 300)  # 10s, 20s, 40s, 80s, 160s, 300s, ...
            if attempt < max_retries - 1:
                print(f"  Row {row_idx}: error '{e}' (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                print(f"  Row {row_idx}: all retries exhausted")
                return (row_idx, "", f"ERROR: {e}")

    return (row_idx, "", "ERROR: unexpected retry loop exit")


async def run_inference_batch(
    client: AsyncOpenAI,
    prompts: List[Tuple[int, str, int]],  # (row_idx, prompt_text, max_tokens)
    model: str,
    temperature: float,
    top_k: int,
    parser_name: str,
    tokenizer,
    top_p: float = 1.0,
    presence_penalty: float = 0.0,
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
    max_concurrent: int = 16,
    batch_size: int = 64,
    max_retries: int = 6,
    base_timeout: float = 600.0,
    port: int = 8000,
) -> List[Tuple[int, str, str]]:
    """
    Send batch of requests concurrently, return (row_idx, reasoning, summary).

    Processes prompts in smaller batches to avoid overwhelming the server.
    Each batch runs max_concurrent requests in parallel.
    """
    total = len(prompts)
    all_results: List[Tuple[int, str, str]] = []
    completed = 0
    max_health_failures = 5
    consecutive_health_failures = 0

    # Process in batches
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_prompts = prompts[batch_start:batch_end]

        # Check server health before each batch (with progressive backoff)
        if not check_server_health(port):
            consecutive_health_failures += 1
            print(f"  WARNING: vLLM server health check failed (attempt {consecutive_health_failures}/{max_health_failures})")
            if consecutive_health_failures >= max_health_failures:
                print(f"  ERROR: vLLM server appears to be dead. Marking remaining {total - completed} prompts as errors.")
                for idx, prompt, _mt in prompts[batch_start:]:
                    all_results.append((idx, "", "ERROR: vLLM server died"))
                return all_results
            # Progressive backoff: 10s, 20s, 40s, 60s
            health_wait = min(10 * (2 ** (consecutive_health_failures - 1)), 60)
            print(f"  Waiting {health_wait}s before health re-check...")
            await asyncio.sleep(health_wait)
            if not check_server_health(port):
                print(f"  WARNING: vLLM server still not responding, will retry on next batch.")
        else:
            consecutive_health_failures = 0

        print(f"  Processing batch {batch_start + 1}-{batch_end} of {total}...")

        # Use semaphore to limit concurrent requests within batch
        semaphore = asyncio.Semaphore(max_concurrent)

        async def bounded_request(row_idx: int, prompt: str, prompt_max_tokens: int) -> Tuple[int, str, str]:
            async with semaphore:
                return await single_inference_request(
                    client=client,
                    row_idx=row_idx,
                    prompt=prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=prompt_max_tokens,
                    top_k=top_k,
                    parser_name=parser_name,
                    tokenizer=tokenizer,
                    top_p=top_p,
                    presence_penalty=presence_penalty,
                    min_p=min_p,
                    repetition_penalty=repetition_penalty,
                    max_retries=max_retries,
                    base_timeout=base_timeout,
                )

        # Create tasks for this batch only
        tasks = [bounded_request(idx, prompt, mt) for idx, prompt, mt in batch_prompts]
        batch_results = await asyncio.gather(*tasks)
        all_results.extend(batch_results)

        completed += len(batch_results)

        # Count errors in this batch
        batch_errors = sum(1 for _, _, summary in batch_results if summary.startswith("ERROR:"))
        if batch_errors > 0:
            print(f"  Progress: {completed}/{total} completed ({batch_errors} errors in this batch)")
        else:
            print(f"  Progress: {completed}/{total} completed")

    return all_results


async def process_all_rounds(
    rounds: List[List[Tuple[str, int, str, str, str]]],
    patient_chunk_order: Dict[str, List[int]],
    patient_summaries: Dict[str, str],
    all_results: Dict[int, Tuple[str, str, str]],
    completed_rounds: int,
    args: argparse.Namespace,
    server_clients: List[Tuple[AsyncOpenAI, int]],
    prompt_pool: Pool,
    parser_name: str,
    tokenizer,
):
    """
    Process all remaining rounds using async inference across multiple servers.

    Args:
        rounds: List of rounds, each containing (patient_id, chunk_idx, first_date, last_date, chunk_text)
        patient_chunk_order: Dict mapping patient_id -> list of chunk indices
        patient_summaries: Dict tracking current summary per patient (mutated)
        all_results: Dict tracking all results by chunk_idx (mutated)
        completed_rounds: Number of rounds already completed
        args: Command line arguments
        server_clients: List of (AsyncOpenAI client, port) tuples, one per server
        prompt_pool: Multiprocessing pool for parallel prompt building
    """
    n_servers = len(server_clients)

    # Build reverse mapping: chunk_idx -> patient_id
    chunk_to_patient: Dict[int, str] = {}
    for pid, chunk_indices in patient_chunk_order.items():
        for cidx in chunk_indices:
            chunk_to_patient[cidx] = pid

    for round_idx in range(completed_rounds, len(rounds)):
        round_items = rounds[round_idx]
        print(f"\n=== Round {round_idx + 1}/{len(rounds)}: {len(round_items)} patients ===")

        # Build prior summaries (fast, sequential dict lookups)
        round_prior_summaries: Dict[int, str] = {}
        work_items = []
        for pid, chunk_idx, first_date, last_date, chunk_text in round_items:
            prior_summary = patient_summaries.get(pid, None)
            prior_text = prior_summary if prior_summary else "None - this is the first segment for this patient"
            round_prior_summaries[chunk_idx] = prior_text
            work_items.append((chunk_idx, prior_summary, first_date, last_date, chunk_text, args.max_model_len))

        # Build prompts in parallel across CPU cores
        print(f"Building {len(work_items)} prompts in parallel...")
        chunksize = max(1, len(work_items) // (prompt_pool._processes * 4))
        prompt_results: List[Tuple[int, str, int]] = list(prompt_pool.map(_build_prompt_worker, work_items, chunksize=chunksize))

        # Compute per-prompt max_tokens: max_model_len - prompt_token_count
        prompts: List[Tuple[int, str, int]] = []
        for chunk_idx, prompt, prompt_token_count in prompt_results:
            gen_tokens = args.max_model_len - prompt_token_count
            if args.max_tokens is not None:
                gen_tokens = min(gen_tokens, args.max_tokens)
            gen_tokens = max(gen_tokens, 1)  # safety floor
            prompts.append((chunk_idx, prompt, gen_tokens))
        print(f"Prompts built.")

        # Distribute prompts across servers round-robin
        server_prompt_groups: List[List[Tuple[int, str, int]]] = [[] for _ in range(n_servers)]
        for i, prompt_item in enumerate(prompts):
            server_prompt_groups[i % n_servers].append(prompt_item)

        print(f"Distributing {len(prompts)} requests across {n_servers} server(s)...")
        for si, (_, port) in enumerate(server_clients):
            print(f"  Server {si} (port {port}): {len(server_prompt_groups[si])} prompts")

        # Launch inference on all servers concurrently
        tasks = []
        for server_idx, (client, port) in enumerate(server_clients):
            group = server_prompt_groups[server_idx]
            if group:
                tasks.append(
                    run_inference_batch(
                        client=client,
                        prompts=group,
                        model=args.model,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        parser_name=parser_name,
                        tokenizer=tokenizer,
                        top_p=args.top_p,
                        presence_penalty=args.presence_penalty,
                        min_p=args.min_p,
                        repetition_penalty=args.repetition_penalty,
                        max_concurrent=args.max_concurrent_requests,
                        batch_size=args.batch_size,
                        max_retries=args.max_retries,
                        base_timeout=args.request_timeout,
                        port=port,
                    )
                )

        all_batch_results = await asyncio.gather(*tasks)
        results = []
        for batch_result in all_batch_results:
            results.extend(batch_result)

        # Update state
        round_results_with_prior: List[Tuple[int, str, str, str]] = []
        for chunk_idx, reasoning, summary in results:
            prior_text = round_prior_summaries[chunk_idx]
            all_results[chunk_idx] = (reasoning, summary, prior_text)
            round_results_with_prior.append((chunk_idx, reasoning, summary, prior_text))

            # Update patient summary for next round
            pid = chunk_to_patient[chunk_idx]
            patient_summaries[pid] = summary

        # Save round shard
        save_round_shard(args.shard_dir, round_idx, round_results_with_prior)
        print(f"Round {round_idx + 1} complete.")


async def process_all_rounds_via_pool(
    rounds: List[List[Tuple[str, int, str, str, str]]],
    patient_chunk_order: Dict[str, List[int]],
    patient_summaries: Dict[str, str],
    all_results: Dict[int, Tuple[str, str, str]],
    completed_rounds: int,
    args: argparse.Namespace,
    prompt_pool: Pool,
    parser_name: str,
    tokenizer,
):
    """Remote-pool variant of process_all_rounds. Runs each round as a
    separate run_pool call against a single DynamicServerRegistry so the
    sequential dependency (chunk N depends on chunk N-1's summary) is
    preserved while still benefiting from the pool's fault tolerance and
    adaptive concurrency."""
    from remote_vllm_pool import (
        CompletionSampling,
        make_completion_work_fn,
        run_pool,
        build_registry_from_args,
    )

    if args.max_concurrent_per_server is None:
        args.max_concurrent_per_server = int(args.max_concurrent_requests)

    sampling = CompletionSampling(
        model=args.model,
        temperature=float(args.temperature),
        top_k=int(args.top_k),
        top_p=float(args.top_p),
        presence_penalty=float(args.presence_penalty),
        min_p=float(args.min_p),
        repetition_penalty=float(args.repetition_penalty),
        request_timeout=float(args.request_timeout),
    )
    work_fn = make_completion_work_fn(sampling, parser_name, tokenizer)
    registry = build_registry_from_args(args)
    await registry.start()

    chunk_to_patient: Dict[int, str] = {}
    for pid, chunk_indices in patient_chunk_order.items():
        for cidx in chunk_indices:
            chunk_to_patient[cidx] = pid

    try:
        for round_idx in range(completed_rounds, len(rounds)):
            round_items = rounds[round_idx]
            print(f"\n=== Round {round_idx + 1}/{len(rounds)}: "
                  f"{len(round_items)} patients (remote pool) ===")

            # Build per-chunk inputs for the prompt pool (same shape as
            # process_all_rounds).
            round_prior_summaries: Dict[int, str] = {}
            prompt_inputs = []
            for pid, chunk_idx, first_date, last_date, chunk_text in round_items:
                prior_summary = patient_summaries.get(pid, None)
                prior_text = prior_summary if prior_summary else "None - this is the first segment for this patient"
                round_prior_summaries[chunk_idx] = prior_text
                prompt_inputs.append(
                    (chunk_idx, prior_summary, first_date, last_date, chunk_text, args.max_model_len)
                )

            print(f"Building {len(prompt_inputs)} prompts in parallel...")
            chunksize = max(1, len(prompt_inputs) // (prompt_pool._processes * 4))
            prompt_results = list(
                prompt_pool.map(_build_prompt_worker, prompt_inputs, chunksize=chunksize)
            )

            work_items = []
            for chunk_idx, prompt, prompt_token_count in prompt_results:
                gen_tokens = args.max_model_len - prompt_token_count
                if args.max_tokens is not None:
                    gen_tokens = min(gen_tokens, args.max_tokens)
                gen_tokens = max(gen_tokens, 1)
                work_items.append((int(chunk_idx),
                                   {"prompt": prompt, "max_tokens": int(gen_tokens)}))

            # Collect results into memory; we will then save_round_shard for
            # the whole round (matching the existing local-mode shard layout).
            collected: List[Tuple[int, str, str]] = []

            def shard_writer(payload, shard_idx):
                for chunk_idx, result in payload:
                    if isinstance(result, str) and result.startswith("ERROR:"):
                        reasoning, summary = "", result
                    else:
                        reasoning, summary = result
                    collected.append((int(chunk_idx), reasoning, summary))

            await run_pool(
                work_items=work_items,
                work_fn=work_fn,
                registry=registry,
                shard_writer=shard_writer,
                results_per_shard=10 ** 9,  # one flush at end-of-round
                starting_shard_idx=0,
                max_attempts=int(args.max_attempts),
            )

            # Update state and write the round shard.
            round_results_with_prior: List[Tuple[int, str, str, str]] = []
            for chunk_idx, reasoning, summary in collected:
                prior_text = round_prior_summaries[chunk_idx]
                all_results[chunk_idx] = (reasoning, summary, prior_text)
                round_results_with_prior.append((chunk_idx, reasoning, summary, prior_text))
                pid = chunk_to_patient[chunk_idx]
                patient_summaries[pid] = summary

            save_round_shard(args.shard_dir, round_idx, round_results_with_prior)
            print(f"Round {round_idx + 1} complete.")
    finally:
        await registry.stop()


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser("Chunk-based patient summarization with iterative updates using vLLM server.")
    ap.add_argument("--input_parquet", required=True)
    ap.add_argument("--output_parquet", required=True)
    ap.add_argument("--patient_summaries_parquet", default="../data/no_phi/patient_summaries.parquet",
                    help="Output parquet with just the last row per patient (default: ../data/no_phi/patient_summaries.parquet)")
    ap.add_argument("--shard_dir", required=True, help="Directory for checkpoint shards")
    ap.add_argument("--patient_id_col", default="pseudo_mrn", help="Column name for patient ID")
    ap.add_argument("--date_col", default="date", help="Column name for note date (will be created if missing and --generate_dates is set)")
    ap.add_argument("--text_col", default="synthetic_note", help="Column name for note text")
    ap.add_argument("--generate_dates", action="store_true",
                    help="Generate synthetic dates if date column is missing")
    ap.add_argument("--synthetic_start_date", default="2020-01-01",
                    help="Start date for first note of each patient (format: YYYY-MM-DD)")
    ap.add_argument("--synthetic_min_days", type=int, default=7,
                    help="Minimum days between consecutive notes")
    ap.add_argument("--synthetic_max_days", type=int, default=90,
                    help="Maximum days between consecutive notes")
    ap.add_argument("--chunk_size", type=int, default=20000,
                    help="Maximum tokens per chunk when concatenating patient notes (default: 10000)")
    ap.add_argument("--chunk_overlap", type=int, default=500,
                    help="Token overlap between consecutive chunks (default: 500)")
    ap.add_argument("--model", default="gpt-oss-120b")
    ap.add_argument("--download_dir", required=True)
    ap.add_argument("--gpu_ids", default=None,
                    help="Comma-separated list of GPU IDs (e.g., 0,1,2,3). Used with --gpus_per_server to determine number of servers. Required for local mode (omit when using --server_urls / --server_urls_file).")
    ap.add_argument("--gpus_per_server", type=int, default=None,
                    help="Number of GPUs per vLLM server. n_servers = len(gpu_ids) // gpus_per_server. "
                         "tensor_parallel_size is set to this value. Required for local mode.")
    ap.add_argument("--max_model_len", type=int, default=50000)
    ap.add_argument("--enforce_eager", action="store_true",
                    help="Pass --enforce-eager to vLLM (disables CUDA graphs; helps surface engine crash tracebacks)")
    ap.add_argument("--additional_vllm_args", type=str, default="",
                    help="Extra arguments to append verbatim to the vLLM server command. "
                         "Pass as a single shell-quoted string, e.g. "
                         "--additional_vllm_args \"--quantization nvfp4 --kv-cache-dtype fp8\". "
                         "Ignored when --server_urls is set.")
    ap.add_argument("--max_num_seqs", type=int, default=900, help="vLLM max_num_seqs (concurrent request cap).")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_k", type=int, default=1)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--presence_penalty", type=float, default=0.0)
    ap.add_argument("--min_p", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=20000,
                    help="Max generation tokens per prompt. If not set, auto-computed as max_model_len minus prompt token count.")
    ap.add_argument("--repetition_penalty", type=float, default=1.1)
    from vllm_reasoning_utils import add_reasoning_cli_args
    add_reasoning_cli_args(ap)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    ap.add_argument("--base_port", type=int, default=8000,
                    help="Base port for vLLM servers. Server i uses base_port + i (default: 8000)")
    ap.add_argument("--server_urls", type=str, default=None,
                    help="Comma-separated URLs of existing vLLM servers "
                         "(e.g. 'http://localhost:8000/v1,http://localhost:8001/v1'). "
                         "When set, no servers are started or stopped.")
    ap.add_argument("--server_urls_file", type=str, default=None,
                    help="Path to JSON file maintained by gcp_vllm_orchestrator.py "
                         "with the dynamic list of healthy server URLs. Mutually "
                         "exclusive with --server_urls.")
    ap.add_argument("--server_urls_refresh", type=float, default=15.0,
                    help="Seconds between re-reads of --server_urls_file.")
    ap.add_argument("--max_concurrent_per_server", type=int, default=None,
                    help="Per-server concurrency ceiling for remote-pool mode "
                         "(adaptive). Defaults to --max_concurrent_requests.")
    ap.add_argument("--max_attempts", type=int, default=200,
                    help="Per-item max retries before recording an ERROR placeholder.")
    ap.add_argument("--max_concurrent_requests", type=int, default=16,
                    help="Maximum concurrent requests to vLLM server (default: 16)")
    ap.add_argument("--batch_size", type=int, default=1000,
                    help="Number of prompts to process per batch before waiting (default: 1000)")
    ap.add_argument("--request_timeout", type=float, default=600.0,
                    help="Timeout in seconds for individual inference requests (default: 600)")
    ap.add_argument("--max_retries", type=int, default=6,
                    help="Maximum retries for failed requests (default: 6)")
    ap.add_argument("--server_timeout", type=int, default=600,
                    help="Timeout in seconds waiting for vLLM server to start (default: 600)")
    ap.add_argument("--max_patients", type=int, default=None,
                    help="Limit to first N patients (for testing)")
    ap.add_argument("--run_deterministic", action="store_true",
                    help="Override LLM parameters for deterministic output: "
                         "temperature=0.0, top_k=1, repetition_penalty=1.0, "
                         "max_model_len=120000, chunk_size=50000, max_tokens=10000")
    args = ap.parse_args()

    if args.server_urls and args.server_urls_file:
        ap.error("Specify only one of --server_urls / --server_urls_file.")
    remote_mode = bool(args.server_urls or args.server_urls_file)
    if not remote_mode and (not args.gpu_ids or args.gpus_per_server is None):
        ap.error("--gpu_ids and --gpus_per_server are required unless --server_urls / --server_urls_file is set.")

    # Resolve reasoning parser from --model + --reasoning-parser (default auto)
    from vllm_reasoning_utils import resolve_parser_name
    reasoning_parser = resolve_parser_name(args.model, args.reasoning_parser)
    print(f"Using vLLM reasoning parser: {reasoning_parser}")

    # Apply deterministic overrides
    if args.run_deterministic:
        args.temperature = 0.0
        args.top_k = 1
        args.repetition_penalty = 1.0
        args.max_model_len = 120000
        args.chunk_size = 50000
        args.max_tokens = 10000
        print("Deterministic mode: temperature=0.0, top_k=1, repetition_penalty=1.0, "
              "max_model_len=120000, chunk_size=50000, max_tokens=10000")

    # Load tokenizer up front — used for chunking (when not cached) AND for
    # parsing server responses through the reasoning parser.
    print("Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        cache_dir=args.download_dir,
        trust_remote_code=True
    )

    # Try loading cached prepared chunks first to skip expensive data prep
    cached = load_prepared_chunks(args.shard_dir)
    if cached is not None:
        rounds, patient_chunk_order, patient_last_dates = cached
    else:
        # No cache — load data, generate dates, chunk notes, and save
        print(f"Loading {args.input_parquet}...")
        df = pd.read_parquet(args.input_parquet)

        # Validate columns
        if args.patient_id_col not in df.columns:
            raise ValueError(f"Column '{args.patient_id_col}' (--patient_id_col) not found in input. Available: {df.columns.tolist()}")
        if args.text_col not in df.columns:
            raise ValueError(f"Column '{args.text_col}' (--text_col) not found in input. Available: {df.columns.tolist()}")

        # Handle date column - generate synthetic dates if missing and --generate_dates is set
        if args.date_col not in df.columns:
            if args.generate_dates:
                print(f"Date column '{args.date_col}' not found. Generating synthetic dates...")
                df = generate_synthetic_dates(
                    df,
                    patient_id_col=args.patient_id_col,
                    date_col=args.date_col,
                    start_date_str=args.synthetic_start_date,
                    min_days=args.synthetic_min_days,
                    max_days=args.synthetic_max_days
                )
                print(f"Generated synthetic dates in column '{args.date_col}'")
            else:
                raise ValueError(
                    f"Column '{args.date_col}' (--date_col) not found in input. "
                    f"Use --generate_dates to create synthetic dates. Available columns: {df.columns.tolist()}"
                )

        # Filter to max_patients if specified
        if args.max_patients is not None:
            unique_patients = df[args.patient_id_col].unique()[:args.max_patients]
            df = df[df[args.patient_id_col].isin(unique_patients)].copy()
            print(f"Limited to {args.max_patients} patients ({len(df)} rows)")

        df = df.reset_index(drop=True)

        print("Preparing work rounds (concatenating and chunking notes per patient)...")
        rounds, patient_chunk_order, patient_last_dates = prepare_rounds(
            df, args.patient_id_col, args.date_col, args.text_col,
            tokenizer, args.chunk_size, args.chunk_overlap
        )
        save_prepared_chunks(args.shard_dir, rounds, patient_chunk_order, patient_last_dates)

    total_chunks = sum(len(clist) for clist in patient_chunk_order.values())
    print(f"Organized into {len(rounds)} rounds for {len(patient_chunk_order)} patients ({total_chunks} total chunks)")

    # Load existing shards for resume
    completed_rounds, all_results = load_existing_shards(args.shard_dir)
    if completed_rounds > 0:
        print(f"Resuming from round {completed_rounds} (loaded {len(all_results)} existing results)")

    # Track current summaries per patient
    patient_summaries_dict: Dict[str, str] = {}

    # Reconstruct patient summaries from completed rounds
    if completed_rounds > 0:
        for round_idx in range(completed_rounds):
            for pid, chunk_idx, _, _, _ in rounds[round_idx]:
                if chunk_idx in all_results:
                    _, summary, _ = all_results[chunk_idx]
                    patient_summaries_dict[pid] = summary

    # Check if there's work to do
    if completed_rounds >= len(rounds):
        print("All rounds already completed. Building final output...")
    else:
        # Ensure shard_dir exists for server log files
        os.makedirs(args.shard_dir, exist_ok=True)

        # Create multiprocessing pool for parallel prompt building
        n_workers = min(os.cpu_count() or 4, 32)
        print(f"Creating prompt-building pool with {n_workers} workers...")
        prompt_pool = Pool(
            processes=n_workers,
            initializer=_init_prompt_worker,
            initargs=(args.model, args.download_dir),
        )

        # --- Server setup: remote pool, static URLs, or launch our own ---
        server_infos: List[Tuple[subprocess.Popen, int]] = []  # (process, port)
        server_clients: List[Tuple[AsyncOpenAI, int]] = []

        if remote_mode:
            # Server discovery / health / failover is handled inside the
            # remote_vllm_pool via DynamicServerRegistry. No clients here.
            n_servers = 0
            print("Remote-pool mode: using DynamicServerRegistry "
                  "(servers managed by gcp_vllm_orchestrator).")
        elif args.server_urls:
            # Legacy static --server_urls path
            urls = [u.strip() for u in args.server_urls.split(",") if u.strip()]
            print(f"Using {len(urls)} external server(s): {urls}")
            for url in urls:
                # Extract port from URL for logging (e.g. http://localhost:8000/v1 → 8000)
                from urllib.parse import urlparse
                parsed = urlparse(url)
                port = parsed.port or 0
                client = AsyncOpenAI(
                    base_url=url,
                    api_key="not-needed",
                    timeout=args.request_timeout + 60,
                )
                server_clients.append((client, port))
            n_servers = len(urls)
            print(f"All {n_servers} external server(s) connected.")
        else:
            # Normalize gpu_ids to remove any semicolons and use comma format
            gpu_ids_normalized = args.gpu_ids.replace(";", ",")
            gpu_list = [g.strip() for g in gpu_ids_normalized.split(",") if g.strip()]

            # Validate GPU count vs gpus_per_server
            if len(gpu_list) % args.gpus_per_server != 0:
                remainder = len(gpu_list) % args.gpus_per_server
                usable = len(gpu_list) - remainder
                print(f"WARNING: {len(gpu_list)} GPUs not evenly divisible by gpus_per_server={args.gpus_per_server}. "
                      f"Using first {usable} GPUs, ignoring GPUs: {gpu_list[usable:]}")
                gpu_list = gpu_list[:usable]

            n_servers = len(gpu_list) // args.gpus_per_server
            if n_servers == 0:
                raise ValueError(f"Not enough GPUs ({len(gpu_list)}) for gpus_per_server={args.gpus_per_server}")

            print(f"Starting {n_servers} vLLM server(s), each with {args.gpus_per_server} GPU(s)")

            extra_vllm_args = shlex.split(args.additional_vllm_args) if args.additional_vllm_args else []
            if extra_vllm_args:
                print(f"Appending additional vLLM args: {extra_vllm_args}")

            # Start N vLLM servers (subprocess spawns are non-blocking, so models load concurrently)
            for server_idx in range(n_servers):
                gpu_start = server_idx * args.gpus_per_server
                gpu_end = gpu_start + args.gpus_per_server
                server_gpu_ids = ",".join(gpu_list[gpu_start:gpu_end])
                server_port = args.base_port + server_idx

                log_file = os.path.join(args.shard_dir, f"vllm_server_{server_idx}.log")

                process = start_vllm_server(
                    model=args.model,
                    download_dir=args.download_dir,
                    gpu_ids=server_gpu_ids,
                    tensor_parallel_size=args.gpus_per_server,
                    max_model_len=args.max_model_len,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    reasoning_parser=reasoning_parser,
                    max_num_seqs=args.max_num_seqs,
                    port=server_port,
                    log_file=log_file,
                    enforce_eager=args.enforce_eager,
                    additional_vllm_args=extra_vllm_args,
                )
                server_infos.append((process, server_port))

            # Wait for all servers to be ready
            for i, (process, port) in enumerate(server_infos):
                print(f"Waiting for server {i} (port {port})...")
                if not wait_for_server_ready(port, timeout=args.server_timeout):
                    print(f"Failed to start vLLM server {i} on port {port}. Shutting down all servers.")
                    for proc, _ in server_infos:
                        shutdown_server(proc)
                    sys.exit(1)

            # Create async OpenAI clients, one per server
            for i, (process, port) in enumerate(server_infos):
                client = AsyncOpenAI(
                    base_url=f"http://localhost:{port}/v1",
                    api_key="not-needed",
                    timeout=args.request_timeout + 60,
                )
                server_clients.append((client, port))

            print(f"All {n_servers} vLLM server(s) ready.")

        try:
            if remote_mode:
                asyncio.run(process_all_rounds_via_pool(
                    rounds=rounds,
                    patient_chunk_order=patient_chunk_order,
                    patient_summaries=patient_summaries_dict,
                    all_results=all_results,
                    completed_rounds=completed_rounds,
                    args=args,
                    prompt_pool=prompt_pool,
                    parser_name=reasoning_parser,
                    tokenizer=tokenizer,
                ))
            else:
                asyncio.run(process_all_rounds(
                    rounds=rounds,
                    patient_chunk_order=patient_chunk_order,
                    patient_summaries=patient_summaries_dict,
                    all_results=all_results,
                    completed_rounds=completed_rounds,
                    args=args,
                    server_clients=server_clients,
                    prompt_pool=prompt_pool,
                    parser_name=reasoning_parser,
                    tokenizer=tokenizer,
                ))

        finally:
            # Always shutdown prompt pool
            prompt_pool.close()
            prompt_pool.join()
            # Only shutdown servers we started ourselves
            if server_infos:
                print(f"Shutting down {len(server_infos)} vLLM server(s)...")
                for i, (process, port) in enumerate(server_infos):
                    print(f"  Shutting down server {i} (port {port})...")
                    shutdown_server(process)
                print("All servers shut down.")
            elif remote_mode:
                print("Remote vLLM servers (managed by orchestrator) left running.")
            else:
                print("External servers left running (not managed by this script).")

    # Build final output (one row per chunk per patient)
    def split_boilerplate(text: str) -> Tuple[str, str]:
        """Split summary into main summary and boilerplate text."""
        if not text:
            return "", ""
        markers = ["Boilerplate:", "BOILERPLATE:", "boilerplate:", "Boilerplate conditions:", "BOILERPLATE CONDITIONS:", "boilerplate conditions:"]
        for marker in markers:
            if marker in text:
                parts = text.split(marker, 1)
                patient_summary = parts[0].strip()
                boilerplate = parts[1].strip() if len(parts) > 1 else ""
                return patient_summary, boilerplate
        return text.strip(), ""

    from datetime import date
    summary_generation_date = date.today().isoformat()

    print(f"\nBuilding full output ({total_chunks} chunks)...")

    # Build chunk_idx -> (first_date, last_date) lookup for efficient output building
    chunk_dates: Dict[int, Tuple[str, str]] = {}
    for rnd in rounds:
        for item in rnd:
            chunk_dates[item[1]] = (item[2], item[3])

    output_rows = []
    for pid, chunk_indices in patient_chunk_order.items():
        for i, cidx in enumerate(chunk_indices):
            if cidx in all_results:
                reasoning, summary, prior_summary = all_results[cidx]
            else:
                reasoning, summary, prior_summary = "", "", ""

            ps, bp = split_boilerplate(summary)
            first_date, last_date = chunk_dates.get(cidx, ("", ""))

            output_rows.append({
                args.patient_id_col: pid,
                "chunk_index": i,
                "first_date": first_date,
                "last_date": last_date,
                "prior_summary": prior_summary,
                "new_summary_reasoning": reasoning,
                "new_summary": summary,
                "patient_summary": ps,
                "patient_boilerplate_text": bp,
                "summary_generation_date": summary_generation_date,
            })

    output_df = pd.DataFrame(output_rows)
    output_df.to_parquet(args.output_parquet, index=False)
    print(f"Wrote {args.output_parquet} with {len(output_df)} rows.")

    # Create patient summaries output (one row per patient, using last chunk's summary)
    print("Building patient summaries...")
    patient_rows = []
    for pid, chunk_indices in patient_chunk_order.items():
        last_cidx = chunk_indices[-1]
        if last_cidx in all_results:
            _, summary, _ = all_results[last_cidx]
        else:
            summary = ""

        ps, bp = split_boilerplate(summary)

        patient_rows.append({
            args.patient_id_col: pid,
            "patient_summary": ps,
            "patient_boilerplate_text": bp,
            "last_note_date": patient_last_dates.get(pid, ""),
            "summary_generation_date": summary_generation_date,
        })

    patient_summaries_df = pd.DataFrame(patient_rows)
    patient_summaries_df.to_parquet(args.patient_summaries_parquet, index=False)
    print(f"Wrote {args.patient_summaries_parquet} with {len(patient_summaries_df)} patients.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
