#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Serial patient summarization with chunk-based updates using Vertex AI MaaS.

For each patient, all clinical notes are sorted by date and concatenated into
a single text, then chunked into token-length segments. A running summary is
maintained across chunks. Work is scheduled in rounds (Round N = Nth chunk from
each patient).

Uses Google Vertex AI Model-as-a-Service endpoint for gpt-oss-120b instead
of local vLLM servers. Requires Google Cloud Application Default Credentials.

Examples
--------
# Basic usage (uses ADC and default project/region)
python 6a_summarize_patients_vertex.py \
  --input_parquet ../data/no_phi/all_synthetic_notes.parquet \
  --output_parquet ../data/no_phi/patient_serial_summaries.parquet \
  --shard_dir ../data/no_phi/summary_shards \
  --tokenizer openai/gpt-oss-120b \
  --download_dir /data1/ken/models \
  --max_model_len 120000 \
  --chunk_size 50000 \
  --chunk_overlap 500 \
  --generate_dates \
  --synthetic_start_date 2017-01-01 \
  --synthetic_min_days 7 \
  --synthetic_max_days 90 \
  --max_patients 10

# With explicit project and region
python 6a_summarize_patients_vertex.py \
  --input_parquet ../data/no_phi/all_synthetic_notes.parquet \
  --output_parquet ../data/no_phi/patient_serial_summaries.parquet \
  --shard_dir ../data/no_phi/summary_shards \
  --tokenizer openai/gpt-oss-120b \
  --download_dir /data1/ken/models \
  --project_id profile-notes \
  --region us-central1 \
  --max_model_len 120000 \
  --chunk_size 50000 \
  --chunk_overlap 500 \
  --max_concurrent_requests 8
"""

import argparse
import asyncio
import glob
import os
import random
import re
import sys
import time
import warnings
from datetime import datetime, timedelta
from multiprocessing import Pool
from typing import List, Dict, Tuple, Optional

import google.auth
import google.auth.transport.requests
import pandas as pd
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


def build_prompt_messages(
    tokenizer,
    prior_summary: Optional[str],
    first_date: str,
    last_date: str,
    chunk_text: str,
    max_model_len: int,
    margin_tokens: int = 5000,
    model_name: str = ""
) -> Tuple[List[Dict[str, str]], int]:
    """
    Build messages for iterative summarization via chat completions API.
    Truncates chunk_text if too long, keeping head & tail.

    Returns:
        (messages, prompt_token_count) tuple
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

    system_content = 'Reasoning: high' if 'gpt-oss' in model_name.lower() else ''
    messages = [
        {'role': 'system', 'content': system_content},
        {'role': 'user', 'content': user_content}
    ]

    # Count tokens using the tokenizer's chat template
    prompt_token_count = len(tokenizer.apply_chat_template(
        conversation=messages,
        add_generation_prompt=True,
        tokenize=True
    ))

    return messages, prompt_token_count


# Worker functions for parallel prompt building via multiprocessing.Pool.
# Each worker process loads its own tokenizer to avoid pickling issues.
_worker_tokenizer = None
_worker_model_name = None


def _init_prompt_worker(tokenizer_name, download_dir, model_name):
    """Initialize tokenizer in each worker process."""
    global _worker_tokenizer, _worker_model_name
    from transformers import AutoTokenizer
    _worker_tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, cache_dir=download_dir, trust_remote_code=True
    )
    _worker_model_name = model_name


def _build_prompt_worker(item):
    """Build a single prompt in a worker process. Returns (chunk_idx, messages, prompt_token_count)."""
    chunk_idx, prior_summary, first_date, last_date, chunk_text, max_model_len = item
    messages, prompt_token_count = build_prompt_messages(
        _worker_tokenizer, prior_summary, first_date, last_date, chunk_text, max_model_len,
        model_name=_worker_model_name
    )
    return (chunk_idx, messages, prompt_token_count)


def postprocess_output(raw_text: str, reasoning_marker: str = "assistantfinal") -> Tuple[str, str]:
    """
    Split output into reasoning (before reasoning_marker) and summary (after).
    Returns (reasoning, summary).
    """
    if reasoning_marker in raw_text:
        parts = raw_text.split(reasoning_marker, 1)
        reasoning = parts[0].strip()
        summary = parts[1].strip()
    else:
        # If no marker, treat entire output as summary
        reasoning = ""
        summary = raw_text.strip()

    return reasoning, summary


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
# Vertex AI Token Management
# -------------------------

def get_gcp_access_token() -> Tuple[str, float]:
    """
    Get a fresh GCP access token using Application Default Credentials.
    Returns (token_string, expiry_timestamp).
    """
    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    auth_request = google.auth.transport.requests.Request()
    credentials.refresh(auth_request)
    token = credentials.token
    expiry = credentials.expiry.timestamp() if credentials.expiry else time.time() + 3600
    return token, expiry


class TokenRefresher:
    """
    Async-safe token manager that refreshes GCP access tokens
    before they expire (with a configurable safety margin).
    """
    def __init__(self, refresh_margin_seconds: int = 300):
        self._token: Optional[str] = None
        self._expiry: float = 0.0
        self._refresh_margin = refresh_margin_seconds
        self._lock = asyncio.Lock()

    async def get_token(self) -> str:
        async with self._lock:
            if self._token is None or time.time() >= (self._expiry - self._refresh_margin):
                loop = asyncio.get_event_loop()
                self._token, self._expiry = await loop.run_in_executor(
                    None, get_gcp_access_token
                )
                print(f"Refreshed GCP access token (expires in {self._expiry - time.time():.0f}s)")
            return self._token


# -------------------------
# Async Inference
# -------------------------

async def single_inference_request(
    client: AsyncOpenAI,
    row_idx: int,
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float = 1.0,
    presence_penalty: float = 0.0,
    reasoning_marker: str = "assistantfinal",
    max_retries: int = 6,
    base_timeout: float = 600.0,
) -> Tuple[int, str, str]:
    """
    Send a single chat completion request with retry logic.
    Returns (row_idx, reasoning, summary).
    """
    for attempt in range(max_retries):
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    presence_penalty=presence_penalty,
                ),
                timeout=base_timeout
            )
            raw_text = response.choices[0].message.content
            reasoning, summary = postprocess_output(raw_text, reasoning_marker)
            return (row_idx, reasoning, summary)

        except asyncio.TimeoutError:
            wait_time = min((2 ** attempt) * 10, 300)  # 10s, 20s, 40s, 80s, 160s, 300s
            if attempt < max_retries - 1:
                print(f"  Row {row_idx}: timeout (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                print(f"  Row {row_idx}: all retries exhausted (timeout)")
                return (row_idx, "", "ERROR: timeout after all retries")

        except Exception as e:
            wait_time = min((2 ** attempt) * 5, 120)  # 5s, 10s, 20s, 40s, 80s, 120s
            if attempt < max_retries - 1:
                print(f"  Row {row_idx}: error '{e}' (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                print(f"  Row {row_idx}: all retries exhausted")
                return (row_idx, "", f"ERROR: {e}")

    return (row_idx, "", "ERROR: unexpected retry loop exit")


async def run_inference_batch(
    client: AsyncOpenAI,
    prompts: List[Tuple[int, List[Dict[str, str]], int]],  # (row_idx, messages, max_tokens)
    model: str,
    temperature: float,
    top_p: float = 1.0,
    presence_penalty: float = 0.0,
    reasoning_marker: str = "assistantfinal",
    max_concurrent: int = 16,
    batch_size: int = 64,
    max_retries: int = 6,
    base_timeout: float = 600.0,
    token_refresher: Optional[TokenRefresher] = None,
) -> List[Tuple[int, str, str]]:
    """
    Send batch of requests concurrently, return (row_idx, reasoning, summary).

    Processes prompts in smaller batches to avoid overwhelming the endpoint.
    Each batch runs max_concurrent requests in parallel.
    Refreshes GCP access token before each sub-batch if needed.
    """
    total = len(prompts)
    all_results: List[Tuple[int, str, str]] = []
    completed = 0

    # Process in batches
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_prompts = prompts[batch_start:batch_end]

        # Refresh token before each sub-batch
        if token_refresher is not None:
            fresh_token = await token_refresher.get_token()
            client.api_key = fresh_token

        print(f"  Processing batch {batch_start + 1}-{batch_end} of {total}...")

        # Use semaphore to limit concurrent requests within batch
        semaphore = asyncio.Semaphore(max_concurrent)

        async def bounded_request(row_idx: int, messages: List[Dict[str, str]], prompt_max_tokens: int) -> Tuple[int, str, str]:
            async with semaphore:
                return await single_inference_request(
                    client=client,
                    row_idx=row_idx,
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=prompt_max_tokens,
                    top_p=top_p,
                    presence_penalty=presence_penalty,
                    reasoning_marker=reasoning_marker,
                    max_retries=max_retries,
                    base_timeout=base_timeout,
                )

        # Create tasks for this batch only
        tasks = [bounded_request(idx, msgs, mt) for idx, msgs, mt in batch_prompts]
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
    client: AsyncOpenAI,
    token_refresher: TokenRefresher,
    prompt_pool: Pool,
):
    """
    Process all remaining rounds using async inference against Vertex AI.

    Args:
        rounds: List of rounds, each containing (patient_id, chunk_idx, first_date, last_date, chunk_text)
        patient_chunk_order: Dict mapping patient_id -> list of chunk indices
        patient_summaries: Dict tracking current summary per patient (mutated)
        all_results: Dict tracking all results by chunk_idx (mutated)
        completed_rounds: Number of rounds already completed
        args: Command line arguments
        client: AsyncOpenAI client for Vertex AI
        token_refresher: Token refresher for GCP auth
        prompt_pool: Multiprocessing pool for parallel prompt building
    """
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
        prompt_results: List[Tuple[int, List[Dict[str, str]], int]] = list(
            prompt_pool.map(_build_prompt_worker, work_items, chunksize=chunksize)
        )

        # Compute per-prompt max_tokens: max_model_len - prompt_token_count
        prompts: List[Tuple[int, List[Dict[str, str]], int]] = []
        for chunk_idx, messages, prompt_token_count in prompt_results:
            gen_tokens = args.max_model_len - prompt_token_count
            if args.max_tokens is not None:
                gen_tokens = min(gen_tokens, args.max_tokens)
            gen_tokens = max(gen_tokens, 1)  # safety floor
            prompts.append((chunk_idx, messages, gen_tokens))
        print(f"Prompts built.")

        print(f"Sending {len(prompts)} requests to Vertex AI...")

        # Single call to run_inference_batch (no multi-server distribution)
        results = await run_inference_batch(
            client=client,
            prompts=prompts,
            model=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
            presence_penalty=args.presence_penalty,
            reasoning_marker=args.reasoning_marker,
            max_concurrent=args.max_concurrent_requests,
            batch_size=args.batch_size,
            max_retries=args.max_retries,
            base_timeout=args.request_timeout,
            token_refresher=token_refresher,
        )

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


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser("Chunk-based patient summarization with iterative updates using Vertex AI MaaS.")

    # Data arguments
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
    ap.add_argument("--chunk_size", type=int, default=10000,
                    help="Maximum tokens per chunk when concatenating patient notes (default: 10000)")
    ap.add_argument("--chunk_overlap", type=int, default=500,
                    help="Token overlap between consecutive chunks (default: 500)")

    # Model / Vertex AI arguments
    ap.add_argument("--model", default="gpt-oss-120b-maas",
                    help="Model ID for the Vertex AI MaaS endpoint (default: gpt-oss-120b-maas)")
    ap.add_argument("--tokenizer", default="openai/gpt-oss-120b",
                    help="HuggingFace tokenizer name for token counting and chunking (default: openai/gpt-oss-120b)")
    ap.add_argument("--download_dir", required=True,
                    help="Directory for caching the HuggingFace tokenizer")
    ap.add_argument("--project_id", default="profile-notes",
                    help="Google Cloud project ID (default: profile-notes)")
    ap.add_argument("--region", default="us-central1",
                    help="Vertex AI region (default: us-central1)")

    # Generation parameters
    ap.add_argument("--max_model_len", type=int, default=120000)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--presence_penalty", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=15000,
                    help="Max generation tokens per prompt. If not set, auto-computed as max_model_len minus prompt token count.")
    ap.add_argument("--reasoning_marker", type=str, default="assistantfinal",
                    help="Marker string that separates reasoning from final summary in model output (default: assistantfinal)")

    # Request handling
    ap.add_argument("--max_concurrent_requests", type=int, default=16,
                    help="Maximum concurrent requests to Vertex AI endpoint (default: 16)")
    ap.add_argument("--batch_size", type=int, default=1000,
                    help="Number of prompts to process per batch before waiting (default: 1000)")
    ap.add_argument("--request_timeout", type=float, default=600.0,
                    help="Timeout in seconds for individual inference requests (default: 600)")
    ap.add_argument("--max_retries", type=int, default=6,
                    help="Maximum retries for failed requests (default: 6)")
    ap.add_argument("--token_refresh_margin", type=int, default=300,
                    help="Seconds before GCP token expiry to trigger refresh (default: 300)")

    # Misc
    ap.add_argument("--max_patients", type=int, default=None,
                    help="Limit to first N patients (for testing)")
    ap.add_argument("--run_deterministic", action="store_true",
                    help="Override LLM parameters for deterministic output: "
                         "temperature=0.0, max_model_len=120000, chunk_size=50000, max_tokens=10000")
    args = ap.parse_args()

    # Apply deterministic overrides
    if args.run_deterministic:
        args.temperature = 0.0
        args.max_model_len = 120000
        args.chunk_size = 50000
        args.max_tokens = 10000
        print("Deterministic mode: temperature=0.0, "
              "max_model_len=120000, chunk_size=50000, max_tokens=10000")

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

        # Load tokenizer for chunking (needed by prepare_rounds)
        print("Loading tokenizer...")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer,
            cache_dir=args.download_dir,
            trust_remote_code=True
        )

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
        # Ensure shard_dir exists
        os.makedirs(args.shard_dir, exist_ok=True)

        # Create multiprocessing pool for parallel prompt building
        n_workers = min(os.cpu_count() or 4, 32)
        print(f"Creating prompt-building pool with {n_workers} workers...")
        prompt_pool = Pool(
            processes=n_workers,
            initializer=_init_prompt_worker,
            initargs=(args.tokenizer, args.download_dir, args.model),
        )

        # Set up Vertex AI client
        base_url = (
            f"https://aiplatform.googleapis.com/v1/projects/{args.project_id}"
            f"/locations/{args.region}/endpoints/openapi"
        )
        print(f"Vertex AI endpoint: {base_url}")

        token_refresher = TokenRefresher(refresh_margin_seconds=args.token_refresh_margin)

        # Get initial token synchronously
        initial_token, initial_expiry = get_gcp_access_token()
        token_refresher._token = initial_token
        token_refresher._expiry = initial_expiry
        print(f"Initial GCP access token obtained (expires in {initial_expiry - time.time():.0f}s)")

        client = AsyncOpenAI(
            base_url=base_url,
            api_key=initial_token,
            timeout=args.request_timeout + 60,
        )

        try:
            # Process all rounds
            asyncio.run(process_all_rounds(
                rounds=rounds,
                patient_chunk_order=patient_chunk_order,
                patient_summaries=patient_summaries_dict,
                all_results=all_results,
                completed_rounds=completed_rounds,
                args=args,
                client=client,
                token_refresher=token_refresher,
                prompt_pool=prompt_pool,
            ))

        finally:
            prompt_pool.close()
            prompt_pool.join()

    # Build final output (one row per chunk per patient)
    def split_boilerplate(text: str) -> Tuple[str, str]:
        """Split summary into main summary and boilerplate text."""
        if not text:
            return "", ""
        markers = ["Boilerplate:", "BOILERPLATE:", "boilerplate:"]
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
