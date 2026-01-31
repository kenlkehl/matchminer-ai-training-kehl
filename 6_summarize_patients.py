#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Serial patient summarization with iterative updates using vLLM server.

For each patient, notes are processed chronologically. Each new note updates
the prior summary. Work is scheduled in rounds to maximize GPU utilization.

This version uses a single vLLM server for all inference, eliminating the
overhead of loading/unloading the model for each round.

Examples
--------
# Basic usage (tensor parallelism inferred from gpu count)
python 6_summarize_patients.py \
  --input_parquet ../data/no_phi/all_synthetic_notes.parquet \
  --output_parquet ../data/no_phi/patient_serial_summaries.parquet \
  --shard_dir ../data/no_phi/summary_shards \
  --model openai/gpt-oss-120b \
  --download_dir /data1/ken/models \
  --gpu_ids 2,3 \
  --max_model_len 10000 \
  --generate_dates \
  --synthetic_start_date 2017-01-01 \
  --synthetic_min_days 7 \
  --synthetic_max_days 90 \
  --max_patients 10

# With custom port and concurrency
python 6_summarize_patients.py \
  --input_parquet ../data/no_phi/all_synthetic_notes.parquet \
  --output_parquet ../data/no_phi/patient_serial_summaries.parquet \
  --shard_dir ../data/no_phi/summary_shards \
  --model openai/gpt-oss-120b \
  --download_dir ../meta_ai \
  --gpu_ids 0,1,2,3 \
  --max_model_len 120000 \
  --port 8000 \
  --max_concurrent_requests 100
"""

import argparse
import asyncio
import glob
import os
import random
import re
import signal
import subprocess
import sys
import time
import warnings
from datetime import datetime, timedelta
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


def build_prompt_text(
    tokenizer,
    prior_summary: Optional[str],
    note_date: str,
    note_text: str,
    max_model_len: int,
    margin_tokens: int = 5000
) -> str:
    """
    Build a single prompt for iterative summarization.
    Truncates note_text if too long, keeping head & tail.
    """
    threshold = max(1024, max_model_len - margin_tokens)

    # Truncate note_text if needed
    toks = tokenizer(note_text, add_special_tokens=False).input_ids
    if len(toks) > threshold:
        half = threshold // 2
        first_part = toks[:half]
        last_part = toks[-half:]
        note_text = tokenizer.decode(first_part) + " ... " + tokenizer.decode(last_part)

    prior_summary_text = prior_summary if prior_summary else "None - this is the first note for this patient"

    user_content = f"""You are an experienced clinical oncology history summarization bot.

You are maintaining a running summary of a patient's cancer history based on their electronic health record.
You will be given:
1. A PRIOR SUMMARY of the patient's history (may be empty for first note)
2. A NEW CLINICAL NOTE to incorporate

Your task:
- Update the summary to incorporate any new relevant information from the new note
- If the new note contains no information that would change the summary, output the prior summary exactly as-is
- The patient may not yet have a cancer diagnosis. If not, state "No cancer diagnosis documented as of [date]" and summarize relevant medical history that might be relevant to a future oncology workup.

Document the patient's most recent age; sex; cancer type/primary site (eg breast cancer, lung cancer, etc); histology (eg adenocarcinoma, squamous carcinoma, etc); current extent (localized, advanced, metastatic, etc); biomarkers (genomic results, protein expression, etc); and treatment history (surgery, radiation, chemotherapy/targeted therapy/immunotherapy, etc, including start and stop dates and best response if known).
Do not consider localized basal cell or squamous carcinomas of the skin, or colon polyps, to be cancers for your purposes.
Do not include the patient's name, but do include relevant dates whenever documented.
If a patient has a history of more than one cancer, document the cancers one at a time.
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

NEW NOTE (dated {note_date}):
{note_text}
---
Now, write your updated summary. Do not add preceding text before the abstraction, and do not add commentary afterwards."""

    messages = [
        {'role': 'system', 'content': 'Reasoning: high'},
        {'role': 'user', 'content': user_content}
    ]

    prompt = tokenizer.apply_chat_template(
        conversation=messages,
        add_generation_prompt=True,
        tokenize=False
    )
    return prompt


def postprocess_output(raw_text: str) -> Tuple[str, str]:
    """
    Split output into reasoning (before 'assistantfinal') and summary (after).
    Returns (reasoning, summary).
    """
    reasoning_marker = "assistantfinal"

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

def prepare_rounds(
    df: pd.DataFrame,
    patient_id_col: str,
    date_col: str,
    text_col: str
) -> Tuple[List[List[Tuple[str, int, str, str]]], Dict[str, List[int]]]:
    """
    Organize work into rounds for parallel processing.

    Returns:
        rounds: List of rounds, each containing list of (patient_id, row_idx, date_str, note_text)
        patient_row_order: Dict mapping patient_id -> list of row indices in chronological order
    """
    # Sort by patient and date
    df = df.sort_values([patient_id_col, date_col]).reset_index(drop=True)

    # Group by patient and get ordered row indices
    patient_row_order: Dict[str, List[int]] = {}
    for idx, row in df.iterrows():
        pid = str(row[patient_id_col])
        if pid not in patient_row_order:
            patient_row_order[pid] = []
        patient_row_order[pid].append(idx)

    # Determine max notes per patient
    max_notes = max(len(rows) for rows in patient_row_order.values())

    # Build rounds: round i contains the (i+1)th note for each patient that has one
    rounds: List[List[Tuple[str, int, str, str]]] = []
    for round_idx in range(max_notes):
        round_items = []
        for pid, row_indices in patient_row_order.items():
            if round_idx < len(row_indices):
                row_idx = row_indices[round_idx]
                date_val = df.loc[row_idx, date_col]
                date_str = str(date_val) if pd.notna(date_val) else "unknown date"
                note_text = str(df.loc[row_idx, text_col])
                round_items.append((pid, row_idx, date_str, note_text))
        if round_items:
            rounds.append(round_items)

    return rounds, patient_row_order


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
    port: int = 8000,
    log_file: Optional[str] = None,
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
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--port", str(port),
    ]

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
    repetition_penalty: float,
    max_retries: int = 3,
    base_timeout: float = 600.0,
) -> Tuple[int, str, str]:
    """
    Send a single inference request with retry logic.
    Returns (row_idx, reasoning, summary).
    """
    for attempt in range(max_retries):
        try:
            response = await asyncio.wait_for(
                client.completions.create(
                    model=model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body={
                        "top_k": top_k,
                        "repetition_penalty": repetition_penalty,
                    }
                ),
                timeout=base_timeout
            )
            raw_text = response.choices[0].text
            reasoning, summary = postprocess_output(raw_text)
            return (row_idx, reasoning, summary)

        except asyncio.TimeoutError:
            wait_time = (2 ** attempt) * 5  # 5s, 10s, 20s
            if attempt < max_retries - 1:
                print(f"  Row {row_idx}: timeout (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                print(f"  Row {row_idx}: all retries exhausted (timeout)")
                return (row_idx, "", "ERROR: timeout after all retries")

        except Exception as e:
            wait_time = (2 ** attempt) * 2  # 2s, 4s, 8s
            if attempt < max_retries - 1:
                print(f"  Row {row_idx}: error '{e}' (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                print(f"  Row {row_idx}: all retries exhausted")
                return (row_idx, "", f"ERROR: {e}")

    return (row_idx, "", "ERROR: unexpected retry loop exit")


async def run_inference_batch(
    client: AsyncOpenAI,
    prompts: List[Tuple[int, str]],  # (row_idx, prompt_text)
    model: str,
    temperature: float,
    max_tokens: int,
    top_k: int,
    repetition_penalty: float,
    max_concurrent: int = 16,
    batch_size: int = 64,
    max_retries: int = 3,
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
    consecutive_health_failures = 0

    # Process in batches
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_prompts = prompts[batch_start:batch_end]

        # Check server health before each batch
        if not check_server_health(port):
            consecutive_health_failures += 1
            print(f"  WARNING: vLLM server health check failed (attempt {consecutive_health_failures}/3)")
            if consecutive_health_failures >= 3:
                print(f"  ERROR: vLLM server appears to be dead. Marking remaining {total - completed} prompts as errors.")
                # Mark remaining prompts as errors
                for idx, prompt in prompts[batch_start:]:
                    all_results.append((idx, "", "ERROR: vLLM server died"))
                return all_results
            # Wait and retry
            await asyncio.sleep(10)
            if not check_server_health(port):
                print(f"  ERROR: vLLM server still not responding after wait.")
                for idx, prompt in prompts[batch_start:]:
                    all_results.append((idx, "", "ERROR: vLLM server died"))
                return all_results
        else:
            consecutive_health_failures = 0

        print(f"  Processing batch {batch_start + 1}-{batch_end} of {total}...")

        # Use semaphore to limit concurrent requests within batch
        semaphore = asyncio.Semaphore(max_concurrent)

        async def bounded_request(row_idx: int, prompt: str) -> Tuple[int, str, str]:
            async with semaphore:
                return await single_inference_request(
                    client=client,
                    row_idx=row_idx,
                    prompt=prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    max_retries=max_retries,
                    base_timeout=base_timeout,
                )

        # Create tasks for this batch only
        tasks = [bounded_request(idx, prompt) for idx, prompt in batch_prompts]
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
    rounds: List[List[Tuple[str, int, str, str]]],
    patient_row_order: Dict[str, List[int]],
    patient_summaries: Dict[str, str],
    all_results: Dict[int, Tuple[str, str, str]],
    completed_rounds: int,
    args: argparse.Namespace,
    client: AsyncOpenAI,
    tokenizer
):
    """
    Process all remaining rounds using async inference.

    Args:
        rounds: List of rounds, each containing (patient_id, row_idx, date_str, note_text)
        patient_row_order: Dict mapping patient_id -> list of row indices
        patient_summaries: Dict tracking current summary per patient (mutated)
        all_results: Dict tracking all results by row_idx (mutated)
        completed_rounds: Number of rounds already completed
        args: Command line arguments
        client: AsyncOpenAI client
        tokenizer: HuggingFace tokenizer for prompt building
    """
    # Build reverse mapping: row_idx -> patient_id
    row_to_patient: Dict[int, str] = {}
    for pid, row_indices in patient_row_order.items():
        for row_idx in row_indices:
            row_to_patient[row_idx] = pid

    for round_idx in range(completed_rounds, len(rounds)):
        round_items = rounds[round_idx]
        print(f"\n=== Round {round_idx + 1}/{len(rounds)}: {len(round_items)} patients ===")

        # Build prompts with prior summaries
        prompts: List[Tuple[int, str]] = []
        round_prior_summaries: Dict[int, str] = {}

        for pid, row_idx, date_str, note_text in round_items:
            prior_summary = patient_summaries.get(pid, None)
            prior_text = prior_summary if prior_summary else "None - this is the first note for this patient"
            round_prior_summaries[row_idx] = prior_text

            prompt = build_prompt_text(
                tokenizer, prior_summary, date_str, note_text, args.max_model_len
            )
            prompts.append((row_idx, prompt))

        # Run inference
        print(f"Sending {len(prompts)} requests to vLLM server...")
        results = await run_inference_batch(
            client=client,
            prompts=prompts,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            max_concurrent=args.max_concurrent_requests,
            batch_size=args.batch_size,
            max_retries=args.max_retries,
            base_timeout=args.request_timeout,
            port=args.port,
        )

        # Update state
        round_results_with_prior: List[Tuple[int, str, str, str]] = []
        for row_idx, reasoning, summary in results:
            prior_text = round_prior_summaries[row_idx]
            all_results[row_idx] = (reasoning, summary, prior_text)
            round_results_with_prior.append((row_idx, reasoning, summary, prior_text))

            # Update patient summary for next round
            pid = row_to_patient[row_idx]
            patient_summaries[pid] = summary

        # Save round shard
        save_round_shard(args.shard_dir, round_idx, round_results_with_prior)
        print(f"Round {round_idx + 1} complete.")


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser("Serial patient summarization with iterative updates using vLLM server.")
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
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--download_dir", required=True)
    ap.add_argument("--gpu_ids", required=True,
                    help="Comma-separated list of GPU IDs (e.g., 0,1,2,3). Tensor parallel size is inferred from the count.")
    ap.add_argument("--max_model_len", type=int, default=120000)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_k", type=int, default=1)
    ap.add_argument("--max_tokens", type=int, default=7500)
    ap.add_argument("--repetition_penalty", type=float, default=1.2)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.93)
    ap.add_argument("--port", type=int, default=8000,
                    help="Port for vLLM server (default: 8000)")
    ap.add_argument("--max_concurrent_requests", type=int, default=16,
                    help="Maximum concurrent requests to vLLM server (default: 16)")
    ap.add_argument("--batch_size", type=int, default=64,
                    help="Number of prompts to process per batch before waiting (default: 64)")
    ap.add_argument("--request_timeout", type=float, default=600.0,
                    help="Timeout in seconds for individual inference requests (default: 600)")
    ap.add_argument("--max_retries", type=int, default=3,
                    help="Maximum retries for failed requests (default: 3)")
    ap.add_argument("--server_timeout", type=int, default=600,
                    help="Timeout in seconds waiting for vLLM server to start (default: 600)")
    ap.add_argument("--max_patients", type=int, default=None,
                    help="Limit to first N patients (for testing)")
    args = ap.parse_args()

    # Load data
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

    # Keep original index for output mapping
    df = df.reset_index(drop=True)
    original_len = len(df)

    # Prepare rounds
    print("Preparing work rounds...")
    rounds, patient_row_order = prepare_rounds(df, args.patient_id_col, args.date_col, args.text_col)
    print(f"Organized into {len(rounds)} rounds for {len(patient_row_order)} patients")

    # Load existing shards for resume
    completed_rounds, all_results = load_existing_shards(args.shard_dir)
    if completed_rounds > 0:
        print(f"Resuming from round {completed_rounds} (loaded {len(all_results)} existing results)")

    # Track current summaries per patient
    patient_summaries: Dict[str, str] = {}

    # Reconstruct patient summaries from completed rounds
    if completed_rounds > 0:
        for round_idx in range(completed_rounds):
            for pid, row_idx, _, _ in rounds[round_idx]:
                if row_idx in all_results:
                    _, summary, _ = all_results[row_idx]
                    patient_summaries[pid] = summary

    # Check if there's work to do
    if completed_rounds >= len(rounds):
        print("All rounds already completed. Building final output...")
    else:
        # Start vLLM server
        # Normalize gpu_ids to remove any semicolons and use comma format
        gpu_ids_normalized = args.gpu_ids.replace(";", ",")
        gpu_list = [g.strip() for g in gpu_ids_normalized.split(",") if g.strip()]
        tensor_parallel_size = len(gpu_list)

        server_process = start_vllm_server(
            model=args.model,
            download_dir=args.download_dir,
            gpu_ids=gpu_ids_normalized,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            port=args.port
        )

        try:
            # Wait for server to be ready
            if not wait_for_server_ready(args.port, timeout=args.server_timeout):
                print("Failed to start vLLM server. Exiting.")
                shutdown_server(server_process)
                sys.exit(1)

            # Create async OpenAI client with timeout
            client = AsyncOpenAI(
                base_url=f"http://localhost:{args.port}/v1",
                api_key="not-needed",  # vLLM doesn't require API key
                timeout=args.request_timeout + 60,  # Give extra buffer beyond request timeout
            )

            # Load tokenizer for prompt building
            print("Loading tokenizer...")
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                args.model,
                cache_dir=args.download_dir,
                trust_remote_code=True
            )

            # Process all rounds
            asyncio.run(process_all_rounds(
                rounds=rounds,
                patient_row_order=patient_row_order,
                patient_summaries=patient_summaries,
                all_results=all_results,
                completed_rounds=completed_rounds,
                args=args,
                client=client,
                tokenizer=tokenizer
            ))

        finally:
            # Always shutdown server
            shutdown_server(server_process)

    # Build final output
    print(f"\nBuilding final output ({original_len} rows)...")
    reasoning_list = []
    summary_list = []
    prior_summary_list = []

    for idx in range(original_len):
        if idx in all_results:
            reasoning, summary, prior_summary = all_results[idx]
        else:
            reasoning, summary, prior_summary = "", "", ""
        reasoning_list.append(reasoning)
        summary_list.append(summary)
        prior_summary_list.append(prior_summary)

    df["prior_summary"] = prior_summary_list
    df["new_summary_reasoning"] = reasoning_list
    df["new_summary"] = summary_list

    # Split new_summary into patient_summary and patient_boilerplate_text
    # The model is instructed to separate boilerplate with "Boilerplate:" label
    def split_boilerplate(text: str) -> Tuple[str, str]:
        """Split summary into main summary and boilerplate text."""
        if not text:
            return "", ""

        # Try different variations of the boilerplate marker
        markers = ["Boilerplate:", "BOILERPLATE:", "boilerplate:"]
        for marker in markers:
            if marker in text:
                parts = text.split(marker, 1)
                patient_summary = parts[0].strip()
                boilerplate = parts[1].strip() if len(parts) > 1 else ""
                return patient_summary, boilerplate

        # No boilerplate marker found - entire text is the summary
        return text.strip(), ""

    patient_summary_list = []
    boilerplate_list = []
    for summary in summary_list:
        ps, bp = split_boilerplate(summary)
        patient_summary_list.append(ps)
        boilerplate_list.append(bp)

    df["patient_summary"] = patient_summary_list
    df["patient_boilerplate_text"] = boilerplate_list

    # Add summary_generation_date (the date the script was run)
    from datetime import date
    summary_generation_date = date.today().isoformat()
    df["summary_generation_date"] = summary_generation_date

    # Save full output (all rows)
    df.to_parquet(args.output_parquet, index=False)
    print(f"Wrote {args.output_parquet} with {len(df)} rows.")

    # Create patient summaries output (last row per patient)
    # Data is already sorted by patient_id and date from prepare_rounds
    patient_summaries_df = df.groupby(args.patient_id_col).last().reset_index()
    # Rename date column to last_note_date for clarity
    if args.date_col in patient_summaries_df.columns:
        patient_summaries_df = patient_summaries_df.rename(columns={args.date_col: "last_note_date"})
    patient_summaries_df.to_parquet(args.patient_summaries_parquet, index=False)
    print(f"Wrote {args.patient_summaries_parquet} with {len(patient_summaries_df)} patients.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
