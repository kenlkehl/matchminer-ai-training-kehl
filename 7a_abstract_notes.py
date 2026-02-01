#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Note-level clinical abstraction with structured JSON output.

Abstracts each individual patient note into a list of JSON dicts representing
clinical concepts (age, sex, cancer_type, histology, burden_of_disease,
biomarker, treatment, boilerplate_exclusion).

Supports two inference modes:
- Python API mode (default): Uses vLLM LLM class with ProcessPoolExecutor
- Server mode (--use_server): Uses AsyncOpenAI client with vLLM server

Examples
--------
# Python API mode (default) - multi-GPU parallelization
python 7a_abstract_notes.py \
    --input_parquet ../data/no_phi/all_synthetic_notes.parquet \
    --output_parquet ../data/no_phi/note_abstractions.parquet \
    --shard_dir ../data/no_phi/abstraction_shards \
    --model openai/gpt-oss-120b \
    --download_dir /data1/ken/models \
    --gpu_ids 0,1,2,3 \
    --max_model_len 10000 \
    --max_patients 10

# Server mode - uses vLLM OpenAI-compatible server
python 7a_abstract_notes.py \
    --input_parquet ../data/no_phi/all_synthetic_notes.parquet \
    --output_parquet ../data/no_phi/note_abstractions.parquet \
    --shard_dir ../data/no_phi/abstraction_shards \
    --model openai/gpt-oss-120b \
    --download_dir /data1/ken/models \
    --gpu_ids 0,1,2,3 \
    --max_model_len 10000 \
    --use_server \
    --port 8000
"""

import argparse
import asyncio
import glob
import json
import os
import random
import re
import subprocess
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import multiprocessing as mp

import pandas as pd
import requests


# -------------------------
# Prompts
# -------------------------

NOTE_ABSTRACTION_PROMPT = """You are extracting structured clinical information from a patient note.

Output a JSON array of objects, one per clinical concept found in the note.
Each object must have a "concept" field with one of: "age", "sex", "cancer_type",
"histology", "burden_of_disease", "biomarker", "treatment", "boilerplate_exclusion".

IMPORTANT: For each concept, include an "event_date" field if the note mentions when the event occurred, was diagnosed, or was documented. Extract dates exactly as written in the note (e.g., "2023-05-15", "May 2023", "3/2022").

Additional fields per concept type:
- age: "value" (numeric string), "event_date" (when age was recorded, if mentioned)
- sex: "value" ("Male", "Female", "Unknown")
- cancer_type: "value", "primary_site", "event_date" (diagnosis date if mentioned)
- histology: "value", "event_date" (when determined, if mentioned)
- burden_of_disease: "value" (localized/advanced/metastatic), "sites" (array of metastatic sites), "event_date" (when staging was done, if mentioned)
- biomarker: "name", "result", "event_date" (test date if mentioned)
- treatment: "type" (surgery/radiation/systemic), "regimen", "start_date", "end_date", "response"
- boilerplate_exclusion: "condition", "status", "details", "event_date" (when condition was noted, if mentioned)

If the note contains no relevant clinical information, output an empty array: []

Output ONLY valid JSON. No markdown, no explanation, no text before or after.

NOTE TEXT:
{note_text}"""


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
    """Generate synthetic dates for notes when no date column exists."""
    df = df.copy()
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")

    dates = []
    current_patient = None
    current_date = start_date

    for idx, row in df.iterrows():
        pid = row[patient_id_col]

        if pid != current_patient:
            current_patient = pid
            current_date = start_date
        else:
            days_to_add = random.randint(min_days, max_days)
            current_date = current_date + timedelta(days=days_to_add)

        dates.append(current_date)

    df[date_col] = dates
    return df


def build_abstraction_prompt(
    tokenizer,
    note_text: str,
    max_model_len: int,
    margin_tokens: int = 5000
) -> str:
    """Build prompt for note abstraction. Truncates note_text if too long."""
    threshold = max(1024, max_model_len - margin_tokens)

    # Truncate note_text if needed
    toks = tokenizer(note_text, add_special_tokens=False).input_ids
    if len(toks) > threshold:
        half = threshold // 2
        first_part = toks[:half]
        last_part = toks[-half:]
        note_text = tokenizer.decode(first_part) + " ... " + tokenizer.decode(last_part)

    user_content = NOTE_ABSTRACTION_PROMPT.format(note_text=note_text)

    messages = [
        {'role': 'system', 'content': 'You are a clinical information extraction assistant. Output only valid JSON.'},
        {'role': 'user', 'content': user_content}
    ]

    prompt = tokenizer.apply_chat_template(
        conversation=messages,
        add_generation_prompt=True,
        tokenize=False
    )
    return prompt


def is_valid_json(text: str) -> Tuple[bool, Optional[List[Dict]]]:
    """
    Validate that text is valid JSON array.
    Returns (is_valid, parsed_data or None).
    """
    if not text:
        return False, None

    # Clean potential markdown code blocks
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return True, data
        return False, None
    except json.JSONDecodeError:
        return False, None


def inject_note_date(concepts: List[Dict], note_date: str) -> List[Dict]:
    """Add note_date field to each concept dict."""
    for concept in concepts:
        concept['note_date'] = note_date
    return concepts


def postprocess_output(raw_text: str) -> str:
    """
    Extract JSON from output, handling reasoning model output.
    Looks for 'assistantfinal' marker if present.
    """
    reasoning_marker = "assistantfinal"

    if reasoning_marker in raw_text:
        parts = raw_text.split(reasoning_marker, 1)
        return parts[1].strip()
    else:
        return raw_text.strip()


# -------------------------
# Sharding / Resume
# -------------------------

def get_shard_filename(temp_dir: str, batch_start: int, batch_end: int) -> str:
    """Generate consistent shard filename."""
    return os.path.join(temp_dir, f"shard_{batch_start:08d}_{batch_end:08d}.parquet")


def get_completed_shards(temp_dir: str) -> set:
    """Get set of (batch_start, batch_end) tuples for completed shards."""
    completed = set()
    if not os.path.exists(temp_dir):
        return completed

    pattern = re.compile(r"shard_(\d+)_(\d+)\.parquet")
    for filename in os.listdir(temp_dir):
        match = pattern.match(filename)
        if match:
            batch_start = int(match.group(1))
            batch_end = int(match.group(2))
            filepath = os.path.join(temp_dir, filename)
            try:
                df = pd.read_parquet(filepath)
                if len(df) > 0:
                    completed.add((batch_start, batch_end))
            except Exception:
                pass

    return completed


def load_existing_shards(shard_dir: str) -> Dict[int, Dict[str, Any]]:
    """
    Load existing shard files to enable resume.

    Returns:
        Dict mapping row_idx -> {json_str, parsed_list, valid, retry_count}
    """
    results: Dict[int, Dict[str, Any]] = {}

    if not os.path.exists(shard_dir):
        return results

    shard_files = sorted(glob.glob(os.path.join(shard_dir, "shard_*.parquet")))
    for shard_file in shard_files:
        try:
            shard_df = pd.read_parquet(shard_file)
            for _, row in shard_df.iterrows():
                row_idx = int(row["row_idx"])
                results[row_idx] = {
                    "json_str": str(row["note_abstraction_json"]) if pd.notna(row["note_abstraction_json"]) else "",
                    "parsed": str(row["note_abstraction"]) if pd.notna(row["note_abstraction"]) else "[]",
                    "valid": bool(row["abstraction_valid"]) if pd.notna(row["abstraction_valid"]) else False,
                    "retry_count": int(row["abstraction_retry_count"]) if pd.notna(row["abstraction_retry_count"]) else 0,
                }
        except Exception as e:
            print(f"Warning: Could not load shard {shard_file}: {e}")

    return results


def save_batch_shard(
    shard_dir: str,
    batch_start: int,
    batch_end: int,
    results: List[Dict]
):
    """Save results from a batch to a shard file."""
    os.makedirs(shard_dir, exist_ok=True)
    shard_path = get_shard_filename(shard_dir, batch_start, batch_end)

    shard_df = pd.DataFrame(results)
    shard_df.to_parquet(shard_path, index=False)
    print(f"Saved shard: {shard_path} ({len(results)} records)")


# -------------------------
# vLLM Server Management (for --use_server mode)
# -------------------------

def check_existing_server(port: int, model: str) -> Tuple[bool, bool]:
    """
    Check if a vLLM server is already running on the specified port.

    Args:
        port: Port to check
        model: Expected model name/path

    Returns:
        Tuple of (server_running, model_compatible)
        - server_running: True if a server is responding on the port
        - model_compatible: True if the server is running the expected model
    """
    try:
        # Check health endpoint
        health_response = requests.get(f"http://localhost:{port}/health", timeout=5)
        if health_response.status_code != 200:
            return False, False

        # Check models endpoint to verify the model
        models_response = requests.get(f"http://localhost:{port}/v1/models", timeout=5)
        if models_response.status_code != 200:
            return True, False

        models_data = models_response.json()
        if "data" in models_data and len(models_data["data"]) > 0:
            # Get the model ID from the server
            server_model = models_data["data"][0].get("id", "")

            # Check if model matches (handle both full path and model name)
            # The server might report just the model name or the full path
            model_name = os.path.basename(model.rstrip("/"))
            server_model_name = os.path.basename(server_model.rstrip("/"))

            if model == server_model or model_name == server_model_name:
                return True, True
            else:
                print(f"  Server running model '{server_model}', expected '{model}'")
                return True, False

        return True, False

    except requests.exceptions.RequestException:
        return False, False


def start_vllm_server(
    model: str,
    download_dir: str,
    gpu_ids: str,
    tensor_parallel_size: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    port: int = 8000,
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

    process = subprocess.Popen(cmd, env=env)
    return process


def wait_for_server_ready(port: int, timeout: int = 600, poll_interval: float = 5.0) -> bool:
    """Poll health endpoint until server is ready."""
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


def shutdown_server(process: subprocess.Popen, timeout: int = 30):
    """Gracefully terminate the server subprocess."""
    if process is None:
        return

    print("Shutting down vLLM server...")
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
# Server Mode Async Inference
# -------------------------

async def single_inference_request(
    client,
    row_idx: int,
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    max_retries: int = 3,
    base_timeout: float = 300.0,
) -> Tuple[int, str]:
    """Send a single inference request with retry logic."""
    for attempt in range(max_retries):
        try:
            response = await asyncio.wait_for(
                client.completions.create(
                    model=model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ),
                timeout=base_timeout
            )
            raw_text = response.choices[0].text
            return (row_idx, postprocess_output(raw_text))

        except asyncio.TimeoutError:
            wait_time = (2 ** attempt) * 5
            if attempt < max_retries - 1:
                print(f"  Row {row_idx}: timeout (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                print(f"  Row {row_idx}: all retries exhausted (timeout)")
                return (row_idx, "")

        except Exception as e:
            wait_time = (2 ** attempt) * 2
            if attempt < max_retries - 1:
                print(f"  Row {row_idx}: error '{e}' (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                print(f"  Row {row_idx}: all retries exhausted")
                return (row_idx, "")

    return (row_idx, "")


async def run_inference_batch_async(
    client,
    prompts: List[Tuple[int, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    max_concurrent: int = 16,
) -> List[Tuple[int, str]]:
    """Send batch of requests concurrently."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded_request(row_idx: int, prompt: str) -> Tuple[int, str]:
        async with semaphore:
            return await single_inference_request(
                client=client,
                row_idx=row_idx,
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    tasks = [bounded_request(idx, prompt) for idx, prompt in prompts]
    results = await asyncio.gather(*tasks)
    return list(results)


# -------------------------
# Python API Mode (ProcessPoolExecutor)
# -------------------------

def worker_process(
    worker_id: int,
    gpu_ids: list,
    batches: list,
    df: pd.DataFrame,
    args: argparse.Namespace,
):
    """
    Worker process that runs vLLM inference on assigned batches.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

    from vllm import LLM, SamplingParams

    print(f"[Worker {worker_id}] Starting on GPUs {gpu_ids}, processing {len(batches)} batches")

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        download_dir=args.download_dir,
    )

    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    for batch_start, batch_end in batches:
        shard_path = get_shard_filename(args.shard_dir, batch_start, batch_end)

        if os.path.exists(shard_path):
            try:
                test_df = pd.read_parquet(shard_path)
                if len(test_df) > 0:
                    print(f"[Worker {worker_id}] Skipping batch {batch_start}-{batch_end} (already exists)")
                    continue
            except Exception:
                pass

        print(f"[Worker {worker_id}] Processing batch {batch_start}-{batch_end}")

        batch_df = df.iloc[batch_start:batch_end].copy()

        # Build prompts
        prompts = []
        row_indices = []
        note_dates = []
        for idx, (orig_idx, row) in enumerate(batch_df.iterrows()):
            note_text = str(row[args.text_col])
            date_val = row[args.date_col]
            if pd.notna(date_val):
                date_str = str(pd.Timestamp(date_val).date())
            else:
                date_str = "unknown"

            prompt = build_abstraction_prompt(tokenizer, note_text, args.max_model_len)
            prompts.append(prompt)
            row_indices.append(batch_start + idx)
            note_dates.append(date_str)

        # Run inference with retries for invalid JSON
        current_prompts = list(zip(row_indices, prompts, note_dates))
        results_dict: Dict[int, Dict] = {}

        for retry_round in range(args.max_retries + 1):
            if not current_prompts:
                break

            prompts_only = [p for _, p, _ in current_prompts]
            responses = llm.generate(prompts_only, sampling_params)

            invalid_indices = []
            for i, (row_idx, prompt, note_date) in enumerate(current_prompts):
                raw_text = responses[i].outputs[0].text
                json_text = postprocess_output(raw_text)

                valid, parsed = is_valid_json(json_text)

                if valid:
                    # Inject note_date into each concept
                    parsed = inject_note_date(parsed, note_date)
                    results_dict[row_idx] = {
                        "row_idx": row_idx,
                        "note_abstraction_json": json_text,
                        "note_abstraction": json.dumps(parsed),
                        "abstraction_valid": True,
                        "abstraction_retry_count": retry_round,
                    }
                else:
                    invalid_indices.append((row_idx, prompt, note_date))

            if invalid_indices:
                if retry_round < args.max_retries:
                    print(f"[Worker {worker_id}] Batch {batch_start}-{batch_end}: {len(invalid_indices)} invalid JSON, retry {retry_round + 1}")
                current_prompts = invalid_indices
            else:
                current_prompts = []

        # Mark remaining invalid ones
        for row_idx, prompt, note_date in current_prompts:
            results_dict[row_idx] = {
                "row_idx": row_idx,
                "note_abstraction_json": "",
                "note_abstraction": "[]",
                "abstraction_valid": False,
                "abstraction_retry_count": args.max_retries,
            }

        # Save shard
        results_list = [results_dict[batch_start + i] for i in range(batch_end - batch_start)]
        save_batch_shard(args.shard_dir, batch_start, batch_end, results_list)

    print(f"[Worker {worker_id}] Finished all batches")
    return worker_id


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser("Note-level clinical abstraction with JSON output.")
    ap.add_argument("--input_parquet", required=True, help="Input parquet file")
    ap.add_argument("--output_parquet", required=True, help="Output parquet with note abstractions")
    ap.add_argument("--shard_dir", required=True, help="Directory for checkpoint shards")
    ap.add_argument("--patient_id_col", default="pseudo_mrn", help="Column name for patient ID")
    ap.add_argument("--date_col", default="date", help="Column name for note date")
    ap.add_argument("--text_col", default="synthetic_note", help="Column name for note text")
    ap.add_argument("--generate_dates", action="store_true",
                    help="Generate synthetic dates if date column is missing")
    ap.add_argument("--synthetic_start_date", default="2020-01-01",
                    help="Start date for synthetic dates (YYYY-MM-DD)")
    ap.add_argument("--synthetic_min_days", type=int, default=7,
                    help="Minimum days between consecutive notes")
    ap.add_argument("--synthetic_max_days", type=int, default=90,
                    help="Maximum days between consecutive notes")
    ap.add_argument("--model", default="openai/gpt-oss-120b", help="Model path or HuggingFace ID")
    ap.add_argument("--download_dir", required=True, help="Model download directory")
    ap.add_argument("--gpu_ids", required=True,
                    help="Comma-separated GPU IDs (e.g., 0,1,2,3)")
    ap.add_argument("--tensor_parallel_size", type=int, default=1,
                    help="Tensor parallel size per vLLM instance (Python API mode)")
    ap.add_argument("--max_model_len", type=int, default=10000, help="Max model context length")
    ap.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    ap.add_argument("--max_tokens", type=int, default=4000, help="Max tokens to generate")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.93,
                    help="GPU memory utilization fraction")
    ap.add_argument("--max_retries", type=int, default=5,
                    help="Max JSON validation retries (default: 5)")
    ap.add_argument("--batch_size", type=int, default=100,
                    help="Batch size for processing")
    ap.add_argument("--max_patients", type=int, default=None,
                    help="Limit to first N patients (for testing)")

    # Server mode arguments
    ap.add_argument("--use_server", action="store_true",
                    help="Use vLLM server instead of Python API")
    ap.add_argument("--port", type=int, default=8000,
                    help="vLLM server port (default: 8000)")
    ap.add_argument("--max_concurrent_requests", type=int, default=16,
                    help="Max concurrent requests (server mode)")
    ap.add_argument("--server_timeout", type=int, default=600,
                    help="Timeout waiting for vLLM server to start")

    args = ap.parse_args()

    # Load data
    print(f"Loading {args.input_parquet}...")
    df = pd.read_parquet(args.input_parquet)

    # Validate columns
    if args.patient_id_col not in df.columns:
        raise ValueError(f"Column '{args.patient_id_col}' not found. Available: {df.columns.tolist()}")
    if args.text_col not in df.columns:
        raise ValueError(f"Column '{args.text_col}' not found. Available: {df.columns.tolist()}")

    # Handle date column
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
        else:
            raise ValueError(
                f"Column '{args.date_col}' not found. Use --generate_dates to create synthetic dates."
            )

    # Filter to max_patients if specified
    if args.max_patients is not None:
        unique_patients = df[args.patient_id_col].unique()[:args.max_patients]
        df = df[df[args.patient_id_col].isin(unique_patients)].copy()
        print(f"Limited to {args.max_patients} patients ({len(df)} rows)")

    df = df.reset_index(drop=True)
    original_len = len(df)
    print(f"Total rows to process: {original_len}")

    # Create shard directory
    os.makedirs(args.shard_dir, exist_ok=True)

    # Parse GPU IDs
    gpu_ids_normalized = args.gpu_ids.replace(";", ",")
    gpu_list = [int(g.strip()) for g in gpu_ids_normalized.split(",") if g.strip()]

    if args.use_server:
        # Server mode
        tensor_parallel_size = len(gpu_list)

        # Load existing results
        existing_results = load_existing_shards(args.shard_dir)
        if existing_results:
            print(f"Loaded {len(existing_results)} existing results from shards")

        # Find rows that need processing
        rows_to_process = [i for i in range(original_len) if i not in existing_results]

        if not rows_to_process:
            print("All rows already processed. Building final output...")
        else:
            print(f"Processing {len(rows_to_process)} remaining rows in server mode")

            # Check for existing server
            server_running, model_compatible = check_existing_server(args.port, args.model)
            server_process = None
            we_started_server = False

            if server_running and model_compatible:
                print(f"Found existing vLLM server on port {args.port} with compatible model. Using it.")
            elif server_running and not model_compatible:
                print(f"ERROR: Server on port {args.port} is running a different model.")
                print("Please stop the existing server or use a different port.")
                sys.exit(1)
            else:
                print(f"No existing server found on port {args.port}. Starting new server...")
                server_process = start_vllm_server(
                    model=args.model,
                    download_dir=args.download_dir,
                    gpu_ids=gpu_ids_normalized,
                    tensor_parallel_size=tensor_parallel_size,
                    max_model_len=args.max_model_len,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    port=args.port
                )
                we_started_server = True

                if not wait_for_server_ready(args.port, timeout=args.server_timeout):
                    print("Failed to start vLLM server. Exiting.")
                    shutdown_server(server_process)
                    sys.exit(1)

            try:

                from openai import AsyncOpenAI
                client = AsyncOpenAI(
                    base_url=f"http://localhost:{args.port}/v1",
                    api_key="not-needed",
                    timeout=300.0,
                )

                from transformers import AutoTokenizer
                print("Loading tokenizer...")
                tokenizer = AutoTokenizer.from_pretrained(
                    args.model,
                    cache_dir=args.download_dir,
                    trust_remote_code=True
                )

                # Process in batches
                for batch_start in range(0, len(rows_to_process), args.batch_size):
                    batch_end = min(batch_start + args.batch_size, len(rows_to_process))
                    batch_indices = rows_to_process[batch_start:batch_end]

                    print(f"Processing batch {batch_start}-{batch_end} of {len(rows_to_process)}")

                    # Build prompts
                    prompts = []
                    note_dates = []
                    for row_idx in batch_indices:
                        row = df.iloc[row_idx]
                        note_text = str(row[args.text_col])
                        date_val = row[args.date_col]
                        if pd.notna(date_val):
                            date_str = str(pd.Timestamp(date_val).date())
                        else:
                            date_str = "unknown"

                        prompt = build_abstraction_prompt(tokenizer, note_text, args.max_model_len)
                        prompts.append((row_idx, prompt))
                        note_dates.append(date_str)

                    # Run inference with retries
                    current_prompts = list(zip(batch_indices, [p[1] for p in prompts], note_dates))
                    batch_results: Dict[int, Dict] = {}

                    for retry_round in range(args.max_retries + 1):
                        if not current_prompts:
                            break

                        async_prompts = [(idx, p) for idx, p, _ in current_prompts]
                        results = asyncio.run(run_inference_batch_async(
                            client=client,
                            prompts=async_prompts,
                            model=args.model,
                            temperature=args.temperature,
                            max_tokens=args.max_tokens,
                            max_concurrent=args.max_concurrent_requests,
                        ))

                        results_map = {idx: text for idx, text in results}
                        invalid_indices = []

                        for row_idx, prompt, note_date in current_prompts:
                            json_text = results_map.get(row_idx, "")
                            valid, parsed = is_valid_json(json_text)

                            if valid:
                                parsed = inject_note_date(parsed, note_date)
                                batch_results[row_idx] = {
                                    "row_idx": row_idx,
                                    "note_abstraction_json": json_text,
                                    "note_abstraction": json.dumps(parsed),
                                    "abstraction_valid": True,
                                    "abstraction_retry_count": retry_round,
                                }
                            else:
                                invalid_indices.append((row_idx, prompt, note_date))

                        if invalid_indices and retry_round < args.max_retries:
                            print(f"  {len(invalid_indices)} invalid JSON, retry {retry_round + 1}")
                        current_prompts = invalid_indices

                    # Mark remaining invalid
                    for row_idx, prompt, note_date in current_prompts:
                        batch_results[row_idx] = {
                            "row_idx": row_idx,
                            "note_abstraction_json": "",
                            "note_abstraction": "[]",
                            "abstraction_valid": False,
                            "abstraction_retry_count": args.max_retries,
                        }

                    # Update existing_results
                    existing_results.update({r["row_idx"]: r for r in batch_results.values()})

                    # Save batch shard
                    results_list = [batch_results[idx] for idx in batch_indices]
                    first_idx = batch_indices[0]
                    last_idx = batch_indices[-1] + 1
                    save_batch_shard(args.shard_dir, first_idx, last_idx, results_list)

            finally:
                # Only shutdown if we started the server
                if we_started_server and server_process is not None:
                    shutdown_server(server_process)

    else:
        # Python API mode with ProcessPoolExecutor
        num_gpus = len(gpu_list)
        tp_size = args.tensor_parallel_size

        if num_gpus % tp_size != 0:
            print(f"Error: Number of GPUs ({num_gpus}) must be divisible by tensor_parallel_size ({tp_size})")
            sys.exit(1)

        num_workers = num_gpus // tp_size
        print(f"Configuration: {num_gpus} GPUs, tensor_parallel_size={tp_size}, {num_workers} parallel workers")

        # Generate batch list
        all_batches = []
        for batch_start in range(0, original_len, args.batch_size):
            batch_end = min(batch_start + args.batch_size, original_len)
            all_batches.append((batch_start, batch_end))

        print(f"Total batches: {len(all_batches)}")

        # Check for completed shards
        completed = get_completed_shards(args.shard_dir)
        remaining_batches = [b for b in all_batches if b not in completed]

        if completed:
            print(f"Found {len(completed)} completed shards, {len(remaining_batches)} remaining")

        if not remaining_batches:
            print("All batches already completed!")
        else:
            # Distribute batches across workers
            worker_batches = [[] for _ in range(num_workers)]
            for i, batch in enumerate(remaining_batches):
                worker_batches[i % num_workers].append(batch)

            # Assign GPUs to workers
            worker_gpus = []
            for i in range(num_workers):
                start_idx = i * tp_size
                end_idx = start_idx + tp_size
                worker_gpus.append(gpu_list[start_idx:end_idx])

            for i, (gpus, batches) in enumerate(zip(worker_gpus, worker_batches)):
                print(f"Worker {i}: GPUs {gpus}, {len(batches)} batches")

            mp.set_start_method("spawn", force=True)

            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = []
                for worker_id, (gpus, batches) in enumerate(zip(worker_gpus, worker_batches)):
                    if batches:
                        future = executor.submit(
                            worker_process,
                            worker_id,
                            gpus,
                            batches,
                            df,
                            args,
                        )
                        futures.append(future)

                for future in as_completed(futures):
                    try:
                        worker_id = future.result()
                        print(f"Worker {worker_id} completed successfully")
                    except Exception as e:
                        print(f"Worker failed with error: {e}")
                        raise

    # Build final output by loading all shards
    print("\nBuilding final output...")
    all_results = load_existing_shards(args.shard_dir)

    # Create output columns
    abstraction_json_list = []
    abstraction_list = []
    valid_list = []
    retry_count_list = []

    for idx in range(original_len):
        if idx in all_results:
            res = all_results[idx]
            abstraction_json_list.append(res.get("json_str", ""))
            abstraction_list.append(res.get("parsed", "[]"))
            valid_list.append(res.get("valid", False))
            retry_count_list.append(res.get("retry_count", 0))
        else:
            abstraction_json_list.append("")
            abstraction_list.append("[]")
            valid_list.append(False)
            retry_count_list.append(0)

    df["note_abstraction_json"] = abstraction_json_list
    df["note_abstraction"] = abstraction_list
    df["abstraction_valid"] = valid_list
    df["abstraction_retry_count"] = retry_count_list

    # Save output
    df.to_parquet(args.output_parquet, index=False)
    print(f"Wrote {args.output_parquet} with {len(df)} rows.")

    # Summary stats
    valid_count = sum(valid_list)
    print(f"Valid abstractions: {valid_count}/{original_len} ({100*valid_count/original_len:.1f}%)")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
