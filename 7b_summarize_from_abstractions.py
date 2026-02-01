#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Patient-level summary generation from note-level abstractions.

Aggregates note-level JSON abstractions (from 7a_abstract_notes.py) into
patient-level summaries. Performs deduplication of repeated concepts while
preserving date ranges.

Supports two inference modes:
- Python API mode (default): Uses vLLM LLM class with ProcessPoolExecutor
- Server mode (--use_server): Uses AsyncOpenAI client with vLLM server

Examples
--------
# Python API mode (default)
python 7b_summarize_from_abstractions.py \
    --input_parquet ../data/no_phi/note_abstractions.parquet \
    --output_parquet ../data/no_phi/patient_summaries_all.parquet \
    --patient_summaries_parquet ../data/no_phi/patient_summaries.parquet \
    --shard_dir ../data/no_phi/summary_shards \
    --model openai/gpt-oss-120b \
    --download_dir /data1/ken/models \
    --gpu_ids 0,1,2,3 \
    --max_model_len 10000

# Server mode
python 7b_summarize_from_abstractions.py \
    --input_parquet ../data/no_phi/note_abstractions.parquet \
    --output_parquet ../data/no_phi/patient_summaries_all.parquet \
    --patient_summaries_parquet ../data/no_phi/patient_summaries.parquet \
    --shard_dir ../data/no_phi/summary_shards \
    --model openai/gpt-oss-120b \
    --download_dir /data1/ken/models \
    --gpu_ids 0,1,2,3 \
    --use_server \
    --port 8000
"""

import argparse
import asyncio
import glob
import json
import os
import re
import subprocess
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple
import multiprocessing as mp

import pandas as pd
import requests


# -------------------------
# Prompts
# -------------------------

PATIENT_SUMMARY_PROMPT = """You are an experienced clinical oncology history summarization bot.

Given the following structured clinical data extracted from a patient's medical records,
synthesize a comprehensive patient summary.

The data is a list of clinical concepts with dates. Concepts may have:
- first_noted/last_noted: date range when this information was documented

EXTRACTED CLINICAL DATA:
{json_data}

Write a summary following this format:
Age: [most recent age]
Sex: [sex]
Cancer type: [cancer type(s)]
Histology: [histology]
Current extent: [most recent disease burden]
Biomarkers: [all known biomarkers]
Treatment history:
# [date range]: [treatment] - [response if known]
Boilerplate:
[Any exclusion criteria concerns, or "No evidence of common boilerplate exclusion criteria"]

Use dates from the data. Format as plain text only. Do not use markdown or Unicode symbols."""


# -------------------------
# Deduplication
# -------------------------

def deduplicate_concepts(all_concepts: List[Dict]) -> List[Dict]:
    """
    Remove duplicate concepts, keeping track of date ranges.

    Two concepts are considered duplicates if they are identical in all fields
    EXCEPT note_date. When duplicates are found, we keep one copy with
    first_noted and last_noted fields showing the date range.
    """
    seen = {}
    for concept in all_concepts:
        # Create key from all fields except note_date
        key_dict = {k: v for k, v in concept.items() if k != 'note_date'}
        key = json.dumps(key_dict, sort_keys=True)

        if key not in seen:
            seen[key] = concept.copy()
            note_date = concept.get('note_date', 'unknown')
            seen[key]['first_noted'] = note_date
            seen[key]['last_noted'] = note_date
            if 'note_date' in seen[key]:
                del seen[key]['note_date']  # Replace with first/last_noted
        else:
            # Update date range
            note_date = concept.get('note_date', 'unknown')
            if note_date != 'unknown':
                current_first = seen[key].get('first_noted', 'unknown')
                current_last = seen[key].get('last_noted', 'unknown')

                if current_first == 'unknown' or (note_date < current_first):
                    seen[key]['first_noted'] = note_date
                if current_last == 'unknown' or (note_date > current_last):
                    seen[key]['last_noted'] = note_date

    return list(seen.values())


def aggregate_patient_concepts(
    patient_df: pd.DataFrame,
    abstraction_col: str = "note_abstraction"
) -> Tuple[List[Dict], str]:
    """
    Aggregate all note abstractions for a patient and deduplicate.

    Returns:
        Tuple of (deduplicated_concepts, json_string)
    """
    all_concepts = []

    for _, row in patient_df.iterrows():
        abstraction_str = row.get(abstraction_col, "[]")
        if pd.isna(abstraction_str) or not abstraction_str:
            continue

        try:
            concepts = json.loads(abstraction_str)
            if isinstance(concepts, list):
                all_concepts.extend(concepts)
        except json.JSONDecodeError:
            continue

    if not all_concepts:
        return [], "[]"

    deduplicated = deduplicate_concepts(all_concepts)
    return deduplicated, json.dumps(deduplicated, indent=2)


# -------------------------
# Utilities
# -------------------------

def build_summary_prompt(
    tokenizer,
    concepts_json: str,
    max_model_len: int,
    margin_tokens: int = 5000
) -> str:
    """Build prompt for patient summary generation."""
    threshold = max(1024, max_model_len - margin_tokens)

    # Truncate if too long
    toks = tokenizer(concepts_json, add_special_tokens=False).input_ids
    if len(toks) > threshold:
        # Truncate the JSON (best effort - may break JSON structure)
        half = threshold // 2
        first_part = toks[:half]
        last_part = toks[-half:]
        concepts_json = tokenizer.decode(first_part) + " ... " + tokenizer.decode(last_part)

    user_content = PATIENT_SUMMARY_PROMPT.format(json_data=concepts_json)

    messages = [
        {'role': 'system', 'content': 'You are an experienced clinical oncology summarization assistant.'},
        {'role': 'user', 'content': user_content}
    ]

    prompt = tokenizer.apply_chat_template(
        conversation=messages,
        add_generation_prompt=True,
        tokenize=False
    )
    return prompt


def postprocess_output(raw_text: str) -> str:
    """Extract summary from output, handling reasoning model output."""
    reasoning_marker = "assistantfinal"

    if reasoning_marker in raw_text:
        parts = raw_text.split(reasoning_marker, 1)
        return parts[1].strip()
    else:
        return raw_text.strip()


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


# -------------------------
# Sharding / Resume
# -------------------------

def get_shard_filename(temp_dir: str, patient_id: str) -> str:
    """Generate consistent shard filename for a patient."""
    # Sanitize patient_id for filesystem
    safe_id = re.sub(r'[^\w\-_]', '_', str(patient_id))
    return os.path.join(temp_dir, f"patient_{safe_id}.parquet")


def get_completed_patients(temp_dir: str) -> set:
    """Get set of completed patient IDs."""
    completed = set()
    if not os.path.exists(temp_dir):
        return completed

    for filename in os.listdir(temp_dir):
        if filename.startswith("patient_") and filename.endswith(".parquet"):
            try:
                filepath = os.path.join(temp_dir, filename)
                df = pd.read_parquet(filepath)
                if len(df) > 0 and "patient_id" in df.columns:
                    patient_id = df["patient_id"].iloc[0]
                    completed.add(str(patient_id))
            except Exception:
                pass

    return completed


def load_existing_shards(shard_dir: str) -> Dict[str, Dict[str, Any]]:
    """Load existing shard files for resume."""
    results: Dict[str, Dict[str, Any]] = {}

    if not os.path.exists(shard_dir):
        return results

    shard_files = glob.glob(os.path.join(shard_dir, "patient_*.parquet"))
    for shard_file in shard_files:
        try:
            shard_df = pd.read_parquet(shard_file)
            for _, row in shard_df.iterrows():
                patient_id = str(row["patient_id"]) if "patient_id" in row else None
                if patient_id:
                    results[patient_id] = {
                        "patient_summary": str(row.get("patient_summary", "")) if pd.notna(row.get("patient_summary")) else "",
                        "patient_boilerplate_text": str(row.get("patient_boilerplate_text", "")) if pd.notna(row.get("patient_boilerplate_text")) else "",
                        "aggregated_concepts_json": str(row.get("aggregated_concepts_json", "[]")) if pd.notna(row.get("aggregated_concepts_json")) else "[]",
                    }
        except Exception as e:
            print(f"Warning: Could not load shard {shard_file}: {e}")

    return results


def save_patient_shard(shard_dir: str, patient_id: str, result: Dict):
    """Save result for a single patient."""
    os.makedirs(shard_dir, exist_ok=True)
    shard_path = get_shard_filename(shard_dir, patient_id)

    df = pd.DataFrame([{
        "patient_id": patient_id,
        **result
    }])
    df.to_parquet(shard_path, index=False)


def save_batch_shard(shard_dir: str, batch_idx: int, results: List[Dict]):
    """Save results for a batch of patients."""
    os.makedirs(shard_dir, exist_ok=True)
    shard_path = os.path.join(shard_dir, f"batch_{batch_idx:06d}.parquet")

    df = pd.DataFrame(results)
    df.to_parquet(shard_path, index=False)
    print(f"Saved shard: {shard_path} ({len(results)} patients)")


# -------------------------
# vLLM Server Management
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
    patient_id: str,
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    max_retries: int = 3,
    base_timeout: float = 300.0,
) -> Tuple[str, str]:
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
            return (patient_id, postprocess_output(raw_text))

        except asyncio.TimeoutError:
            wait_time = (2 ** attempt) * 5
            if attempt < max_retries - 1:
                print(f"  Patient {patient_id}: timeout (attempt {attempt + 1}/{max_retries}), retrying...")
                await asyncio.sleep(wait_time)
            else:
                return (patient_id, "")

        except Exception as e:
            wait_time = (2 ** attempt) * 2
            if attempt < max_retries - 1:
                print(f"  Patient {patient_id}: error '{e}' (attempt {attempt + 1}/{max_retries}), retrying...")
                await asyncio.sleep(wait_time)
            else:
                return (patient_id, "")

    return (patient_id, "")


async def run_inference_batch_async(
    client,
    prompts: List[Tuple[str, str]],  # (patient_id, prompt)
    model: str,
    temperature: float,
    max_tokens: int,
    max_concurrent: int = 16,
) -> List[Tuple[str, str]]:
    """Send batch of requests concurrently."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded_request(patient_id: str, prompt: str) -> Tuple[str, str]:
        async with semaphore:
            return await single_inference_request(
                client=client,
                patient_id=patient_id,
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    tasks = [bounded_request(pid, prompt) for pid, prompt in prompts]
    results = await asyncio.gather(*tasks)
    return list(results)


# -------------------------
# Python API Mode (ProcessPoolExecutor)
# -------------------------

def worker_process(
    worker_id: int,
    gpu_ids: list,
    patient_data: List[Tuple[str, str, str]],  # (patient_id, concepts_json, prompt)
    args: argparse.Namespace,
):
    """
    Worker process that runs vLLM inference on assigned patients.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

    from vllm import LLM, SamplingParams

    print(f"[Worker {worker_id}] Starting on GPUs {gpu_ids}, processing {len(patient_data)} patients")

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        download_dir=args.download_dir,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    # Process in batches
    results = []
    for batch_start in range(0, len(patient_data), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(patient_data))
        batch = patient_data[batch_start:batch_end]

        print(f"[Worker {worker_id}] Processing batch {batch_start}-{batch_end}")

        prompts = [prompt for _, _, prompt in batch]
        responses = llm.generate(prompts, sampling_params)

        for i, (patient_id, concepts_json, _) in enumerate(batch):
            raw_text = responses[i].outputs[0].text
            summary_text = postprocess_output(raw_text)
            patient_summary, boilerplate = split_boilerplate(summary_text)

            result = {
                "patient_id": patient_id,
                "patient_summary": patient_summary,
                "patient_boilerplate_text": boilerplate,
                "aggregated_concepts_json": concepts_json,
            }
            results.append(result)

            # Save individual shard
            save_patient_shard(args.shard_dir, patient_id, result)

    print(f"[Worker {worker_id}] Finished all patients")
    return results


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser("Patient-level summary from note abstractions.")
    ap.add_argument("--input_parquet", required=True,
                    help="Input parquet (output from 7a_abstract_notes.py)")
    ap.add_argument("--output_parquet", required=True,
                    help="Output parquet with patient summaries (all rows)")
    ap.add_argument("--patient_summaries_parquet", default=None,
                    help="One-row-per-patient output parquet")
    ap.add_argument("--shard_dir", required=True, help="Checkpoint directory")
    ap.add_argument("--patient_id_col", default="pseudo_mrn", help="Patient ID column")
    ap.add_argument("--date_col", default="date", help="Date column for ordering")
    ap.add_argument("--model", default="openai/gpt-oss-120b", help="Model path")
    ap.add_argument("--download_dir", required=True, help="Model download directory")
    ap.add_argument("--gpu_ids", required=True, help="Comma-separated GPU IDs")
    ap.add_argument("--tensor_parallel_size", type=int, default=1,
                    help="Tensor parallel size per vLLM instance")
    ap.add_argument("--max_model_len", type=int, default=10000, help="Max model context length")
    ap.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    ap.add_argument("--max_tokens", type=int, default=4000, help="Max tokens to generate")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.93,
                    help="GPU memory utilization fraction")
    ap.add_argument("--batch_size", type=int, default=50,
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

    # Check for note_abstraction column
    if "note_abstraction" not in df.columns:
        raise ValueError("Column 'note_abstraction' not found. Run 7a_abstract_notes.py first.")

    # Filter to max_patients if specified
    if args.max_patients is not None:
        unique_patients = df[args.patient_id_col].unique()[:args.max_patients]
        df = df[df[args.patient_id_col].isin(unique_patients)].copy()
        print(f"Limited to {args.max_patients} patients")

    # Sort by patient and date
    if args.date_col in df.columns:
        df = df.sort_values([args.patient_id_col, args.date_col]).reset_index(drop=True)
    else:
        df = df.sort_values([args.patient_id_col]).reset_index(drop=True)

    # Get unique patients
    unique_patients = df[args.patient_id_col].unique()
    print(f"Total patients: {len(unique_patients)}")

    # Create shard directory
    os.makedirs(args.shard_dir, exist_ok=True)

    # Parse GPU IDs
    gpu_ids_normalized = args.gpu_ids.replace(";", ",")
    gpu_list = [int(g.strip()) for g in gpu_ids_normalized.split(",") if g.strip()]

    # Load existing results
    existing_results = load_existing_shards(args.shard_dir)
    if existing_results:
        print(f"Loaded {len(existing_results)} existing patient results from shards")

    # Find patients that need processing
    patients_to_process = [p for p in unique_patients if str(p) not in existing_results]

    if not patients_to_process:
        print("All patients already processed. Building final output...")
    else:
        print(f"Processing {len(patients_to_process)} remaining patients")

        # Prepare patient data: aggregate concepts for each patient
        print("Aggregating concepts for each patient...")
        patient_data = []  # (patient_id, concepts_json, prompt)

        # Load tokenizer
        from transformers import AutoTokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            cache_dir=args.download_dir,
            trust_remote_code=True
        )

        for patient_id in patients_to_process:
            patient_df = df[df[args.patient_id_col] == patient_id]
            concepts, concepts_json = aggregate_patient_concepts(patient_df)

            if not concepts:
                # No concepts - store empty result
                existing_results[str(patient_id)] = {
                    "patient_summary": "No clinical information extracted from notes.",
                    "patient_boilerplate_text": "",
                    "aggregated_concepts_json": "[]",
                }
                save_patient_shard(args.shard_dir, str(patient_id), existing_results[str(patient_id)])
                continue

            prompt = build_summary_prompt(tokenizer, concepts_json, args.max_model_len)
            patient_data.append((str(patient_id), concepts_json, prompt))

        print(f"Prepared {len(patient_data)} patients with concepts for summarization")

        if patient_data:
            if args.use_server:
                # Server mode
                tensor_parallel_size = len(gpu_list)

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

                    # Process in batches
                    for batch_start in range(0, len(patient_data), args.batch_size):
                        batch_end = min(batch_start + args.batch_size, len(patient_data))
                        batch = patient_data[batch_start:batch_end]

                        print(f"Processing batch {batch_start}-{batch_end} of {len(patient_data)}")

                        prompts = [(pid, prompt) for pid, _, prompt in batch]
                        results = asyncio.run(run_inference_batch_async(
                            client=client,
                            prompts=prompts,
                            model=args.model,
                            temperature=args.temperature,
                            max_tokens=args.max_tokens,
                            max_concurrent=args.max_concurrent_requests,
                        ))

                        results_map = {pid: text for pid, text in results}

                        for patient_id, concepts_json, _ in batch:
                            summary_text = results_map.get(patient_id, "")
                            patient_summary, boilerplate = split_boilerplate(summary_text)

                            result = {
                                "patient_summary": patient_summary,
                                "patient_boilerplate_text": boilerplate,
                                "aggregated_concepts_json": concepts_json,
                            }
                            existing_results[patient_id] = result
                            save_patient_shard(args.shard_dir, patient_id, result)

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

                # Distribute patients across workers
                worker_data = [[] for _ in range(num_workers)]
                for i, data in enumerate(patient_data):
                    worker_data[i % num_workers].append(data)

                # Assign GPUs to workers
                worker_gpus = []
                for i in range(num_workers):
                    start_idx = i * tp_size
                    end_idx = start_idx + tp_size
                    worker_gpus.append(gpu_list[start_idx:end_idx])

                for i, (gpus, data) in enumerate(zip(worker_gpus, worker_data)):
                    print(f"Worker {i}: GPUs {gpus}, {len(data)} patients")

                mp.set_start_method("spawn", force=True)

                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    futures = []
                    for worker_id, (gpus, data) in enumerate(zip(worker_gpus, worker_data)):
                        if data:
                            future = executor.submit(
                                worker_process,
                                worker_id,
                                gpus,
                                data,
                                args,
                            )
                            futures.append(future)

                    for future in as_completed(futures):
                        try:
                            results = future.result()
                            for result in results:
                                patient_id = result["patient_id"]
                                existing_results[patient_id] = {
                                    "patient_summary": result["patient_summary"],
                                    "patient_boilerplate_text": result["patient_boilerplate_text"],
                                    "aggregated_concepts_json": result["aggregated_concepts_json"],
                                }
                            print(f"Worker completed: {len(results)} patients")
                        except Exception as e:
                            print(f"Worker failed with error: {e}")
                            raise

    # Build final output
    print("\nBuilding final output...")

    # For each patient, get the last row and add summary columns
    patient_summary_list = []
    patient_boilerplate_list = []
    aggregated_concepts_list = []

    for _, row in df.iterrows():
        patient_id = str(row[args.patient_id_col])
        if patient_id in existing_results:
            res = existing_results[patient_id]
            patient_summary_list.append(res.get("patient_summary", ""))
            patient_boilerplate_list.append(res.get("patient_boilerplate_text", ""))
            aggregated_concepts_list.append(res.get("aggregated_concepts_json", "[]"))
        else:
            patient_summary_list.append("")
            patient_boilerplate_list.append("")
            aggregated_concepts_list.append("[]")

    df["patient_summary"] = patient_summary_list
    df["patient_boilerplate_text"] = patient_boilerplate_list
    df["aggregated_concepts_json"] = aggregated_concepts_list

    # Add summary generation date
    from datetime import date
    df["summary_generation_date"] = date.today().isoformat()

    # Save full output
    df.to_parquet(args.output_parquet, index=False)
    print(f"Wrote {args.output_parquet} with {len(df)} rows.")

    # Create one-row-per-patient output
    if args.patient_summaries_parquet:
        # Get last row per patient (already sorted by patient_id and date)
        patient_summaries_df = df.groupby(args.patient_id_col).last().reset_index()
        if args.date_col in patient_summaries_df.columns:
            patient_summaries_df = patient_summaries_df.rename(columns={args.date_col: "last_note_date"})
        patient_summaries_df.to_parquet(args.patient_summaries_parquet, index=False)
        print(f"Wrote {args.patient_summaries_parquet} with {len(patient_summaries_df)} patients.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
