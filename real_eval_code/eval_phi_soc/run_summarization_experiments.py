#!/usr/bin/env python3
"""
Grid experiment runner for patient summarization — parallelized.

Runs the iterative summarization pipeline across a grid of models and chunk
sizes on a sample of real patients from the PHI SOC (standard of care) dataset.
Each experiment produces its own output directory with patient summaries and
checkpoint shards. A comparison CSV is generated at the end.

GPUs are divided among models ("model batches"). Within each batch, all
(model, chunk_size) experiments run concurrently via asyncio.gather — so
when one chunk_size experiment is in its sparse tail rounds, the server stays
busy with another experiment's dense rounds.

Usage:
    # Dry run — print the experiment grid and batching plan
    python run_summarization_experiments.py \
      --gpu_ids 0,1 --gpus_per_server 1 --download_dir /data1/ken/models --dry_run

    # Run full grid on 4 GPUs (4 models run simultaneously)
    python run_summarization_experiments.py \
      --gpu_ids 0,1,2,3 --gpus_per_server 1 --download_dir /data1/ken/models

    # Run a single model/chunk_size for testing
    python run_summarization_experiments.py \
      --gpu_ids 0 --gpus_per_server 1 --download_dir /data1/ken/models \
      --models openai/gpt-oss-20b --chunk_sizes 10000 --n_patients 2
"""


import argparse
import asyncio
import os
import sys
import time
import warnings
from datetime import date
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# Import reusable functions from the main summarization script
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]  # matchminer-ai-training
sys.path.insert(0, str(REPO_ROOT))

from importlib import import_module
_summarize = import_module("6_summarize_patients")

concatenate_and_chunk_notes = _summarize.concatenate_and_chunk_notes
deduplicate_patient_notes = _summarize.deduplicate_patient_notes
prepare_rounds = _summarize.prepare_rounds
build_prompt_text = _summarize.build_prompt_text
postprocess_output = _summarize.postprocess_output
start_vllm_server = _summarize.start_vllm_server
wait_for_server_ready = _summarize.wait_for_server_ready
shutdown_server = _summarize.shutdown_server
check_server_health = _summarize.check_server_health
single_inference_request = _summarize.single_inference_request
run_inference_batch = _summarize.run_inference_batch
load_existing_shards = _summarize.load_existing_shards
save_round_shard = _summarize.save_round_shard
_init_prompt_worker = _summarize._init_prompt_worker
_build_prompt_worker = _summarize._build_prompt_worker

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = REPO_ROOT.parent / "data/phi/soc"

# Per-model configs: reasoning marker + sampling parameters.
# Qwen instruct (non-thinking) mode for reasoning tasks.
# gpt-oss defaults.
MODEL_CONFIGS = {
    "google/gemma-4-31b-it": {
        "max_model_len": 260_000,
        "reasoning_marker": "<channel|>",
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
    },
    "Qwen/Qwen3.5-9B": {
        "max_model_len": 260_000,
        "reasoning_marker": "</think>",
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
    },
    "Qwen/Qwen3.5-35B-A3B": {
        "max_model_len": 260_000,
        "reasoning_marker": "</think>",
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
    },
    "openai/gpt-oss-20b": {
        "max_model_len": 120_000,
        "reasoning_marker": "assistantfinal",
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
    },
    "openai/gpt-oss-120b": {
        "max_model_len": 120_000,
        "reasoning_marker": "assistantfinal",
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
    },
}

# Assumed token overhead for prompt template + prior summary (excluding chunk text).
# Used to auto-compute max generation tokens: max_model_len - chunk_size - PROMPT_OVERHEAD.
PROMPT_OVERHEAD_TOKENS = 10_000

# Fallback defaults for unknown models
_QWEN_DEFAULTS = MODEL_CONFIGS["Qwen/Qwen3.5-9B"]
_GPT_DEFAULTS = MODEL_CONFIGS["openai/gpt-oss-20b"]

DEFAULT_MODELS = list(MODEL_CONFIGS.keys())
DEFAULT_CHUNK_SIZES = [2000, 10000, 25000, 50000]


def get_model_config(model: str) -> dict:
    """Get full config dict for a model, with fallback heuristics."""
    if model in MODEL_CONFIGS:
        return MODEL_CONFIGS[model]
    if "qwen" in model.lower():
        return _QWEN_DEFAULTS
    return _GPT_DEFAULTS


def get_reasoning_marker(model: str) -> str:
    """Get reasoning marker for a model."""
    return get_model_config(model)["reasoning_marker"]


def sanitize_model_name(model: str) -> str:
    """Convert model name to a filesystem-safe directory name."""
    return model.replace("/", "_")


def split_boilerplate(text: str) -> Tuple[str, str]:
    """Split summary into main summary and boilerplate text."""
    if not text:
        return "", ""
    for marker in ["Boilerplate:", "BOILERPLATE:", "boilerplate:"]:
        if marker in text:
            parts = text.split(marker, 1)
            return parts[0].strip(), parts[1].strip() if len(parts) > 1 else ""
    return text.strip(), ""


# ---------------------------------------------------------------------------
# SOC-specific: enrich summaries with treatment metadata
# ---------------------------------------------------------------------------

def enrich_experiment_summaries(patient_df: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Enrich patient summaries with dfci_mrn, trial_start_dt, and split from SOC treatment data.

    Returns the enriched DataFrame (also useful for the comparison CSV).
    """
    treatments_path = data_dir / "processed_soc_treatments.csv"
    if not treatments_path.exists():
        return patient_df

    treat_df = pd.read_csv(treatments_path)
    meta_cols = ['pseudo_mrn', 'dfci_mrn', 'trial_start_dt', 'split']
    available_cols = [c for c in meta_cols if c in treat_df.columns]
    if len(available_cols) < 2:
        return patient_df

    meta = treat_df[available_cols].drop_duplicates(subset=['pseudo_mrn'])
    patient_df = patient_df.copy()
    patient_df['pseudo_mrn'] = patient_df['pseudo_mrn'].astype(int)
    meta['pseudo_mrn'] = meta['pseudo_mrn'].astype(int)

    # Drop columns that already exist to avoid duplicates on merge
    existing = [c for c in meta.columns if c in patient_df.columns and c != 'pseudo_mrn']
    if existing:
        patient_df = patient_df.drop(columns=existing)

    patient_df = patient_df.merge(meta, on='pseudo_mrn', how='left')
    return patient_df


# ---------------------------------------------------------------------------
# Core experiment runner
# ---------------------------------------------------------------------------

async def run_single_experiment(
    df: pd.DataFrame,
    model: str,
    chunk_size: int,
    experiment_dir: Path,
    server_clients: List[Tuple[AsyncOpenAI, int]],
    prompt_pool: Pool,
    max_model_len: int,
    max_tokens: int,
    max_concurrent: int,
    batch_size: int,
    request_timeout: float,
    max_retries: int,
    reasoning_marker: str,
    temperature: float,
    top_k: int,
    top_p: float,
    presence_penalty: float,
    min_p: float,
    repetition_penalty: float,
    chunk_overlap: int,
    data_dir: Path,
):
    """Run a single summarization experiment for one (model, chunk_size) pair."""

    label = f"[{model} / chunk={chunk_size}]"
    exp_t0 = time.time()
    shard_dir = str(experiment_dir / "round_shards")
    os.makedirs(shard_dir, exist_ok=True)

    # Load tokenizer for chunking
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model, cache_dir=None, trust_remote_code=True
    )

    # Prepare rounds (chunk data)
    print(f"    {label} Chunking notes...")
    rounds, patient_chunk_order, patient_last_dates = prepare_rounds(
        df, "pseudo_mrn", "date", "text", tokenizer, chunk_size, chunk_overlap
    )
    total_chunks = sum(len(clist) for clist in patient_chunk_order.values())
    print(f"    {label} {len(patient_chunk_order)} patients, {total_chunks} chunks, {len(rounds)} rounds")

    # Load existing shards for resume
    completed_rounds, all_results = load_existing_shards(shard_dir)
    if completed_rounds > 0:
        print(f"    {label} Resuming from round {completed_rounds} ({len(all_results)} existing results)")

    # Reconstruct patient summaries from completed rounds
    patient_summaries: Dict[str, str] = {}
    if completed_rounds > 0:
        for round_idx in range(completed_rounds):
            for pid, chunk_idx, _, _, _ in rounds[round_idx]:
                if chunk_idx in all_results:
                    _, summary, _ = all_results[chunk_idx]
                    patient_summaries[pid] = summary

    if completed_rounds >= len(rounds):
        print(f"    {label} All rounds already completed.")
    else:
        # Build reverse mapping: chunk_idx -> patient_id
        chunk_to_patient: Dict[int, str] = {}
        for pid, chunk_indices in patient_chunk_order.items():
            for cidx in chunk_indices:
                chunk_to_patient[cidx] = pid

        # Process remaining rounds
        for round_idx in range(completed_rounds, len(rounds)):
            round_items = rounds[round_idx]
            print(f"    {label} Round {round_idx + 1}/{len(rounds)}: {len(round_items)} patients")

            # Build prior summaries and work items
            round_prior_summaries: Dict[int, str] = {}
            work_items = []
            for pid, chunk_idx, first_date, last_date, chunk_text in round_items:
                prior_summary = patient_summaries.get(pid, None)
                prior_text = prior_summary if prior_summary else "None - this is the first segment for this patient"
                round_prior_summaries[chunk_idx] = prior_text
                work_items.append((chunk_idx, prior_summary, first_date, last_date, chunk_text, max_model_len))

            # Build prompts in parallel (non-blocking for the event loop)
            chunksize_mp = max(1, len(work_items) // (prompt_pool._processes * 4))
            prompt_results = await asyncio.to_thread(
                lambda wi=work_items, cs=chunksize_mp: list(
                    prompt_pool.map(_build_prompt_worker, wi, chunksize=cs)
                )
            )

            # Compute per-prompt max_tokens
            prompts: List[Tuple[int, str, int]] = []
            for chunk_idx, prompt, prompt_token_count in prompt_results:
                gen_tokens = max_model_len - prompt_token_count
                if max_tokens is not None:
                    gen_tokens = min(gen_tokens, max_tokens)
                gen_tokens = max(gen_tokens, 1)
                prompts.append((chunk_idx, prompt, gen_tokens))

            # Distribute across servers round-robin
            n_servers = len(server_clients)
            server_prompt_groups: List[List[Tuple[int, str, int]]] = [[] for _ in range(n_servers)]
            for i, prompt_item in enumerate(prompts):
                server_prompt_groups[i % n_servers].append(prompt_item)

            # Launch inference on all servers concurrently
            tasks = []
            for server_idx, (client, port) in enumerate(server_clients):
                group = server_prompt_groups[server_idx]
                if group:
                    tasks.append(
                        run_inference_batch(
                            client=client,
                            prompts=group,
                            model=model,
                            temperature=temperature,
                            top_k=top_k,
                            top_p=top_p,
                            presence_penalty=presence_penalty,
                            min_p=min_p,
                            repetition_penalty=repetition_penalty,
                            reasoning_marker=reasoning_marker,
                            max_concurrent=max_concurrent,
                            batch_size=batch_size,
                            max_retries=max_retries,
                            base_timeout=request_timeout,
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
                pid = chunk_to_patient[chunk_idx]
                patient_summaries[pid] = summary

            save_round_shard(shard_dir, round_idx, round_results_with_prior)

    # Build patient-level output (last chunk's summary per patient)
    summary_generation_date = date.today().isoformat()
    patient_rows = []
    for pid, chunk_indices in patient_chunk_order.items():
        last_cidx = chunk_indices[-1]
        if last_cidx in all_results:
            _, summary, _ = all_results[last_cidx]
        else:
            summary = ""
        ps, bp = split_boilerplate(summary)
        patient_rows.append({
            "pseudo_mrn": pid,
            "patient_summary": ps,
            "patient_boilerplate_text": bp,
            "last_note_date": patient_last_dates.get(pid, ""),
            "summary_generation_date": summary_generation_date,
            "num_chunks": len(chunk_indices),
        })

    patient_df = pd.DataFrame(patient_rows)

    # Enrich with SOC treatment metadata (dfci_mrn, trial_start_dt, split)
    patient_df = enrich_experiment_summaries(patient_df, data_dir)

    # Record wall time for this experiment
    wall_time_seconds = time.time() - exp_t0
    patient_df["wall_time_seconds"] = round(wall_time_seconds, 1)

    out_path = experiment_dir / "patient_summaries.parquet"
    patient_df.to_parquet(out_path, index=False)
    print(f"    {label} Saved {out_path} ({len(patient_df)} patients, {wall_time_seconds:.1f}s wall time)")

    return patient_df


async def run_batch_experiments(
    batch_experiments: List[dict],
) -> List:
    """Run all experiments in a batch concurrently via asyncio.gather."""
    tasks = [run_single_experiment(**exp_kwargs) for exp_kwargs in batch_experiments]
    return await asyncio.gather(*tasks, return_exceptions=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Grid experiment runner for patient summarization on SOC data (parallelized)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--gpu_ids", required=True,
                    help="Comma-separated GPU IDs (e.g. '0,1,2,3')")
    ap.add_argument("--gpus_per_server", type=int, required=True,
                    help="GPUs per vLLM server (tensor_parallel_size)")
    ap.add_argument("--download_dir", required=True,
                    help="Model weight cache directory")
    ap.add_argument("--output_dir", type=str, default=None,
                    help="Output directory (default: DATA_DIR/summarization_experiments)")
    ap.add_argument("--input_parquet", type=str, default=None,
                    help="Input parquet with patient notes (default: DATA_DIR/note_level_dataset.parquet)")
    ap.add_argument("--n_patients", type=int, default=10,
                    help="Number of patients to sample (default: 10)")
    ap.add_argument("--chunk_sizes", type=str, default=",".join(str(c) for c in DEFAULT_CHUNK_SIZES),
                    help=f"Comma-separated chunk sizes (default: {','.join(str(c) for c in DEFAULT_CHUNK_SIZES)})")
    ap.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS),
                    help=f"Comma-separated model names (default: {','.join(DEFAULT_MODELS)})")
    ap.add_argument("--max_model_len", type=int, default=None,
                    help="Override max context length for vLLM (default: per-model from MODEL_CONFIGS)")
    ap.add_argument("--max_tokens", type=int, default=None,
                    help="Override max generation tokens per prompt "
                         "(default: auto = max_model_len - chunk_size - 10000)")
    ap.add_argument("--base_port", type=int, default=8000,
                    help="Base port for vLLM servers (default: 8000)")
    ap.add_argument("--max_concurrent", type=int, default=16,
                    help="Max concurrent requests per server (default: 16)")
    ap.add_argument("--batch_size", type=int, default=1000,
                    help="Prompts per batch (default: 1000)")
    ap.add_argument("--request_timeout", type=float, default=600.0,
                    help="Timeout per inference request in seconds (default: 600)")
    ap.add_argument("--max_retries", type=int, default=6,
                    help="Max retries for failed requests (default: 6)")
    ap.add_argument("--server_timeout", type=int, default=600,
                    help="Timeout waiting for vLLM server to start (default: 600)")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.90,
                    help="GPU memory utilization for vLLM (default: 0.90)")
    ap.add_argument("--max_num_seqs", type=int, default=900,
                    help="vLLM max_num_seqs (concurrent request cap; default: 900)")
    ap.add_argument("--chunk_overlap", type=int, default=500,
                    help="Token overlap between chunks (default: 500)")
    ap.add_argument("--split_filter", type=str, default="test",
                    help="Filter patients by split (e.g. 'test', 'train', 'validation'). "
                         "Set to empty string to disable. Default: 'test'.")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print experiment grid and exit")
    return ap.parse_args()


def main():
    args = parse_args()

    # Resolve defaults
    input_parquet = args.input_parquet or str(DATA_DIR / "note_level_dataset.parquet")
    output_dir = Path(args.output_dir) if args.output_dir else DATA_DIR / "summarization_experiments"

    # Parse grid parameters
    models = [m.strip() for m in args.models.split(",")]
    chunk_sizes = [int(c.strip()) for c in args.chunk_sizes.split(",")]

    # Parse GPU list
    gpu_list = [g.strip() for g in args.gpu_ids.split(",") if g.strip()]
    if len(gpu_list) % args.gpus_per_server != 0:
        usable = len(gpu_list) - (len(gpu_list) % args.gpus_per_server)
        print(f"WARNING: {len(gpu_list)} GPUs not evenly divisible by gpus_per_server={args.gpus_per_server}. "
              f"Using first {usable} GPUs.")
        gpu_list = gpu_list[:usable]

    # How many models can run concurrently?
    n_concurrent_models = len(gpu_list) // args.gpus_per_server

    # Batch models: each batch runs concurrently, batches run sequentially
    model_batches = [
        models[i:i + n_concurrent_models]
        for i in range(0, len(models), n_concurrent_models)
    ]

    # Print experiment grid
    total_experiments = len(models) * len(chunk_sizes)
    print("=" * 70)
    print("SUMMARIZATION EXPERIMENT GRID — SOC DATA (PARALLELIZED)")
    print("=" * 70)
    print(f"  Models: {models}")
    print(f"  Chunk sizes: {chunk_sizes}")
    print(f"  Total experiments: {total_experiments}")
    print(f"  Patients: {args.n_patients}")
    print(f"  Split filter: {args.split_filter or '(none)'}")
    print(f"  GPUs: {gpu_list} ({args.gpus_per_server} GPU(s) per server)")
    print(f"  Concurrent model slots: {n_concurrent_models}")
    print(f"  Model batches: {len(model_batches)}")
    print(f"  Input: {input_parquet}")
    print(f"  Output: {output_dir}")
    print()

    for model in models:
        cfg = get_model_config(model)
        model_max_len = args.max_model_len or cfg["max_model_len"]
        print(f"  {model}  (max_model_len={model_max_len:,})")
        print(f"    sampling: temp={cfg['temperature']}, top_p={cfg['top_p']}, top_k={cfg['top_k']}, "
              f"min_p={cfg['min_p']}, presence_penalty={cfg['presence_penalty']}, "
              f"repetition_penalty={cfg['repetition_penalty']}")
        print(f"    reasoning_marker: {cfg['reasoning_marker']}")
        for cs in chunk_sizes:
            exp_dir = output_dir / sanitize_model_name(model) / f"chunk_{cs:05d}"
            done = (exp_dir / "patient_summaries.parquet").exists()
            status = "[DONE]" if done else "[PENDING]"
            mt = args.max_tokens if args.max_tokens is not None else (model_max_len - cs - PROMPT_OVERHEAD_TOKENS)
            print(f"    chunk_size={cs:>6}  max_tokens={mt:>6,}  {status}")
    print()

    # Print batching plan
    print("BATCHING PLAN:")
    port_counter = args.base_port
    for batch_idx, batch_models in enumerate(model_batches):
        gpu_start_for_batch = 0  # GPUs are reused across batches
        gpu_assignments = []
        for slot_idx, model in enumerate(batch_models):
            g_start = slot_idx * args.gpus_per_server
            g_end = g_start + args.gpus_per_server
            slot_gpus = gpu_list[g_start:g_end]
            port = args.base_port + slot_idx
            gpu_assignments.append((model, slot_gpus, port))
        print(f"  Batch {batch_idx + 1}/{len(model_batches)} (GPUs {','.join(gpu_list[:len(batch_models) * args.gpus_per_server])}):")
        for model, slot_gpus, port in gpu_assignments:
            print(f"    GPU {','.join(slot_gpus)}: {model:40s} → port {port}")
        n_experiments = len(batch_models) * len(chunk_sizes)
        print(f"    Experiments: {len(batch_models)} models × {len(chunk_sizes)} chunk_sizes = {n_experiments} (all concurrent)")
    print()

    if args.dry_run:
        print("[DRY RUN] Exiting without executing.")
        return

    # Load data
    print(f"Loading {input_parquet}...")
    df = pd.read_parquet(input_parquet)

    # Apply split filter (SOC data has a patient_split column from prepare_data.py)
    if args.split_filter and "patient_split" in df.columns:
        before = df["pseudo_mrn"].nunique()
        df = df[df["patient_split"].str.contains(args.split_filter, na=False)].copy()
        after = df["pseudo_mrn"].nunique()
        print(f"  Split filter '{args.split_filter}': {before} → {after} patients")

    unique_patients = sorted(df["pseudo_mrn"].unique())[:args.n_patients]
    df = df[df["pseudo_mrn"].isin(unique_patients)].copy()
    df = df.sort_values(["pseudo_mrn", "date"]).reset_index(drop=True)
    print(f"Selected {len(unique_patients)} patients ({len(df)} notes)")

    os.makedirs(output_dir, exist_ok=True)

    # Track results for comparison CSV
    all_experiment_results: List[pd.DataFrame] = []

    # Process model batches sequentially
    for batch_idx, batch_models in enumerate(model_batches):
        print()
        print("=" * 70)
        print(f"BATCH {batch_idx + 1}/{len(model_batches)}: {batch_models}")
        print("=" * 70)

        # Check which models in this batch are entirely done (all chunk_sizes complete)
        models_to_run = []
        for model in batch_models:
            model_dir_name = sanitize_model_name(model)
            reasoning_marker = get_model_config(model)["reasoning_marker"]
            all_done = all(
                (output_dir / model_dir_name / f"chunk_{cs:05d}" / "patient_summaries.parquet").exists()
                for cs in chunk_sizes
            )
            if all_done:
                print(f"  All chunk sizes already completed for {model}. Loading results...")
                for cs in chunk_sizes:
                    exp_dir = output_dir / model_dir_name / f"chunk_{cs:05d}"
                    result_df = pd.read_parquet(exp_dir / "patient_summaries.parquet")
                    result_df["model"] = model
                    result_df["chunk_size"] = cs
                    result_df["reasoning_marker"] = reasoning_marker
                    all_experiment_results.append(result_df)
            else:
                models_to_run.append(model)

        if not models_to_run:
            print(f"  All models in batch {batch_idx + 1} already completed. Skipping.")
            continue

        # Start one vLLM server per model on assigned GPUs
        server_infos = []  # [(process, port, model), ...]
        prompt_pools = {}  # model -> Pool

        for slot_idx, model in enumerate(models_to_run):
            gpu_start = slot_idx * args.gpus_per_server
            gpu_end = gpu_start + args.gpus_per_server
            server_gpu_ids = ",".join(gpu_list[gpu_start:gpu_end])
            server_port = args.base_port + slot_idx

            model_cfg = get_model_config(model)
            model_max_len = args.max_model_len or model_cfg["max_model_len"]

            model_dir_name = sanitize_model_name(model)
            server_log_dir = output_dir / model_dir_name / "server_logs"
            os.makedirs(server_log_dir, exist_ok=True)
            log_file = str(server_log_dir / f"vllm_server_batch{batch_idx}.log")

            print(f"  Starting server for {model} on GPU {server_gpu_ids} (port {server_port}, max_model_len={model_max_len})...")
            process = start_vllm_server(
                model=model,
                download_dir=args.download_dir,
                gpu_ids=server_gpu_ids,
                tensor_parallel_size=args.gpus_per_server,
                max_model_len=model_max_len,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_num_seqs=args.max_num_seqs,
                port=server_port,
                log_file=log_file,
            )
            server_infos.append((process, server_port, model))

        # Create one prompt pool per model (each needs its own tokenizer)
        n_workers = min(os.cpu_count() or 4, 32)
        for model in models_to_run:
            prompt_pools[model] = Pool(
                processes=n_workers,
                initializer=_init_prompt_worker,
                initargs=(model, args.download_dir),
            )

        try:
            # Wait for all servers in this batch
            for process, port, model in server_infos:
                print(f"  Waiting for server {model} (port {port})...")
                if not wait_for_server_ready(port, timeout=args.server_timeout):
                    print(f"  FAILED to start server for {model} on port {port}.")
                    raise RuntimeError(f"vLLM server failed to start for {model}")

            # Create async clients (one per model)
            model_clients: Dict[str, List[Tuple[AsyncOpenAI, int]]] = {}
            for process, port, model in server_infos:
                client = AsyncOpenAI(
                    base_url=f"http://localhost:{port}/v1",
                    api_key="not-needed",
                    timeout=args.request_timeout + 60,
                )
                # Each model has one server, so it's a list with one entry
                model_clients[model] = [(client, port)]

            print(f"  All {len(server_infos)} server(s) ready.")

            # Build list of all experiments for this batch
            batch_experiments = []
            for model in models_to_run:
                model_cfg = get_model_config(model)
                model_dir_name = sanitize_model_name(model)
                reasoning_marker = model_cfg["reasoning_marker"]
                model_max_len = args.max_model_len or model_cfg["max_model_len"]

                for chunk_size in chunk_sizes:
                    exp_dir = output_dir / model_dir_name / f"chunk_{chunk_size:05d}"
                    os.makedirs(exp_dir, exist_ok=True)

                    # Check if this specific experiment is already done
                    if (exp_dir / "patient_summaries.parquet").exists():
                        print(f"  {model} / chunk_size={chunk_size}: already completed. Loading...")
                        result_df = pd.read_parquet(exp_dir / "patient_summaries.parquet")
                        result_df["model"] = model
                        result_df["chunk_size"] = chunk_size
                        result_df["reasoning_marker"] = reasoning_marker
                        all_experiment_results.append(result_df)
                        continue

                    # Auto-compute max generation tokens if not overridden:
                    # max_model_len - chunk_size - prompt/summary overhead
                    exp_max_tokens = args.max_tokens
                    if exp_max_tokens is None:
                        exp_max_tokens = model_max_len - chunk_size - PROMPT_OVERHEAD_TOKENS

                    batch_experiments.append({
                        "df": df,
                        "model": model,
                        "chunk_size": chunk_size,
                        "experiment_dir": exp_dir,
                        "server_clients": model_clients[model],
                        "prompt_pool": prompt_pools[model],
                        "max_model_len": model_max_len,
                        "max_tokens": exp_max_tokens,
                        "max_concurrent": args.max_concurrent,
                        "batch_size": args.batch_size,
                        "request_timeout": args.request_timeout,
                        "max_retries": args.max_retries,
                        "reasoning_marker": reasoning_marker,
                        "temperature": model_cfg["temperature"],
                        "top_k": model_cfg["top_k"],
                        "top_p": model_cfg["top_p"],
                        "presence_penalty": model_cfg["presence_penalty"],
                        "min_p": model_cfg["min_p"],
                        "repetition_penalty": model_cfg["repetition_penalty"],
                        "chunk_overlap": args.chunk_overlap,
                        "data_dir": DATA_DIR,
                    })

            if batch_experiments:
                print(f"\n  Launching {len(batch_experiments)} experiments concurrently...")
                t0 = time.time()
                results = asyncio.run(run_batch_experiments(batch_experiments))
                elapsed = time.time() - t0
                print(f"\n  Batch {batch_idx + 1} completed in {elapsed:.1f}s")

                # Collect results, handling any exceptions
                for exp_kwargs, result in zip(batch_experiments, results):
                    model = exp_kwargs["model"]
                    chunk_size = exp_kwargs["chunk_size"]
                    reasoning_marker = exp_kwargs["reasoning_marker"]
                    if isinstance(result, Exception):
                        print(f"  ERROR: {model} / chunk_size={chunk_size}: {result}")
                    else:
                        result["model"] = model
                        result["chunk_size"] = chunk_size
                        result["reasoning_marker"] = reasoning_marker
                        all_experiment_results.append(result)

        except Exception as e:
            print(f"  ERROR in batch {batch_idx + 1}: {e}")
        finally:
            # Shutdown prompt pools
            for model, pool in prompt_pools.items():
                pool.close()
                pool.join()
            # Shutdown servers
            print(f"  Shutting down {len(server_infos)} server(s)...")
            for process, port, model in server_infos:
                shutdown_server(process)
            print(f"  All servers for batch {batch_idx + 1} shut down.")

    # Build comparison CSV
    if all_experiment_results:
        comparison_df = pd.concat(all_experiment_results, ignore_index=True)
        comparison_path = output_dir / "experiment_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        print()
        print("=" * 70)
        print("EXPERIMENT GRID COMPLETE")
        print("=" * 70)
        print(f"  Comparison CSV: {comparison_path}")
        print(f"  Total rows: {len(comparison_df)}")
        print(f"  Experiments completed: {comparison_df.groupby(['model', 'chunk_size']).ngroups}")
        print()

        # Summary statistics
        stats = comparison_df.groupby(["model", "chunk_size"]).agg(
            n_patients=("pseudo_mrn", "nunique"),
            avg_chunks=("num_chunks", "mean"),
            avg_summary_len=("patient_summary", lambda x: x.str.len().mean()),
            wall_time_s=("wall_time_seconds", "first"),
        ).reset_index()
        print(stats.to_string(index=False))
    else:
        print("\nNo experiments completed successfully.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
