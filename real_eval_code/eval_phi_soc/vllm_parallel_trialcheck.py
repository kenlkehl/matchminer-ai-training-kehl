#!/usr/bin/env python3
"""
Parallelized vLLM inference script for patient-trial matching evaluation.

Supports:
- Multi-GPU parallelization with configurable tensor parallelism
- Resumable execution via output shards
- Configurable sampling parameters

python vllm_parallel_trialcheck.py \
     --model ksg-dfci/OncoReasoning-3B-1225 \
     --input-file ../top_cohorts_tocheck_only_open_phi.csv \
     --output-file ./final_patient_centric_soc_reasonable_check_results.csv \
     --temp-dir ./output_shards \
     --gpus 0,1,2,3 \
     --tensor-parallel-size 1 \
     --temperature 1.0 \
     --top-p 0.9 \
     --n-samples 100 \
"""

import argparse
import os
import sys
import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import pandas as pd
import numpy as np


def wrap_qwen35_text_config_for_vllm(hf_config):
    """Wrap Qwen3.5 text-only configs in the top-level config vLLM expects."""
    if getattr(hf_config, "model_type", None) != "qwen3_5_text":
        return hf_config

    from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5Config

    return Qwen3_5Config(
        text_config=hf_config.to_dict(),
        architectures=getattr(hf_config, "architectures", None),
        tie_word_embeddings=getattr(hf_config, "tie_word_embeddings", False),
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run parallelized vLLM inference for patient-trial matching"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the model or HuggingFace model ID",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to input CSV file with patient_summary and this_space columns",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to final merged output CSV file",
    )
    parser.add_argument(
        "--temp-dir",
        type=str,
        required=True,
        help="Directory for temporary output shards (enables resumability)",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        required=True,
        help="Comma-separated list of GPU IDs to use (e.g., '0,1,2,3')",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs per vLLM instance (tensor parallelism)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.01,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling parameter",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=50,
        help="Number of samples to generate per prompt",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of prompts per batch",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=7500,
        help="Maximum tokens to generate per response",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=30000,
        help="Maximum model context length",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=900,
        help="vLLM max_num_seqs (concurrent request cap).",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.20,
        help="GPU memory utilization fraction for vLLM",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="Repetition penalty for generation",
    )
    parser.add_argument(
        "--filter-val",
        action="store_true",
        help="Filter to only validation split rows",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Maximum number of rows to process (for testing)",
    )

    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from vllm_reasoning_utils import add_reasoning_cli_args
    add_reasoning_cli_args(parser)

    return parser.parse_args()


def get_shard_filename(temp_dir: str, batch_start: int, batch_end: int) -> str:
    """Generate consistent shard filename."""
    return os.path.join(temp_dir, f"shard_{batch_start:08d}_{batch_end:08d}.csv")


def get_completed_shards(temp_dir: str) -> set:
    """Get set of (batch_start, batch_end) tuples for completed shards."""
    completed = set()
    if not os.path.exists(temp_dir):
        return completed
    
    pattern = re.compile(r"shard_(\d+)_(\d+)\.csv")
    for filename in os.listdir(temp_dir):
        match = pattern.match(filename)
        if match:
            batch_start = int(match.group(1))
            batch_end = int(match.group(2))
            # Verify file is not empty/corrupted
            filepath = os.path.join(temp_dir, filename)
            try:
                df = pd.read_csv(filepath, nrows=1)
                if len(df) > 0:
                    completed.add((batch_start, batch_end))
            except Exception:
                pass  # File is corrupted, will be reprocessed
    
    return completed


def worker_process(
    worker_id: int,
    gpu_ids: list,
    batches: list,
    df: pd.DataFrame,
    args: argparse.Namespace,
):
    """
    Worker process that runs vLLM inference on assigned batches.
    
    Args:
        worker_id: Unique worker identifier
        gpu_ids: List of GPU IDs assigned to this worker
        batches: List of (batch_start, batch_end) tuples to process
        df: The full dataframe
        args: Command line arguments
    """
    # Set CUDA_VISIBLE_DEVICES for this worker
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    
    # Import vLLM after setting CUDA_VISIBLE_DEVICES
    from vllm import LLM, SamplingParams
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from vllm_reasoning_utils import resolve_parser_name, parse_reasoning_output
    reasoning_parser = resolve_parser_name(args.model, args.reasoning_parser)
    
    print(f"[Worker {worker_id}] Starting on GPUs {gpu_ids}, processing {len(batches)} batches")
    
    # Initialize vLLM model
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        hf_overrides=wrap_qwen35_text_config_for_vllm,
        language_model_only=True,
    )
    
    tokenizer = llm.get_tokenizer()
    
    sampling_params = SamplingParams(
        n=args.n_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
        skip_special_tokens=False,
    )
    
    for batch_start, batch_end in batches:
        shard_path = get_shard_filename(args.temp_dir, batch_start, batch_end)
        
        # Double-check shard doesn't exist (race condition protection)
        if os.path.exists(shard_path):
            try:
                test_df = pd.read_csv(shard_path, nrows=1)
                if len(test_df) > 0:
                    print(f"[Worker {worker_id}] Skipping batch {batch_start}-{batch_end} (already exists)")
                    continue
            except Exception:
                pass
        
        print(f"[Worker {worker_id}] Processing batch {batch_start}-{batch_end}")
        
        batch_df = df.iloc[batch_start:batch_end].copy()
        
        # Build prompts
        prompts = []
        for _, row in batch_df.iterrows():
            patient_summary = row["patient_summary"]
            trial_summary = str(row["this_space"])
            
            messages = [
                {"role": "system", "content": "Reasoning: high"},
                {"role": "user", "content": (
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
                )}
            ]
            
            prompt = tokenizer.apply_chat_template(
                conversation=messages,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=True,
            )
            prompts.append(prompt)
        
        # Run inference
        responses = llm.generate(prompts, sampling_params)

        # Process results
        results = []
        batch_df_reset = batch_df.reset_index(drop=True)

        for prompt_id, request_output in enumerate(responses):
            original_idx = batch_start + prompt_id
            row_data = batch_df_reset.iloc[prompt_id].to_dict()

            for completion_output in request_output.outputs:
                reasoning_text, response_text = parse_reasoning_output(
                    completion_output.text, reasoning_parser, tokenizer
                )
                response_id = completion_output.index

                SCORE_PATTERN = re.compile(r"[Ff]inal\s+[Ss]core\s*:\s*(\d)")
                tail = response_text[-60:].replace("*", "").replace("\u202f", " ")
                m = SCORE_PATTERN.search(tail)
                if m:
                    score = min(int(m.group(1)), 5)
                    eligibility_result = score
                    eligibility_verdict = f"Score:{score}"
                else:
                    tail_upper = tail.upper()
                    fallback_m = re.search(r"SCORE\s*[:\-=]\s*(\d)", tail_upper)
                    if fallback_m:
                        score = min(int(fallback_m.group(1)), 5)
                        eligibility_result = score
                        eligibility_verdict = f"Score:{score}"
                    elif "NOT REASONABLE" in tail_upper or "NOT A REASONABLE" in tail_upper:
                        eligibility_result = 0
                        eligibility_verdict = "Score:0"
                    else:
                        eligibility_result = -1
                        eligibility_verdict = "PARSE_FAILED"

                result = {
                    "prompt_id": original_idx,
                    "response_id": response_id,
                    "llama_reasoning": reasoning_text,
                    "llama_response": response_text,
                    "eligibility_result": eligibility_result,
                    "eligibility_verdict": eligibility_verdict,
                }
                result.update(row_data)
                results.append(result)
        
        # Save shard
        results_df = pd.DataFrame(results)
        results_df.to_csv(shard_path, index=False)
        print(f"[Worker {worker_id}] Saved batch {batch_start}-{batch_end} ({len(results_df)} rows)")
    
    print(f"[Worker {worker_id}] Finished all batches")
    return worker_id


def merge_shards(temp_dir: str, output_file: str):
    """Merge all shards into final output file."""
    shard_files = sorted(Path(temp_dir).glob("shard_*.csv"))
    
    if not shard_files:
        print("No shards found to merge!")
        return
    
    print(f"Merging {len(shard_files)} shards into {output_file}")
    
    dfs = []
    for shard_file in shard_files:
        try:
            df = pd.read_csv(shard_file)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not read {shard_file}: {e}")
    
    if dfs:
        merged_df = pd.concat(dfs, ignore_index=True)
        merged_df.to_csv(output_file, index=False)
        print(f"Merged output saved to {output_file} ({len(merged_df)} total rows)")
    else:
        print("No valid shards to merge!")


def main():
    args = parse_args()
    
    # Parse GPU list
    gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]
    num_gpus = len(gpu_ids)
    tp_size = args.tensor_parallel_size
    
    if num_gpus % tp_size != 0:
        print(f"Error: Number of GPUs ({num_gpus}) must be divisible by tensor_parallel_size ({tp_size})")
        sys.exit(1)
    
    num_workers = num_gpus // tp_size
    print(f"Configuration: {num_gpus} GPUs, tensor_parallel_size={tp_size}, {num_workers} parallel workers")
    
    # Create temp directory
    os.makedirs(args.temp_dir, exist_ok=True)
    
    # Load input data
    print(f"Loading input file: {args.input_file}")
    df = pd.read_csv(args.input_file)
    print(f"Loaded {len(df)} rows")
    
    # Apply filters
    if args.filter_val:
        if "split" in df.columns:
            df = df[df["split"].str.contains("val", na=False)]
            print(f"After filtering to validation split: {len(df)} rows")
    
    df = df[~df["patient_summary"].isnull()]
    print(f"After removing null patient_summary: {len(df)} rows")
    
    if args.max_rows is not None:
        df = df.head(args.max_rows)
        print(f"Limited to {len(df)} rows")
    
    # Reset index for consistent indexing
    df = df.reset_index(drop=True)
    
    # Generate batch list
    all_batches = []
    for batch_start in range(0, len(df), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(df))
        all_batches.append((batch_start, batch_end))
    
    print(f"Total batches: {len(all_batches)}")
    
    # Check for completed shards
    completed = get_completed_shards(args.temp_dir)
    remaining_batches = [b for b in all_batches if b not in completed]
    
    if completed:
        print(f"Found {len(completed)} completed shards, {len(remaining_batches)} remaining")
    
    if not remaining_batches:
        print("All batches already completed!")
        merge_shards(args.temp_dir, args.output_file)
        return
    
    # Distribute batches across workers
    worker_batches = [[] for _ in range(num_workers)]
    for i, batch in enumerate(remaining_batches):
        worker_batches[i % num_workers].append(batch)
    
    # Assign GPUs to workers
    worker_gpus = []
    for i in range(num_workers):
        start_idx = i * tp_size
        end_idx = start_idx + tp_size
        worker_gpus.append(gpu_ids[start_idx:end_idx])
    
    # Print worker assignments
    for i, (gpus, batches) in enumerate(zip(worker_gpus, worker_batches)):
        print(f"Worker {i}: GPUs {gpus}, {len(batches)} batches")
    
    # Use spawn method for CUDA compatibility
    mp.set_start_method("spawn", force=True)
    
    # Run workers in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for worker_id, (gpus, batches) in enumerate(zip(worker_gpus, worker_batches)):
            if batches:  # Only start workers with work to do
                future = executor.submit(
                    worker_process,
                    worker_id,
                    gpus,
                    batches,
                    df,
                    args,
                )
                futures.append(future)
        
        # Wait for all workers to complete
        for future in as_completed(futures):
            try:
                worker_id = future.result()
                print(f"Worker {worker_id} completed successfully")
            except Exception as e:
                print(f"Worker failed with error: {e}")
                raise
    
    # Merge all shards
    merge_shards(args.temp_dir, args.output_file)
    print("Done!")


if __name__ == "__main__":
    main()
