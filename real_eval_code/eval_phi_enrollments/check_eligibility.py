#!/usr/bin/env python3
"""
LLM-based eligibility checking for patient-trial matches.

This script uses an LLM to determine if trial spaces are reasonable considerations
for patients (or vice versa).

Two modes:
- "patient_centric": Check if trial spaces are reasonable for patients
- "trial_centric": Check if patients are reasonable for trial spaces

Usage:
    python check_eligibility.py --gpu 0 --mode patient_centric
    python check_eligibility.py --gpu 0 --mode trial_centric
    python check_eligibility.py --gpu 0 --mode patient_centric --input ./custom_candidates.csv
"""

import argparse
import os
import re
import sys
import glob
import pandas as pd
from pathlib import Path

# Repo root for default paths
REPO_ROOT = Path(__file__).resolve().parents[2]  # matchminer-ai-training


def parse_args():
    parser = argparse.ArgumentParser(description="LLM-based eligibility checking")
    parser.add_argument("--gpu", type=str, required=True,
                        help="GPU ID to use (e.g., '0')")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["patient_centric", "trial_centric"],
                        help="Check mode: 'patient_centric' or 'trial_centric'")
    parser.add_argument("--input", type=str, default=None,
                        help="Input CSV file with candidates (default: derived from mode)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for shard files (default: derived from mode)")
    parser.add_argument("--output-file", type=str, default=None,
                        help="Final merged output CSV file (default: derived from mode)")
    parser.add_argument("--batch-size", type=int, default=2000,
                        help="Batch size for LLM inference")
    parser.add_argument("--model", type=str, default='openai/gpt-oss-120b',
                        help="Model path (default: openai/gpt-oss-120b)")
    parser.add_argument("--download-dir", type=str,
                        default="/data1/ken/meta/2024/meta_ai",
                        help="Download directory for model weights (used if model not found locally)")
    parser.add_argument("--gpu-mem-util", type=float, default=0.90,
                        help="GPU memory utilization")
    parser.add_argument("--max-model-len", type=int, default=10000,
                        help="Maximum model context length")
    parser.add_argument("--max-num-seqs", type=int, default=900,
                        help="vLLM max_num_seqs (concurrent request cap).")
    return parser.parse_args()


def ask_about_trials_loosely(patient_summaries, trial_summaries, llama_model):
    """Ask LLM whether trials are reasonable considerations for patients."""
    from vllm import SamplingParams

    tokenizer = llama_model.get_tokenizer()
    prompts = []

    for patient_summary, trial_summary in zip(patient_summaries, trial_summaries):
        messages = [
            {'role': 'system', 'content': "Reasoning: high"},
            {'role': 'user', 'content': (
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
            conversation=messages, add_generation_prompt=True, tokenize=False, enable_thinking=True
        )
        prompts.append(prompt)

    responses = llama_model.generate(
        prompts,
        SamplingParams(
            temperature=0.0,
            top_k=1,
            max_tokens=5000,
            repetition_penalty=1.0,
            skip_special_tokens=False,
        )
    )

    from vllm.reasoning.gemma4_utils import parse_thinking_output

    parsed = [parse_thinking_output(x.outputs[0].text) for x in responses]
    response_reasonings = [(p["thinking"] or "") for p in parsed]
    response_texts = [(p["answer"] or "") for p in parsed]

    SCORE_PATTERN = re.compile(r"[Ff]inal\s+[Ss]core\s*:\s*(\d)")

    eligibility_results = []
    eligibility_verdicts = []
    for txt in response_texts:
        tail = txt[-60:].replace("*", "").replace("\u202f", " ")
        m = SCORE_PATTERN.search(tail)
        if m:
            score = min(int(m.group(1)), 5)
            eligibility_results.append(score)
            eligibility_verdicts.append(f"Score:{score}")
        else:
            tail_upper = tail.upper()
            fallback_m = re.search(r"SCORE\s*[:\-=]\s*(\d)", tail_upper)
            if fallback_m:
                score = min(int(fallback_m.group(1)), 5)
                eligibility_results.append(score)
                eligibility_verdicts.append(f"Score:{score}")
            elif "NOT REASONABLE" in tail_upper or "NOT A REASONABLE" in tail_upper:
                eligibility_results.append(0)
                eligibility_verdicts.append("Score:0")
            else:
                eligibility_results.append(-1)
                eligibility_verdicts.append("PARSE_FAILED")

    return responses, response_reasonings, response_texts, eligibility_results, eligibility_verdicts


def get_completed_batches(output_dir):
    """Get the highest completed batch index."""
    pattern = os.path.join(output_dir, "*_through_*.csv")
    files = glob.glob(pattern)
    max_idx = -1
    for f in files:
        try:
            # Extract the "through_X" number
            basename = os.path.basename(f)
            idx_str = basename.split("_through_")[1].replace(".csv", "")
            idx = int(idx_str)
            max_idx = max(max_idx, idx)
        except:
            pass
    return max_idx


def merge_shards(output_dir, output_file):
    """Merge all shard files into a single output file."""
    pattern = os.path.join(output_dir, "*_through_*.csv")
    shard_files = sorted(glob.glob(pattern))

    if not shard_files:
        print("No shards found to merge!")
        return False

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
        # Sort by original index if available
        if 'Unnamed: 0' in merged_df.columns:
            merged_df = merged_df.sort_values(by='Unnamed: 0').reset_index(drop=True)
        merged_df.to_csv(output_file, index=False)
        print(f"Merged output saved to {output_file} ({len(merged_df)} total rows)")
        return True
    else:
        print("No valid shards to merge!")
        return False


def main():
    args = parse_args()

    # Set defaults based on mode
    if args.model is None:
        args.model = str(REPO_ROOT.parent / "models/trialchecker")

    if args.input is None:
        if args.mode == "patient_centric":
            args.input = str(REPO_ROOT.parent / "data/phi/patient_centric_candidates.csv")
        else:
            args.input = str(REPO_ROOT.parent / "data/phi/trial_centric_candidates.csv")

    if args.output_dir is None:
        if args.mode == "patient_centric":
            args.output_dir = str(REPO_ROOT.parent / "data/phi/patient_centric_eligibility_checks")
        else:
            args.output_dir = str(REPO_ROOT.parent / "data/phi/trial_centric_eligibility_checks")

    if args.output_file is None:
        if args.mode == "patient_centric":
            args.output_file = str(REPO_ROOT.parent / "data/phi/consolidated_eligibility_patient_centric.csv")
        else:
            args.output_file = str(REPO_ROOT.parent / "data/phi/consolidated_eligibility_trial_centric.csv")

    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    from vllm import LLM

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load input data
    print(f"Loading input from {args.input}...")
    candidates = pd.read_csv(args.input)
    print(f"Loaded {candidates.shape[0]} candidate pairs")

    # Initialize LLM
    print(f"Initializing LLM from {args.model}...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        download_dir=args.download_dir,
        gpu_memory_utilization=args.gpu_mem_util,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
    )

    # Check for resume point
    last_completed = get_completed_batches(str(output_dir))
    start_idx = last_completed + 1 if last_completed >= 0 else 0
    print(f"Starting from index {start_idx}")

    # Process in batches
    batch_list = []
    num_in_batch = 0

    for i in range(start_idx, candidates.shape[0]):
        batch_list.append(candidates.iloc[[i]])
        num_in_batch += 1

        if (num_in_batch == args.batch_size) or (i == (candidates.shape[0] - 1)):
            output = pd.concat(batch_list, axis=0)

            _, output['trialcheck_llm_reasoning'], output['trialcheck_llm_response'], output['eligibility_result'], output['eligibility_verdict'] = ask_about_trials_loosely(
                output['patient_summary'].astype(str).tolist(),
                output['this_space'].astype(str).tolist(),
                llm
            )

            # Determine shard filename based on mode
            if args.mode == "patient_centric":
                shard_file = output_dir / f"patient_centric_eligibility_through_{i}.csv"
            else:
                shard_file = output_dir / f"trial_centric_eligibility_through_{i}.csv"

            output.to_csv(str(shard_file), index=False)
            print(f"Saved batch through {i}")

            num_in_batch = 0
            batch_list = []

    # Merge all shards into final output file
    merge_shards(str(output_dir), args.output_file)
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
