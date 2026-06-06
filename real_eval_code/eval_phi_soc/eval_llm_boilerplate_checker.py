#!/usr/bin/env python3
"""
Evaluate OncoReasoning-3B LLM boilerplate checker performance for SOC data.

Usage:
    python eval_llm_boilerplate_checker.py --mode patient_centric --data-dir /path/to/data --output-dir /path/to/output
    python eval_llm_boilerplate_checker.py --mode trial_centric --data-dir /path/to/data --output-dir /path/to/output
"""

import argparse
import sys
from pathlib import Path

# eval_utils is in the same directory now

import pandas as pd
import numpy as np
from eval_utils import (
    eval_model,
    bootstrap_metric_ci,
    binary_auroc_score,
    format_metric_with_ci,
    load_and_combine_csv_files
)
from sklearn.metrics import roc_auc_score


# Candidate identity. (dfci_mrn, this_space) alone is NOT unique because serial
# summarization pairs the same patient/trial with multiple patient summaries.
KEY_COLS = ['dfci_mrn', 'this_space', 'patient_summary']


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate OncoReasoning-3B LLM boilerplate checker for SOC"
    )
    parser.add_argument("--mode", type=str, required=True,
                        choices=["patient_centric", "trial_centric"],
                        help="Evaluation mode")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory containing candidate and LLM result files")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save evaluation outputs")
    parser.add_argument("--llm-results-file", type=str, default=None,
                        help="Path to LLM results CSV file")
    parser.add_argument("--split-filter", type=str, default=None,
                        help="Filter to specific split")
    return parser.parse_args()


def load_predictions(results_file: Path, candidates_path: Path) -> pd.DataFrame:
    """Load OncoReasoning boilerplate results and attach candidate keys.

    Unlike the trial-check results, the boilerplate results file carries only
    ``prompt_id`` and ``exclusion_result`` (the model score) -- no key columns.
    ``prompt_id`` is the positional row index into the *full* candidates file
    (the way ``vllm_parallel_boilerplate.py`` builds its mapping), so we read the
    candidates file in the same order to recover the candidate identity for each
    prompt. Returns a DataFrame with KEY_COLS and a 'prediction' column.
    """
    print(f"Loading LLM results from: {results_file}")
    input_frame = pd.read_csv(results_file, usecols=['prompt_id', 'exclusion_result'])
    print(f"Loaded {len(input_frame)} rows")
    predictions = input_frame.groupby('prompt_id')['exclusion_result'].mean()
    print(f"Aggregated to {len(predictions)} unique prompts")

    if not candidates_path.exists():
        print(f"Candidates file not found (needed for prompt_id keys): {candidates_path}")
        return pd.DataFrame(columns=KEY_COLS + ['prediction'])

    cand = pd.read_csv(candidates_path, usecols=KEY_COLS).reset_index(drop=True)
    idx = predictions.index.to_numpy()
    in_range = idx < len(cand)
    if not in_range.all():
        print(f"Warning: {int((~in_range).sum())} prompt_ids exceed candidate rows "
              f"({len(cand)}); dropping them")
    idx = idx[in_range]
    keyed = cand.iloc[idx].copy()
    keyed['prediction'] = predictions.to_numpy()[in_range]
    return keyed


def load_gold(gold_path: Path, split_filter: str = None) -> pd.DataFrame:
    """Load reference exclusion labels from the consolidated boilerplate file."""
    print(f"Loading gold exclusion labels: {gold_path}")
    wanted = set(KEY_COLS + ['exclusion_result', 'split'])
    gold = pd.read_csv(gold_path, usecols=lambda c: c in wanted)
    gold = gold[~gold.patient_summary.isnull()]
    if split_filter and 'split' in gold.columns:
        gold = gold[gold.split.str.contains(split_filter)]
        print(f"Filtered gold to '{split_filter}' split: {len(gold)} rows")
    print(f"Loaded {len(gold)} gold standard rows")
    return gold


def evaluate_patient_centric(data_dir: Path, output_dir: Path,
                              llm_results_file: Path = None,
                              split_filter: str = None):
    """Evaluate patient-centric LLM boilerplate checker for SOC."""
    print("=" * 60)
    print("EVALUATING PATIENT-CENTRIC LLM BOILERPLATE CHECKER (SOC)")
    print("=" * 60)

    # Find LLM results file - only OncoReasoning output
    if llm_results_file is None:
        llm_results_file = data_dir / "oncoreasoning_boilerplate_patient_centric.csv"

    if not llm_results_file.exists():
        print(f"LLM results file not found: {llm_results_file}")
        return

    candidates_path = data_dir / "patient_centric_candidates.csv"
    predictions = load_predictions(llm_results_file, candidates_path)

    gold_path = data_dir / "consolidated_boilerplate_patient_centric.csv"
    if not gold_path.exists():
        print(f"Gold boilerplate file not found: {gold_path}")
        return
    gold = load_gold(gold_path, split_filter)

    merged = gold.merge(predictions, on=KEY_COLS, how='inner')
    print(f"Matched {len(merged)} prediction/gold pairs "
          f"(gold={len(gold)}, predictions={len(predictions)})")
    if len(merged) == 0:
        print("No overlapping candidates between predictions and gold; aborting.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Classification Metrics ---")
    auc = roc_auc_score(merged.exclusion_result, merged.prediction.values)
    auc_ci = bootstrap_metric_ci(
        [merged.exclusion_result.values, merged.prediction.values], binary_auroc_score
    )
    print(f"AUC: {format_metric_with_ci(auc, auc_ci)}")

    pdf_path = output_dir / "llm_boilerplate_checker_patient_centric_classification_soc.pdf"
    eval_model(
        merged.prediction.values,
        merged.exclusion_result.values,
        pdf_path=str(pdf_path),
        title_prefix="SOC LLM Boilerplate Checker Patient-Centric"
    )

    print(f"\nEvaluation complete. Reports saved to: {output_dir}")


def evaluate_trial_centric(data_dir: Path, output_dir: Path,
                            llm_results_file: Path = None,
                            split_filter: str = None):
    """Evaluate trial-centric LLM boilerplate checker for SOC."""
    print("=" * 60)
    print("EVALUATING TRIAL-CENTRIC LLM BOILERPLATE CHECKER (SOC)")
    print("=" * 60)

    # Find LLM results file - only OncoReasoning output
    if llm_results_file is None:
        llm_results_file = data_dir / "oncoreasoning_boilerplate_trial_centric.csv"

    if not llm_results_file.exists():
        print(f"LLM results file not found: {llm_results_file}")
        return

    candidates_path = data_dir / "trial_centric_candidates.csv"
    predictions = load_predictions(llm_results_file, candidates_path)

    gold_path = data_dir / "consolidated_boilerplate_trial_centric.csv"
    if not gold_path.exists():
        print(f"Gold boilerplate file not found: {gold_path}")
        return
    gold = load_gold(gold_path, split_filter)

    merged = gold.merge(predictions, on=KEY_COLS, how='inner')
    print(f"Matched {len(merged)} prediction/gold pairs "
          f"(gold={len(gold)}, predictions={len(predictions)})")
    if len(merged) == 0:
        print("No overlapping candidates between predictions and gold; aborting.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Classification Metrics ---")
    auc = roc_auc_score(merged.exclusion_result, merged.prediction.values)
    auc_ci = bootstrap_metric_ci(
        [merged.exclusion_result.values, merged.prediction.values], binary_auroc_score
    )
    print(f"AUC: {format_metric_with_ci(auc, auc_ci)}")

    pdf_path = output_dir / "llm_boilerplate_checker_trial_centric_classification_soc.pdf"
    eval_model(
        merged.prediction.values,
        merged.exclusion_result.values,
        pdf_path=str(pdf_path),
        title_prefix="SOC LLM Boilerplate Checker Trial-Centric"
    )

    print(f"\nEvaluation complete. Reports saved to: {output_dir}")


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    llm_results_file = Path(args.llm_results_file) if args.llm_results_file else None

    if args.mode == "patient_centric":
        evaluate_patient_centric(data_dir, output_dir, llm_results_file, args.split_filter)
    else:
        evaluate_trial_centric(data_dir, output_dir, llm_results_file, args.split_filter)


if __name__ == "__main__":
    main()
