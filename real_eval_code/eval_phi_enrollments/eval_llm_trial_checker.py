#!/usr/bin/env python3
"""
Evaluate OncoReasoning-3B LLM trial checker performance.

This script evaluates the OncoReasoning-3B LLM-based trial eligibility checker by:
- Loading pre-computed LLM inference results
- Computing classification metrics (AUC, F1, precision-recall)
- Computing ranking metrics (MAP@K)
- Generating PDF reports

The script expects:
- LLM results file with prompt_id and eligibility_result columns (the model score)
- Consolidated eligibility file with the reference eligibility_result labels

Usage:
    python eval_llm_trial_checker.py --mode patient_centric --data-dir /path/to/data --output-dir /path/to/output
    python eval_llm_trial_checker.py --mode trial_centric --data-dir /path/to/data --output-dir /path/to/output
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
    calculate_map_at_k,
    format_metric_with_ci,
    generate_ranking_report,
    load_and_combine_csv_files,
    positive_rate_metric
)
from sklearn.metrics import roc_auc_score


# Candidate identity. (dfci_mrn, this_space) alone is NOT unique because serial
# summarization pairs the same patient/trial with multiple patient summaries.
KEY_COLS = ['dfci_mrn', 'this_space', 'patient_summary']


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate OncoReasoning-3B LLM trial checker"
    )
    parser.add_argument("--mode", type=str, required=True,
                        choices=["patient_centric", "trial_centric"],
                        help="Evaluation mode")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory containing candidate and LLM result files")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save evaluation outputs")
    parser.add_argument("--llm-results-file", type=str, default=None,
                        help="Path to LLM results CSV file (overrides auto-detection)")
    parser.add_argument("--k", type=int, default=None,
                        help="K for MAP@K/NDCG@K (default: 20 patient-centric, 40 trial-centric)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold for positive classification (default: 0.5)")
    return parser.parse_args()


def load_predictions(results_file: Path) -> pd.DataFrame:
    """Load OncoReasoning trial-check results, one row per prompt.

    The results file's ``eligibility_result`` column is the *model* score (0-5),
    and every row carries the candidate key columns. We average the score over the
    repeated samples per ``prompt_id`` and keep the keys so predictions can be
    aligned to the gold labels by candidate identity (the shard merges that build
    the gold file do not preserve candidate row order).

    Returns a DataFrame with columns: prompt_id, prediction, and KEY_COLS.
    """
    print(f"Loading LLM results from: {results_file}")
    input_frame = pd.read_csv(
        results_file, usecols=['prompt_id', 'eligibility_result'] + KEY_COLS
    )
    print(f"Loaded {len(input_frame)} rows")
    # Drop the model's own PARSE_FAILED samples (eligibility_result == -1) so a sentinel
    # is never averaged into a prompt's mean prediction (which would push it below 0).
    n_before = len(input_frame)
    input_frame = input_frame[input_frame.eligibility_result >= 0]
    if n_before - len(input_frame):
        print(f"Dropped {n_before - len(input_frame)} unparseable prediction samples "
              f"(eligibility_result == -1)")
    predictions = input_frame.groupby('prompt_id', as_index=False).agg(
        prediction=('eligibility_result', 'mean'),
        dfci_mrn=('dfci_mrn', 'first'),
        this_space=('this_space', 'first'),
        patient_summary=('patient_summary', 'first'),
    )
    print(f"Aggregated to {len(predictions)} unique prompts")
    return predictions


def load_gold(gold_path: Path) -> pd.DataFrame:
    """Load reference eligibility labels from the consolidated eligibility file."""
    print(f"Loading gold eligibility labels: {gold_path}")
    wanted = set(KEY_COLS + ['eligibility_result', 'split'])
    gold = pd.read_csv(gold_path, usecols=lambda c: c in wanted)
    gold = gold[~gold.patient_summary.isnull()]
    # Drop rows where the gold-labeling LLM could not parse a score (eligibility_result == -1).
    # These PARSE_FAILED sentinels are not real grades and would silently become false
    # negatives in the binarized classification labels and corrupt the ranking/positive rate.
    n_before = len(gold)
    gold = gold[gold.eligibility_result >= 0]
    if n_before - len(gold):
        print(f"Dropped {n_before - len(gold)} gold rows with unparseable labels "
              f"(eligibility_result == -1)")
    print(f"Loaded {len(gold)} gold standard rows")
    return gold


def align_predictions_to_gold(predictions: pd.DataFrame,
                               gold: pd.DataFrame) -> pd.DataFrame:
    """Merge predictions onto gold by candidate identity, restoring rank order.

    Ascending ``prompt_id`` is retrieval-rank order; ``calculate_map_at_k`` assumes
    the frame is pre-sorted by rank within each group, so we sort by it here.
    """
    merged = gold.merge(predictions, on=KEY_COLS, how='inner')
    print(f"Matched {len(merged)} prediction/gold pairs "
          f"(gold={len(gold)}, predictions={len(predictions)})")
    merged = merged.sort_values('prompt_id').reset_index(drop=True)
    return merged


def evaluate_patient_centric(data_dir: Path, output_dir: Path,
                              llm_results_file: Path = None,
                              k: int = 20, threshold: float = 0.5):
    """Evaluate patient-centric LLM trial checker performance."""
    print("=" * 60)
    print("EVALUATING PATIENT-CENTRIC LLM TRIAL CHECKER")
    print("=" * 60)

    # Find LLM results file - only OncoReasoning output
    if llm_results_file is None:
        llm_results_file = data_dir / "oncoreasoning_trialcheck_patient_centric.csv"

    if not llm_results_file.exists():
        print(f"LLM results file not found: {llm_results_file}")
        return

    predictions = load_predictions(llm_results_file)

    gold_path = data_dir / "consolidated_eligibility_patient_centric.csv"
    if not gold_path.exists():
        print(f"Gold eligibility file not found: {gold_path}")
        return
    gold = load_gold(gold_path)

    merged = align_predictions_to_gold(predictions, gold)
    if len(merged) == 0:
        print("No overlapping candidates between predictions and gold; aborting.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute classification metrics (binarize graded labels for AUC)
    print("\n--- Classification Metrics ---")
    gold_binary = (merged.eligibility_result > 0).astype(float)
    preds = merged.prediction.values
    auc = roc_auc_score(gold_binary, preds)
    auc_ci = bootstrap_metric_ci(
        [gold_binary.values, preds], binary_auroc_score
    )
    print(f"AUC: {format_metric_with_ci(auc, auc_ci)}")

    # Generate classification PDF report
    pdf_path = output_dir / "llm_trial_checker_patient_centric_classification.pdf"
    eval_model(
        preds,
        gold_binary.values,
        pdf_path=str(pdf_path),
        title_prefix="LLM Trial Checker Patient-Centric"
    )

    # Compute ranking metrics after filtering to positive predictions
    print(f"\n--- Ranking Metrics (threshold={threshold}) ---")

    pruned = merged[merged.prediction >= threshold].copy()

    print(f"Samples after filtering: {len(pruned)}")
    if len(pruned) > 0:
        print(f"Median results per patient: {pruned.groupby('patient_summary').size().median():.1f}")
        print(f"Mean results per patient: {pruned.groupby('patient_summary').size().mean():.1f}")
        print(f"Unique patients: {pruned.patient_summary.nunique()}")
        print(f"Unique trial spaces: {pruned.this_space.nunique()}")
        positive_rate = pruned.eligibility_result.mean()
        positive_rate_ci = bootstrap_metric_ci(
            [pruned.eligibility_result.values], positive_rate_metric
        )
        print(f"Positive rate after filtering: {format_metric_with_ci(positive_rate, positive_rate_ci)}")

        # Calculate MAP@K on filtered set
        map_k, ranking_stats = calculate_map_at_k(
            pruned, 'patient_summary', 'eligibility_result', k
        )
        print(f"MAP@{k} (after LLM check): {format_metric_with_ci(map_k, ranking_stats.get('map_at_k_ci'))}")

        # Generate ranking PDF report
        pdf_path = output_dir / "llm_trial_checker_patient_centric_ranking.pdf"
        generate_ranking_report(
            pruned,
            group_col='patient_summary',
            label_col='eligibility_result',
            pdf_path=str(pdf_path),
            title_prefix="LLM Trial Checker Patient-Centric (Filtered)",
            k=k
        )

    print(f"\nEvaluation complete. Reports saved to: {output_dir}")


def evaluate_trial_centric(data_dir: Path, output_dir: Path,
                            llm_results_file: Path = None,
                            k: int = 20, threshold: float = 0.5):
    """Evaluate trial-centric LLM trial checker performance."""
    print("=" * 60)
    print("EVALUATING TRIAL-CENTRIC LLM TRIAL CHECKER")
    print("=" * 60)

    # Find LLM results file - only OncoReasoning output
    if llm_results_file is None:
        llm_results_file = data_dir / "oncoreasoning_trialcheck_trial_centric.csv"

    if not llm_results_file.exists():
        print(f"LLM results file not found: {llm_results_file}")
        return

    predictions = load_predictions(llm_results_file)

    gold_path = data_dir / "consolidated_eligibility_trial_centric.csv"
    if not gold_path.exists():
        print(f"Gold eligibility file not found: {gold_path}")
        return
    gold = load_gold(gold_path)

    merged = align_predictions_to_gold(predictions, gold)
    if len(merged) == 0:
        print("No overlapping candidates between predictions and gold; aborting.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute classification metrics (binarize graded labels for AUC)
    print("\n--- Classification Metrics ---")
    gold_binary = (merged.eligibility_result > 0).astype(float)
    preds = merged.prediction.values
    auc = roc_auc_score(gold_binary, preds)
    auc_ci = bootstrap_metric_ci(
        [gold_binary.values, preds], binary_auroc_score
    )
    print(f"AUC: {format_metric_with_ci(auc, auc_ci)}")

    pdf_path = output_dir / "llm_trial_checker_trial_centric_classification.pdf"
    eval_model(
        preds,
        gold_binary.values,
        pdf_path=str(pdf_path),
        title_prefix="LLM Trial Checker Trial-Centric"
    )

    # Compute ranking metrics
    print(f"\n--- Ranking Metrics (threshold={threshold}) ---")

    pruned = merged[merged.prediction >= threshold].copy()

    print(f"Samples after filtering: {len(pruned)}")
    if len(pruned) > 0:
        print(f"Median results per trial: {pruned.groupby('this_space').size().median():.1f}")
        print(f"Mean results per trial: {pruned.groupby('this_space').size().mean():.1f}")
        print(f"Unique patients: {pruned.patient_summary.nunique()}")
        print(f"Unique trial spaces: {pruned.this_space.nunique()}")
        positive_rate = pruned.eligibility_result.mean()
        positive_rate_ci = bootstrap_metric_ci(
            [pruned.eligibility_result.values], positive_rate_metric
        )
        print(f"Positive rate after filtering: {format_metric_with_ci(positive_rate, positive_rate_ci)}")

        # For trial-centric, group by trial space
        map_k, ranking_stats = calculate_map_at_k(
            pruned, 'this_space', 'eligibility_result', k
        )
        print(f"MAP@{k} (after LLM check): {format_metric_with_ci(map_k, ranking_stats.get('map_at_k_ci'))}")

        pdf_path = output_dir / "llm_trial_checker_trial_centric_ranking.pdf"
        generate_ranking_report(
            pruned,
            group_col='this_space',
            label_col='eligibility_result',
            pdf_path=str(pdf_path),
            title_prefix="LLM Trial Checker Trial-Centric (Filtered)",
            k=k
        )

    print(f"\nEvaluation complete. Reports saved to: {output_dir}")


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    llm_results_file = Path(args.llm_results_file) if args.llm_results_file else None
    # Ranking depth matches retrieval depth: 20 patient-centric, 40 trial-centric.
    k = args.k if args.k is not None else (40 if args.mode == "trial_centric" else 20)

    if args.mode == "patient_centric":
        evaluate_patient_centric(
            data_dir, output_dir,
            llm_results_file=llm_results_file,
            k=k,
            threshold=args.threshold
        )
    else:
        evaluate_trial_centric(
            data_dir, output_dir,
            llm_results_file=llm_results_file,
            k=k,
            threshold=args.threshold
        )


if __name__ == "__main__":
    main()
