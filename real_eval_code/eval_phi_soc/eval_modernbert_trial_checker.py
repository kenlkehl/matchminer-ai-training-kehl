#!/usr/bin/env python3
"""
Evaluate ModernBERT trial checker regression model performance for SOC data.

Usage:
    python eval_modernbert_trial_checker.py --mode patient_centric --data-dir /path/to/data --output-dir /path/to/output
    python eval_modernbert_trial_checker.py --mode trial_centric --data-dir /path/to/data --output-dir /path/to/output
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from eval_utils import (
    eval_model,
    average_precision_at_k,
    generate_ranking_report,
    load_and_combine_csv_files
)
from sklearn.metrics import roc_auc_score, cohen_kappa_score
from scipy.stats import spearmanr, pearsonr


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate ModernBERT trial checker regression model for SOC"
    )
    parser.add_argument("--mode", type=str, required=True,
                        choices=["patient_centric", "trial_centric"],
                        help="Evaluation mode")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory containing candidate CSV files")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save evaluation outputs")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to trial checker model")
    parser.add_argument("--gpu", type=str, default="0",
                        help="GPU device to use")
    parser.add_argument("--k", type=int, default=20,
                        help="K value for MAP@K calculation (default: 20)")
    parser.add_argument("--run-inference", action="store_true",
                        help="Run model inference (requires GPU)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for inference")
    parser.add_argument("--split-filter", type=str, default=None,
                        help="Filter to specific split (e.g., 'test')")
    # Sharding arguments for multi-GPU parallelization
    parser.add_argument("--shard-id", type=int, default=None,
                        help="Shard ID (0-indexed) for parallel execution")
    parser.add_argument("--num-shards", type=int, default=None,
                        help="Total number of shards for parallel execution")
    parser.add_argument("--shard-dir", type=str, default=None,
                        help="Directory for shard outputs (enables parallel mode)")
    return parser.parse_args()


def run_trial_checker_inference(df: pd.DataFrame, model_path: str,
                                 device: str = "cuda", batch_size: int = 32) -> pd.DataFrame:
    """
    Run regression trial checker inference on patient-trial pairs.

    Model outputs a single logit per sample. We apply sigmoid and scale to [0, 5].
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch

    print(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    df = df.copy()
    df['pt_trial_pair'] = (
        df['this_space'] +
        "\nNow here is the patient summary:" +
        df['patient_summary']
    )
    df = df[~df.pt_trial_pair.isnull()]

    print(f"Running inference on {len(df)} samples...")

    all_scores = []
    all_logits = []

    for i in range(0, len(df), batch_size):
        batch_texts = df.pt_trial_pair.iloc[i:i+batch_size].tolist()
        inputs = tokenizer(batch_texts, truncation=True, padding=True,
                          max_length=4096, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits.squeeze(-1)
            scores = torch.sigmoid(logits) * 5.0
        all_logits.extend(logits.cpu().numpy().tolist())
        all_scores.extend(scores.cpu().numpy().tolist())
        if (i // batch_size) % 10 == 0:
            print(f"  Processed {min(i + batch_size, len(df))}/{len(df)}")

    df = df.reset_index(drop=True)
    df['prediction_score'] = all_scores          # continuous score 0-5
    df['prediction_logit'] = all_logits          # raw logit
    df['prediction_label'] = np.where(
        np.array(all_scores) > 0.5, 'POSITIVE', 'NEGATIVE'
    )

    return df


def merge_shards(shard_dir: Path, output_path: Path, num_shards: int) -> pd.DataFrame:
    """Merge completed shard files into single output."""
    dfs = []
    for i in range(num_shards):
        shard_path = shard_dir / f"shard_{i}.csv"
        if shard_path.exists():
            dfs.append(pd.read_csv(shard_path))
        else:
            print(f"Warning: Missing shard file {shard_path}")
    if not dfs:
        print("No shard files found to merge")
        return None
    combined = pd.concat(dfs, axis=0).reset_index(drop=True)
    combined.to_csv(output_path, index=False)
    print(f"Merged {len(dfs)} shards ({len(combined)} rows) into {output_path}")
    return combined


def _evaluate_classification_and_ranking(validation_set: pd.DataFrame, output_dir: Path,
                                         group_col: str, mode_label: str, k: int):
    """Run classification metrics, regression metrics, and ranking evaluation."""
    if 'prediction_score' not in validation_set.columns or 'eligibility_result' not in validation_set.columns:
        print("Missing prediction_score or eligibility_result columns, skipping evaluation")
        return

    # --- Classification Metrics (binarized) ---
    print("\n--- Classification Metrics ---")
    gold_binary = (validation_set.eligibility_result > 0).astype(float)

    auc = roc_auc_score(gold_binary, validation_set.prediction_score)
    print(f"AUC: {auc:.4f}")

    gold_scores = validation_set.eligibility_result.values

    pdf_path = output_dir / f"trial_checker_{mode_label}_classification_soc.pdf"
    eval_model(
        validation_set.prediction_score.values,
        gold_binary.values,
        pdf_path=str(pdf_path),
        title_prefix=f"SOC Trial Checker {mode_label.replace('_', ' ').title()}",
        gold_continuous=gold_scores
    )

    if 'prediction_label' in validation_set.columns:
        actual_labels = np.where(
            validation_set.eligibility_result == 0.0, 'NEGATIVE', 'POSITIVE'
        )
        kappa = cohen_kappa_score(actual_labels, validation_set.prediction_label.values)
        print(f"Cohen's Kappa: {kappa:.4f}")

        print("\nCrosstab (actual vs predicted):")
        print(pd.crosstab(
            pd.Series(actual_labels, name='actual'),
            pd.Series(validation_set.prediction_label.values, name='predicted')
        ))

    # --- Regression Metrics ---
    print("\n--- Regression Metrics ---")
    pred_scores = validation_set.prediction_score.values
    r, p_r = pearsonr(gold_scores, pred_scores)
    rho, p_rho = spearmanr(gold_scores, pred_scores)
    mae = np.mean(np.abs(gold_scores - pred_scores))
    print(f"Pearson r: {r:.4f} (p={p_r:.4e})")
    print(f"Spearman rho: {rho:.4f} (p={p_rho:.4e})")
    print(f"MAE: {mae:.4f}")

    # --- Ranking Metrics (score-based top-K) ---
    print(f"\n--- Ranking Metrics (top-{k} by regression score) ---")
    scored_set = validation_set.sort_values(
        by=[group_col, 'prediction_score'], ascending=[True, False]
    ).copy()
    scored_set['_binary_label'] = (scored_set.eligibility_result > 0).astype(float)
    top_k = scored_set.groupby(group_col).head(k)

    print(f"Samples in top-{k}: {len(top_k)}")
    if len(top_k) > 0:
        print(f"Positive rate in top-{k}: {top_k._binary_label.mean():.4f}")

        temp = top_k.groupby(group_col)._binary_label.apply(
            lambda x: average_precision_at_k(x.values)
        )
        map_k = temp.mean()
        print(f"MAP@{k}: {map_k:.4f}")

        pdf_path = output_dir / f"trial_checker_{mode_label}_ranking_soc.pdf"
        generate_ranking_report(
            top_k,
            group_col=group_col,
            label_col='_binary_label',
            pdf_path=str(pdf_path),
            title_prefix=f"SOC Trial Checker {mode_label.replace('_', ' ').title()} (Score-Ranked)",
            k=k
        )


def evaluate_patient_centric(data_dir: Path, output_dir: Path,
                              model_path: str = None, gpu: str = "0",
                              k: int = 20, run_inference: bool = False,
                              batch_size: int = 32, split_filter: str = None,
                              shard_id: int = None, num_shards: int = None,
                              shard_dir: str = None):
    """Evaluate patient-centric trial checker performance for SOC."""
    print("=" * 60)
    print("EVALUATING PATIENT-CENTRIC TRIAL CHECKER (SOC, REGRESSION)")
    if shard_id is not None:
        print(f"(Shard {shard_id + 1}/{num_shards})")
    print("=" * 60)

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    # Handle merge-only mode (shard_dir specified but no shard_id)
    shard_dir_path = Path(shard_dir) if shard_dir else None
    if shard_dir_path and num_shards and shard_id is None:
        print("Merge mode: combining shards and running evaluation...")
        output_dir.mkdir(parents=True, exist_ok=True)
        intermediate_path = output_dir / "patient_centric_with_predictions_soc.csv"
        validation_set = merge_shards(shard_dir_path, intermediate_path, num_shards)
        if validation_set is None:
            return
    else:
        consolidated_path = data_dir / "consolidated_eligibility_patient_centric.csv"
        if not consolidated_path.exists():
            print(f"Consolidated eligibility file not found: {consolidated_path}")
            return

        print(f"Loading consolidated file: {consolidated_path}")
        combined_df = pd.read_csv(consolidated_path)
        print(f"Loaded {len(combined_df)} rows")

        if split_filter and 'split' in combined_df.columns:
            combined_df = combined_df[combined_df.split.str.contains(split_filter)]
            print(f"Filtered to {split_filter} split: {len(combined_df)} rows")

        combined_df = combined_df[~combined_df.patient_summary.isnull()]
        validation_set = combined_df.copy()

        # Apply sharding if specified
        if shard_id is not None and num_shards is not None:
            validation_set = validation_set.iloc[shard_id::num_shards].reset_index(drop=True)
            print(f"Processing shard {shard_id + 1}/{num_shards}: {len(validation_set)} samples")

        print(f"Unique trial spaces: {validation_set.this_space.nunique()}")
        print(f"Unique patients: {validation_set.patient_summary.nunique()}")

        if run_inference and model_path:
            validation_set = run_trial_checker_inference(
                validation_set, model_path, device='cuda', batch_size=batch_size
            )
            if shard_dir_path and shard_id is not None:
                shard_dir_path.mkdir(parents=True, exist_ok=True)
                shard_path = shard_dir_path / f"shard_{shard_id}.csv"
                validation_set.to_csv(shard_path, index=False)
                print(f"Saved shard to: {shard_path}")
                return
            else:
                output_dir.mkdir(parents=True, exist_ok=True)
                intermediate_path = output_dir / "patient_centric_with_predictions_soc.csv"
                validation_set.to_csv(intermediate_path, index=False)
                print(f"Saved predictions to: {intermediate_path}")
        elif 'prediction_score' not in validation_set.columns:
            precomputed_path = output_dir / "patient_centric_with_predictions_soc.csv"
            if precomputed_path.exists():
                print(f"Loading pre-computed predictions from: {precomputed_path}")
                validation_set = pd.read_csv(precomputed_path)
            else:
                print("No predictions available. Use --run-inference to generate them.")
                return

    output_dir.mkdir(parents=True, exist_ok=True)

    _evaluate_classification_and_ranking(validation_set, output_dir,
                                         'patient_summary', 'patient_centric', k)

    print(f"\nEvaluation complete. Reports saved to: {output_dir}")


def evaluate_trial_centric(data_dir: Path, output_dir: Path,
                            model_path: str = None, gpu: str = "0",
                            k: int = 20, run_inference: bool = False,
                            batch_size: int = 32, split_filter: str = None,
                            shard_id: int = None, num_shards: int = None,
                            shard_dir: str = None):
    """Evaluate trial-centric trial checker performance for SOC."""
    print("=" * 60)
    print("EVALUATING TRIAL-CENTRIC TRIAL CHECKER (SOC, REGRESSION)")
    if shard_id is not None:
        print(f"(Shard {shard_id + 1}/{num_shards})")
    print("=" * 60)

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    # Handle merge-only mode (shard_dir specified but no shard_id)
    shard_dir_path = Path(shard_dir) if shard_dir else None
    if shard_dir_path and num_shards and shard_id is None:
        print("Merge mode: combining shards and running evaluation...")
        output_dir.mkdir(parents=True, exist_ok=True)
        intermediate_path = output_dir / "trial_centric_with_predictions_soc.csv"
        validation_set = merge_shards(shard_dir_path, intermediate_path, num_shards)
        if validation_set is None:
            return
    else:
        consolidated_path = data_dir / "consolidated_eligibility_trial_centric.csv"
        if not consolidated_path.exists():
            print(f"Consolidated eligibility file not found: {consolidated_path}")
            return

        print(f"Loading consolidated file: {consolidated_path}")
        combined_df = pd.read_csv(consolidated_path)
        print(f"Loaded {len(combined_df)} rows")

        if split_filter and 'split' in combined_df.columns:
            combined_df = combined_df[combined_df.split.str.contains(split_filter)]
            print(f"Filtered to {split_filter} split: {len(combined_df)} rows")

        combined_df = combined_df[~combined_df.patient_summary.isnull()]
        validation_set = combined_df.copy()

        # Apply sharding if specified
        if shard_id is not None and num_shards is not None:
            validation_set = validation_set.iloc[shard_id::num_shards].reset_index(drop=True)
            print(f"Processing shard {shard_id + 1}/{num_shards}: {len(validation_set)} samples")

        print(f"Unique trial spaces: {validation_set.this_space.nunique()}")
        print(f"Unique patients: {validation_set.patient_summary.nunique()}")

        if run_inference and model_path:
            validation_set = run_trial_checker_inference(
                validation_set, model_path, device='cuda', batch_size=batch_size
            )
            if shard_dir_path and shard_id is not None:
                shard_dir_path.mkdir(parents=True, exist_ok=True)
                shard_path = shard_dir_path / f"shard_{shard_id}.csv"
                validation_set.to_csv(shard_path, index=False)
                print(f"Saved shard to: {shard_path}")
                return
            else:
                output_dir.mkdir(parents=True, exist_ok=True)
                intermediate_path = output_dir / "trial_centric_with_predictions_soc.csv"
                validation_set.to_csv(intermediate_path, index=False)
                print(f"Saved predictions to: {intermediate_path}")
        elif 'prediction_score' not in validation_set.columns:
            precomputed_path = output_dir / "trial_centric_with_predictions_soc.csv"
            if precomputed_path.exists():
                print(f"Loading pre-computed predictions from: {precomputed_path}")
                validation_set = pd.read_csv(precomputed_path)
            else:
                print("No predictions available. Use --run-inference to generate them.")
                return

    output_dir.mkdir(parents=True, exist_ok=True)

    _evaluate_classification_and_ranking(validation_set, output_dir,
                                         'this_space', 'trial_centric', k)

    print(f"\nEvaluation complete. Reports saved to: {output_dir}")


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if args.mode == "patient_centric":
        evaluate_patient_centric(
            data_dir, output_dir,
            model_path=args.model_path,
            gpu=args.gpu,
            k=args.k,
            run_inference=args.run_inference,
            batch_size=args.batch_size,
            split_filter=args.split_filter,
            shard_id=args.shard_id,
            num_shards=args.num_shards,
            shard_dir=args.shard_dir
        )
    else:
        evaluate_trial_centric(
            data_dir, output_dir,
            model_path=args.model_path,
            gpu=args.gpu,
            k=args.k,
            run_inference=args.run_inference,
            batch_size=args.batch_size,
            split_filter=args.split_filter,
            shard_id=args.shard_id,
            num_shards=args.num_shards,
            shard_dir=args.shard_dir
        )


if __name__ == "__main__":
    main()
