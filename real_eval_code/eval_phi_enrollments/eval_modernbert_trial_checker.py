#!/usr/bin/env python3
"""
Evaluate ModernBERT trial checker classifier performance.

This script evaluates the ModernBERT-based trial eligibility classifier by:
- Running inference on patient-trial pairs
- Computing classification metrics (AUC, F1, precision-recall)
- Computing ranking metrics (MAP@K)
- Generating PDF reports

Supports both binary (POSITIVE/NEGATIVE) and categorical (5-class verdict) models.

Usage:
    python eval_trial_checker.py --mode patient_centric --data-dir /path/to/data --output-dir /path/to/output
    python eval_trial_checker.py --mode trial_centric --data-dir /path/to/data --output-dir /path/to/output
    python eval_trial_checker.py --mode patient_centric --categorical --data-dir /path/to/data --output-dir /path/to/output
"""

import argparse
import os
import sys
from pathlib import Path

# eval_utils is in the same directory now

import pandas as pd
import numpy as np
from eval_utils import (
    eval_model,
    eval_model_categorical,
    average_precision_at_k,
    generate_ranking_report,
    load_and_combine_csv_files
)
from sklearn.metrics import roc_auc_score, cohen_kappa_score

VERDICT_LABELS = ["NO!", "YES-GENERAL!", "YES-CANCERMATCH!", "YES-BIOMARKERMATCH!", "YES-TARGETED!"]
VERDICT_LABEL2ID = {v: i for i, v in enumerate(VERDICT_LABELS)}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate ModernBERT trial checker classifier"
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
    parser.add_argument("--categorical", action="store_true",
                        help="Use 5-class categorical model instead of binary")
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
    Run trial checker model inference on patient-trial pairs.

    Args:
        df: DataFrame with patient_summary and this_space columns
        model_path: Path to the trial checker model
        device: Device to use for inference
        batch_size: Batch size

    Returns:
        DataFrame with predictions added
    """
    from transformers import pipeline, AutoTokenizer

    print(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    pipe = pipeline(
        'text-classification',
        model_path,
        tokenizer=tokenizer,
        truncation=True,
        padding='max_length',
        max_length=4096,
        device=device,
        batch_size=batch_size
    )

    # Create patient-trial pair text
    df = df.copy()
    df['pt_trial_pair'] = (
        df['this_space'] +
        "\nNow here is the patient summary:" +
        df['patient_summary']
    )
    df = df[~df.pt_trial_pair.isnull()]

    print(f"Running inference on {len(df)} samples...")
    predictions = pipe(df.pt_trial_pair.tolist())

    # Process predictions
    predictions_df = pd.DataFrame(predictions)
    predictions_df['score'] = np.where(
        predictions_df.label == 'NEGATIVE',
        1 - predictions_df.score,
        predictions_df.score
    )
    predictions_df['logit_score'] = np.log(
        predictions_df.score + 1e-6 / (1 - predictions_df.score + 1e-6)
    )

    # Merge predictions back
    df = df.reset_index(drop=True)
    df['prediction_label'] = predictions_df['label']
    df['prediction_score'] = predictions_df['score']
    df['prediction_logit'] = predictions_df['logit_score']

    return df


def run_trial_checker_inference_categorical(df: pd.DataFrame, model_path: str,
                                            device: str = "cuda",
                                            batch_size: int = 32) -> pd.DataFrame:
    """
    Run categorical (5-class) trial checker model inference on patient-trial pairs.

    Returns all class probabilities plus a composite ranking score (1 - P(NO!)).
    """
    from transformers import pipeline, AutoTokenizer

    print(f"Loading categorical model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    pipe = pipeline(
        'text-classification',
        model_path,
        tokenizer=tokenizer,
        truncation=True,
        padding='max_length',
        max_length=4096,
        device=device,
        batch_size=batch_size,
        top_k=None,  # Return all class scores
    )

    df = df.copy()
    df['pt_trial_pair'] = (
        df['this_space'] +
        "\nNow here is the patient summary:" +
        df['patient_summary']
    )
    df = df[~df.pt_trial_pair.isnull()]

    print(f"Running categorical inference on {len(df)} samples...")
    all_predictions = pipe(df.pt_trial_pair.tolist())

    # Process predictions: each item is a list of {label, score} dicts for all classes
    pred_labels = []
    pred_scores = {lbl: [] for lbl in VERDICT_LABELS}
    composite_scores = []

    for sample_preds in all_predictions:
        # Build score lookup for this sample
        score_map = {p['label']: p['score'] for p in sample_preds}
        # Best class
        best = max(sample_preds, key=lambda x: x['score'])
        pred_labels.append(best['label'])
        # Per-class scores
        for lbl in VERDICT_LABELS:
            pred_scores[lbl].append(score_map.get(lbl, 0.0))
        # Composite: P(any YES) = 1 - P(NO!)
        composite_scores.append(1.0 - score_map.get("NO!", 0.0))

    df = df.reset_index(drop=True)
    df['prediction_label'] = pred_labels
    df['prediction_score'] = composite_scores  # 1 - P(NO!) for ranking
    df['prediction_logit'] = np.log(
        np.array(composite_scores) + 1e-6 / (1 - np.array(composite_scores) + 1e-6)
    )
    for lbl in VERDICT_LABELS:
        col_name = 'prediction_score_' + lbl.replace('!', '').replace('-', '_')
        df[col_name] = pred_scores[lbl]

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


def _get_categorical_probs(validation_set: pd.DataFrame) -> np.ndarray:
    """Extract per-class probability columns into an (N, 5) array."""
    prob_cols = []
    for lbl in VERDICT_LABELS:
        col_name = 'prediction_score_' + lbl.replace('!', '').replace('-', '_')
        prob_cols.append(validation_set[col_name].values)
    return np.column_stack(prob_cols)


def _evaluate_categorical_classification(validation_set: pd.DataFrame, output_dir: Path,
                                         mode_label: str):
    """Run categorical classification metrics and generate PDF."""
    if 'eligibility_verdict' not in validation_set.columns:
        print("Warning: eligibility_verdict column not found, skipping categorical classification metrics")
        return

    # Filter to rows with known verdicts
    mask = validation_set.eligibility_verdict.isin(VERDICT_LABELS)
    eval_set = validation_set[mask].copy()
    if len(eval_set) == 0:
        print("No rows with valid eligibility_verdict for categorical evaluation")
        return

    print(f"\n--- Categorical Classification Metrics ({len(eval_set)} samples) ---")

    gold_labels = eval_set.eligibility_verdict.map(VERDICT_LABEL2ID).values
    predicted_probs = _get_categorical_probs(eval_set)

    pdf_path = output_dir / f"trial_checker_{mode_label}_classification_categorical.pdf"
    eval_model_categorical(
        predicted_probs, gold_labels,
        class_names=VERDICT_LABELS,
        pdf_path=str(pdf_path),
        title_prefix=f"Categorical Trial Checker {mode_label.replace('_', ' ').title()}"
    )

    # Crosstab
    print("\nCrosstab (actual verdict vs predicted):")
    print(pd.crosstab(eval_set.eligibility_verdict, eval_set.prediction_label))


def _evaluate_categorical_ranking(validation_set: pd.DataFrame, output_dir: Path,
                                  group_col: str, mode_label: str, k: int):
    """Run both ranking approaches for categorical model."""
    if 'eligibility_result' not in validation_set.columns:
        return

    gold_binary = (validation_set.eligibility_result > 0).astype(float)

    # Approach 1: Binary-collapsed (any YES-* = POSITIVE)
    print(f"\n--- Ranking Metrics: Binary-Collapsed (any YES-* = POSITIVE) ---")
    pruned_set = validation_set[validation_set.prediction_label != "NO!"].copy()
    if group_col == 'patient_summary':
        pruned_set = validation_set.groupby(group_col).head(k)
        pruned_set = pruned_set[pruned_set.prediction_label != "NO!"]

    print(f"Samples after filtering: {len(pruned_set)}")
    if len(pruned_set) > 0 and 'eligibility_result' in pruned_set.columns:
        pruned_binary = (pruned_set.eligibility_result > 0).astype(float)
        print(f"Positive rate after filtering: {pruned_binary.mean():.4f}")

        temp = pruned_set.groupby(group_col).eligibility_result.apply(
            lambda x: average_precision_at_k((x.head(k) > 0).astype(float).values)
        )
        map_k = temp.mean()
        print(f"MAP@{k} (binary-collapsed): {map_k:.4f}")

        pdf_path = output_dir / f"trial_checker_{mode_label}_ranking_categorical_binary.pdf"
        # Need binary labels for generate_ranking_report
        pruned_set = pruned_set.copy()
        pruned_set['_binary_label'] = (pruned_set.eligibility_result > 0).astype(float)
        generate_ranking_report(
            pruned_set,
            group_col=group_col,
            label_col='_binary_label',
            pdf_path=str(pdf_path),
            title_prefix=f"Categorical Trial Checker {mode_label.replace('_', ' ').title()} (Binary-Collapsed)",
            k=k
        )

    # Approach 2: Score-based ranking using prediction_score (= 1 - P(NO!))
    print(f"\n--- Ranking Metrics: Score-Based (sorted by 1 - P(NO!)) ---")
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
        print(f"MAP@{k} (score-based): {map_k:.4f}")

        pdf_path = output_dir / f"trial_checker_{mode_label}_ranking_categorical_scored.pdf"
        generate_ranking_report(
            top_k,
            group_col=group_col,
            label_col='_binary_label',
            pdf_path=str(pdf_path),
            title_prefix=f"Categorical Trial Checker {mode_label.replace('_', ' ').title()} (Score-Based)",
            k=k
        )


def evaluate_patient_centric(data_dir: Path, output_dir: Path,
                              model_path: str = None, gpu: str = "0",
                              k: int = 20, run_inference: bool = False,
                              batch_size: int = 32, categorical: bool = False,
                              shard_id: int = None, num_shards: int = None,
                              shard_dir: str = None):
    """Evaluate patient-centric trial checker performance."""
    model_type = "CATEGORICAL" if categorical else "BINARY"
    print("=" * 60)
    print(f"EVALUATING PATIENT-CENTRIC TRIAL CHECKER ({model_type})")
    if shard_id is not None:
        print(f"(Shard {shard_id + 1}/{num_shards})")
    print("=" * 60)

    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    suffix = "_categorical" if categorical else ""

    # Handle merge-only mode (shard_dir specified but no shard_id)
    shard_dir_path = Path(shard_dir) if shard_dir else None
    if shard_dir_path and num_shards and shard_id is None:
        print("Merge mode: combining shards and running evaluation...")
        output_dir.mkdir(parents=True, exist_ok=True)
        intermediate_path = output_dir / f"patient_centric_with_predictions{suffix}.csv"
        validation_set = merge_shards(shard_dir_path, intermediate_path, num_shards)
        if validation_set is None:
            return
    else:
        # Load consolidated eligibility results from GPT checks
        consolidated_path = data_dir / "consolidated_eligibility_patient_centric.csv"
        if not consolidated_path.exists():
            print(f"Consolidated eligibility file not found: {consolidated_path}")
            return

        print(f"Loading consolidated file: {consolidated_path}")
        combined_df = pd.read_csv(consolidated_path)

        print(f"Loaded {len(combined_df)} rows")

        # Filter out null patient summaries
        combined_df = combined_df[~combined_df.patient_summary.isnull()]

        validation_set = combined_df.copy()

        # Apply sharding if specified
        if shard_id is not None and num_shards is not None:
            validation_set = validation_set.iloc[shard_id::num_shards].reset_index(drop=True)
            print(f"Processing shard {shard_id + 1}/{num_shards}: {len(validation_set)} samples")

        print(f"Unique trial spaces: {validation_set.this_space.nunique()}")
        print(f"Unique patients: {validation_set.patient_summary.nunique()}")

        # Run inference if requested
        if run_inference and model_path:
            if categorical:
                validation_set = run_trial_checker_inference_categorical(
                    validation_set, model_path, device='cuda', batch_size=batch_size
                )
            else:
                validation_set = run_trial_checker_inference(
                    validation_set, model_path, device='cuda', batch_size=batch_size
                )
            # Save to shard file or main output
            if shard_dir_path and shard_id is not None:
                shard_dir_path.mkdir(parents=True, exist_ok=True)
                shard_path = shard_dir_path / f"shard_{shard_id}.csv"
                validation_set.to_csv(shard_path, index=False)
                print(f"Saved shard to: {shard_path}")
                return  # Exit after saving shard - merge step will do evaluation
            else:
                output_dir.mkdir(parents=True, exist_ok=True)
                intermediate_path = output_dir / f"patient_centric_with_predictions{suffix}.csv"
                validation_set.to_csv(intermediate_path, index=False)
                print(f"Saved predictions to: {intermediate_path}")
        elif 'prediction_score' not in validation_set.columns:
            # Try loading pre-computed predictions
            precomputed_path = output_dir / f"patient_centric_with_predictions{suffix}.csv"
            if precomputed_path.exists():
                print(f"Loading pre-computed predictions from: {precomputed_path}")
                validation_set = pd.read_csv(precomputed_path)
            else:
                print("No predictions available. Use --run-inference to generate them.")
                return

    output_dir.mkdir(parents=True, exist_ok=True)

    if categorical:
        # Categorical evaluation
        _evaluate_categorical_classification(validation_set, output_dir, "patient_centric")
        _evaluate_categorical_ranking(validation_set, output_dir, 'patient_summary',
                                      "patient_centric", k)
    else:
        # Original binary evaluation
        # Compute classification metrics (binarize graded labels for AUC)
        if 'prediction_score' in validation_set.columns and 'eligibility_result' in validation_set.columns:
            print("\n--- Classification Metrics ---")
            gold_binary = (validation_set.eligibility_result > 0).astype(float)
            auc = roc_auc_score(gold_binary, validation_set.prediction_score)
            print(f"AUC: {auc:.4f}")

            # Generate classification PDF report
            pdf_path = output_dir / "trial_checker_patient_centric_classification.pdf"
            eval_model(
                validation_set.prediction_score.values,
                gold_binary.values,
                pdf_path=str(pdf_path),
                title_prefix="Trial Checker Patient-Centric"
            )

            # Cohen's kappa
            if 'prediction_label' in validation_set.columns:
                actual_labels = np.where(
                    validation_set.eligibility_result == 0.0, 'NEGATIVE', 'POSITIVE'
                )
                kappa = cohen_kappa_score(actual_labels, validation_set.prediction_label)
                print(f"Cohen's Kappa: {kappa:.4f}")

                # Crosstab
                print("\nCrosstab (actual vs predicted):")
                print(pd.crosstab(validation_set.eligibility_result, validation_set.prediction_label))

        # Compute ranking metrics after filtering to positive predictions
        if 'prediction_label' in validation_set.columns:
            print("\n--- Ranking Metrics (after filtering to POSITIVE predictions) ---")

            # Take top k per patient first, then filter to positive
            pruned_set = validation_set.groupby('patient_summary').head(k)
            pruned_set = pruned_set[pruned_set.prediction_label == 'POSITIVE']

            print(f"Samples after filtering: {len(pruned_set)}")
            print(f"Median results per patient: {pruned_set.groupby('patient_summary').size().median():.1f}")
            print(f"Mean results per patient: {pruned_set.groupby('patient_summary').size().mean():.1f}")
            print(f"Unique patients: {pruned_set.patient_summary.nunique()}")
            print(f"Unique trial spaces: {pruned_set.this_space.nunique()}")

            if 'eligibility_result' in pruned_set.columns:
                # Calculate MAP@K on filtered set
                print(f"\nPositive rate after filtering: {pruned_set.eligibility_result.mean():.4f}")

                temp = pruned_set.groupby('patient_summary').eligibility_result.apply(
                    lambda x: average_precision_at_k(x.head(k).values)
                )
                map_k = temp.mean()
                print(f"MAP@{k} (after trial checker): {map_k:.4f}")

                # Generate ranking PDF report
                pdf_path = output_dir / "trial_checker_patient_centric_ranking.pdf"
                generate_ranking_report(
                    pruned_set,
                    group_col='patient_summary',
                    label_col='eligibility_result',
                    pdf_path=str(pdf_path),
                    title_prefix="Trial Checker Patient-Centric (Filtered)",
                    k=k
                )

    print(f"\nEvaluation complete. Reports saved to: {output_dir}")


def evaluate_trial_centric(data_dir: Path, output_dir: Path,
                            model_path: str = None, gpu: str = "0",
                            k: int = 20, run_inference: bool = False,
                            batch_size: int = 32, categorical: bool = False,
                            shard_id: int = None, num_shards: int = None,
                            shard_dir: str = None):
    """Evaluate trial-centric trial checker performance."""
    model_type = "CATEGORICAL" if categorical else "BINARY"
    print("=" * 60)
    print(f"EVALUATING TRIAL-CENTRIC TRIAL CHECKER ({model_type})")
    if shard_id is not None:
        print(f"(Shard {shard_id + 1}/{num_shards})")
    print("=" * 60)

    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    suffix = "_categorical" if categorical else ""

    # Handle merge-only mode (shard_dir specified but no shard_id)
    shard_dir_path = Path(shard_dir) if shard_dir else None
    if shard_dir_path and num_shards and shard_id is None:
        print("Merge mode: combining shards and running evaluation...")
        output_dir.mkdir(parents=True, exist_ok=True)
        intermediate_path = output_dir / f"trial_centric_with_predictions{suffix}.csv"
        validation_set = merge_shards(shard_dir_path, intermediate_path, num_shards)
        if validation_set is None:
            return
    else:
        # Load consolidated eligibility results from GPT checks
        consolidated_path = data_dir / "consolidated_eligibility_trial_centric.csv"
        if not consolidated_path.exists():
            print(f"Consolidated eligibility file not found: {consolidated_path}")
            return

        print(f"Loading consolidated file: {consolidated_path}")
        combined_df = pd.read_csv(consolidated_path)

        print(f"Loaded {len(combined_df)} rows")

        # Filter out null patient summaries
        combined_df = combined_df[~combined_df.patient_summary.isnull()]

        validation_set = combined_df.copy()

        # Apply sharding if specified
        if shard_id is not None and num_shards is not None:
            validation_set = validation_set.iloc[shard_id::num_shards].reset_index(drop=True)
            print(f"Processing shard {shard_id + 1}/{num_shards}: {len(validation_set)} samples")

        print(f"Unique trial spaces: {validation_set.this_space.nunique()}")
        print(f"Unique patients: {validation_set.patient_summary.nunique()}")

        # Run inference if requested
        if run_inference and model_path:
            if categorical:
                validation_set = run_trial_checker_inference_categorical(
                    validation_set, model_path, device='cuda', batch_size=batch_size
                )
            else:
                validation_set = run_trial_checker_inference(
                    validation_set, model_path, device='cuda', batch_size=batch_size
                )
            # Save to shard file or main output
            if shard_dir_path and shard_id is not None:
                shard_dir_path.mkdir(parents=True, exist_ok=True)
                shard_path = shard_dir_path / f"shard_{shard_id}.csv"
                validation_set.to_csv(shard_path, index=False)
                print(f"Saved shard to: {shard_path}")
                return  # Exit after saving shard - merge step will do evaluation
            else:
                output_dir.mkdir(parents=True, exist_ok=True)
                intermediate_path = output_dir / f"trial_centric_with_predictions{suffix}.csv"
                validation_set.to_csv(intermediate_path, index=False)
                print(f"Saved predictions to: {intermediate_path}")
        elif 'prediction_score' not in validation_set.columns:
            precomputed_path = output_dir / f"trial_centric_with_predictions{suffix}.csv"
            if precomputed_path.exists():
                print(f"Loading pre-computed predictions from: {precomputed_path}")
                validation_set = pd.read_csv(precomputed_path)
            else:
                print("No predictions available. Use --run-inference to generate them.")
                return

    output_dir.mkdir(parents=True, exist_ok=True)

    if categorical:
        # Categorical evaluation
        _evaluate_categorical_classification(validation_set, output_dir, "trial_centric")
        _evaluate_categorical_ranking(validation_set, output_dir, 'this_space',
                                      "trial_centric", k)
    else:
        # Original binary evaluation
        # Compute classification metrics (binarize graded labels for AUC)
        if 'prediction_score' in validation_set.columns and 'eligibility_result' in validation_set.columns:
            print("\n--- Classification Metrics ---")
            gold_binary = (validation_set.eligibility_result > 0).astype(float)
            auc = roc_auc_score(gold_binary, validation_set.prediction_score)
            print(f"AUC: {auc:.4f}")

            pdf_path = output_dir / "trial_checker_trial_centric_classification.pdf"
            eval_model(
                validation_set.prediction_score.values,
                gold_binary.values,
                pdf_path=str(pdf_path),
                title_prefix="Trial Checker Trial-Centric"
            )

            if 'prediction_label' in validation_set.columns:
                actual_labels = np.where(
                    validation_set.eligibility_result == 0.0, 'NEGATIVE', 'POSITIVE'
                )
                kappa = cohen_kappa_score(actual_labels, validation_set.prediction_label)
                print(f"Cohen's Kappa: {kappa:.4f}")

                print("\nCrosstab (actual vs predicted):")
                print(pd.crosstab(validation_set.eligibility_result, validation_set.prediction_label))

        # Compute ranking metrics (grouped by trial space instead of patient)
        if 'prediction_label' in validation_set.columns:
            print("\n--- Ranking Metrics (after filtering to POSITIVE predictions) ---")

            pruned_set = validation_set[validation_set.prediction_label == 'POSITIVE']

            print(f"Samples after filtering: {len(pruned_set)}")
            print(f"Median results per trial: {pruned_set.groupby('this_space').size().median():.1f}")
            print(f"Mean results per trial: {pruned_set.groupby('this_space').size().mean():.1f}")
            print(f"Unique patients: {pruned_set.patient_summary.nunique()}")
            print(f"Unique trial spaces: {pruned_set.this_space.nunique()}")

            if 'eligibility_result' in pruned_set.columns:
                print(f"\nPositive rate after filtering: {pruned_set.eligibility_result.mean():.4f}")

                # For trial-centric, group by trial space
                temp = pruned_set.groupby('this_space').eligibility_result.apply(
                    lambda x: average_precision_at_k(x.head(k).values)
                )
                map_k = temp.mean()
                print(f"MAP@{k} (after trial checker): {map_k:.4f}")

                pdf_path = output_dir / "trial_checker_trial_centric_ranking.pdf"
                generate_ranking_report(
                    pruned_set,
                    group_col='this_space',
                    label_col='eligibility_result',
                    pdf_path=str(pdf_path),
                    title_prefix="Trial Checker Trial-Centric (Filtered)",
                    k=k
                )

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
            categorical=args.categorical,
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
            categorical=args.categorical,
            shard_id=args.shard_id,
            num_shards=args.num_shards,
            shard_dir=args.shard_dir
        )


if __name__ == "__main__":
    main()
