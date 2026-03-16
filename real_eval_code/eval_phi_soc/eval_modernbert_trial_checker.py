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


def _compute_scenario_metrics(df: pd.DataFrame, group_col: str,
                               label_col: str, k: int) -> dict:
    """Compute ranking metrics for a pre-sorted, pre-filtered DataFrame."""
    top_k = df.groupby(group_col).head(k)
    if len(top_k) == 0:
        return {
            'map_at_k': 0.0, 'positive_rate': 0.0, 'num_groups': 0,
            'total_samples': 0, 'median_group_size': 0.0,
            'mean_group_size': 0.0, 'ap_scores': pd.Series(dtype=float),
        }
    ap_scores = top_k.groupby(group_col)[label_col].apply(
        lambda x: average_precision_at_k(x.values)
    )
    group_sizes = top_k.groupby(group_col).size()
    return {
        'map_at_k': ap_scores.mean(),
        'positive_rate': top_k[label_col].mean(),
        'num_groups': len(ap_scores),
        'total_samples': len(top_k),
        'median_group_size': group_sizes.median(),
        'mean_group_size': group_sizes.mean(),
        'ap_scores': ap_scores,
    }


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

    # Generate regression metrics PDF
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from datetime import datetime

    reg_pdf_path = output_dir / f"trial_checker_{mode_label}_regression_soc.pdf"
    with PdfPages(str(reg_pdf_path)) as pdf:
        # Page 1: Summary metrics
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('off')
        reg_text = f"""
SOC Trial Checker {mode_label.replace('_', ' ').title()} Regression Metrics
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 50}

Pearson r:    {r:.4f}  (p = {p_r:.4e})
Spearman rho: {rho:.4f}  (p = {p_rho:.4e})
MAE:          {mae:.4f}

Gold score range:      [{gold_scores.min():.2f}, {gold_scores.max():.2f}]
Predicted score range: [{pred_scores.min():.2f}, {pred_scores.max():.2f}]
N samples:             {len(gold_scores)}
"""
        ax.text(0.1, 0.9, reg_text, transform=ax.transAxes,
                fontsize=12, verticalalignment='top', fontfamily='monospace')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # Page 2: Scatter plot of gold vs predicted
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(gold_scores, pred_scores, alpha=0.15, s=10, edgecolors='none')
        score_min = min(gold_scores.min(), pred_scores.min())
        score_max = max(gold_scores.max(), pred_scores.max())
        ax.plot([score_min, score_max], [score_min, score_max], 'r--', label='y = x')
        ax.set_xlabel('Gold Score')
        ax.set_ylabel('Predicted Score')
        ax.set_title(f'SOC Trial Checker {mode_label.replace("_", " ").title()} Gold vs Predicted (r={r:.3f}, \u03c1={rho:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # Page 3: Residual distribution
        fig, ax = plt.subplots(figsize=(8, 6))
        residuals = pred_scores - gold_scores
        ax.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(0, color='r', linestyle='--', label='Zero error')
        ax.set_xlabel('Prediction Error (predicted - gold)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'SOC Trial Checker {mode_label.replace("_", " ").title()} Residual Distribution (MAE={mae:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    print(f"Regression PDF report saved to: {reg_pdf_path}")

    # --- Ranking Metrics: Three-Scenario Comparison ---
    print(f"\n--- Ranking Metrics (top-{k}, three scenarios) ---")

    base_df = validation_set.copy()
    base_df['_binary_label'] = (base_df.eligibility_result > 0).astype(float)
    base_df['_original_rank'] = base_df.groupby(group_col).cumcount()

    # Scenario A: Without TrialChecker (cosine similarity order only)
    scenario_a_df = base_df.sort_values(by=[group_col, '_original_rank'])
    scenario_a_stats = _compute_scenario_metrics(scenario_a_df, group_col, '_binary_label', k)

    # Scenario B: TrialChecker as hard filter (>= 1.0), keep cosine similarity order
    scenario_b_df = base_df[base_df.prediction_score >= 1.0].copy()
    scenario_b_df = scenario_b_df.sort_values(by=[group_col, '_original_rank'])
    scenario_b_stats = _compute_scenario_metrics(scenario_b_df, group_col, '_binary_label', k)

    # Scenario C: TrialChecker as re-ranker (>= 1.0 filter, rank by prediction_score)
    scenario_c_df = base_df[base_df.prediction_score >= 1.0].copy()
    scenario_c_df = scenario_c_df.sort_values(
        by=[group_col, 'prediction_score'], ascending=[True, False]
    )
    scenario_c_stats = _compute_scenario_metrics(scenario_c_df, group_col, '_binary_label', k)

    scenarios = [
        ('A: No TrialChecker (cosine sim)', scenario_a_stats),
        ('B: TrialChecker Filter (>= 1)', scenario_b_stats),
        ('C: TrialChecker Re-Ranker (>= 1)', scenario_c_stats),
    ]

    kstr = f'MAP@{k}'
    print(f"\n{'Scenario':<38} {kstr:<10} {'PosRate':<10} {'Groups':<8} {'Samples':<10} {'MedSz':<8} {'MeanSz':<8}")
    print("-" * 92)
    for name, stats in scenarios:
        print(f"{name:<38} {stats['map_at_k']:<10.4f} {stats['positive_rate']:<10.4f} "
              f"{stats['num_groups']:<8} {stats['total_samples']:<10} "
              f"{stats['median_group_size']:<8.1f} {stats['mean_group_size']:<8.1f}")

    # Generate comparison ranking PDF
    title_base = f"SOC Trial Checker {mode_label.replace('_', ' ').title()}"
    pdf_path = output_dir / f"trial_checker_{mode_label}_ranking_soc.pdf"

    with PdfPages(str(pdf_path)) as pdf:
        # Page 1: Summary comparison table
        fig, ax = plt.subplots(figsize=(11, 8))
        ax.axis('off')
        summary = f"""{title_base} Ranking Scenario Comparison
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
K = {k}
{'=' * 80}

{'Scenario':<42} {kstr:<9} {'PosRate':<9} {'Groups':<8} {'Samples':<9} {'MedSz':<8} {'MeanSz':<8}
{'-' * 93}
"""
        for name, stats in scenarios:
            summary += (f"{name:<42} {stats['map_at_k']:<9.4f} {stats['positive_rate']:<9.4f} "
                        f"{stats['num_groups']:<8} {stats['total_samples']:<9} "
                        f"{stats['median_group_size']:<8.1f} {stats['mean_group_size']:<8.1f}\n")
        ax.text(0.05, 0.95, summary, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # Page 2: Overlaid AP distribution histograms
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for (name, stats), color in zip(scenarios, colors):
            ap = stats['ap_scores']
            if len(ap) > 0:
                ax.hist(ap, bins=20, alpha=0.4, color=color, edgecolor=color,
                        label=f"{name} (MAP@{k}={stats['map_at_k']:.4f})")
        ax.set_xlabel('Average Precision')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{title_base} AP@{k} Distribution by Scenario')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # Page 3: Side-by-side individual AP histograms
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
        for ax_i, ((name, stats), color) in enumerate(zip(scenarios, colors)):
            ap = stats['ap_scores']
            if len(ap) > 0:
                axes[ax_i].hist(ap, bins=20, alpha=0.7, color=color, edgecolor='black')
                axes[ax_i].axvline(stats['map_at_k'], color='r', linestyle='--',
                                   label=f"MAP@{k}={stats['map_at_k']:.4f}")
            axes[ax_i].set_xlabel('Average Precision')
            if ax_i == 0:
                axes[ax_i].set_ylabel('Frequency')
            axes[ax_i].set_title(name, fontsize=9)
            axes[ax_i].legend(fontsize=8)
            axes[ax_i].grid(True, alpha=0.3)
        fig.suptitle(f'{title_base} AP@{k} Distributions', fontsize=12)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    print(f"Ranking comparison report saved to: {pdf_path}")


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
