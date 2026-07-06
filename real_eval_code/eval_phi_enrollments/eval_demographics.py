#!/usr/bin/env python3
"""
Demographic-stratified evaluation of the trial-matching models.

Re-runs the headline metrics that the existing eval scripts already compute, but
broken down by patient demographics (age category at treatment start, race,
ethnicity, sex):

  - baseline retrieval     -> MAP@K (ranking)
  - LLM trial checker      -> AUC (classification) + MAP@K (ranking, post-filter)
  - LLM boilerplate checker-> AUC (classification)

Demographics are joined from the pan-DFCI structured-data registration tables
(see demographics_utils.py). All metric math is reused from eval_utils.py and the
per-eval candidate frames are reconstructed with the existing loaders in
eval_llm_trial_checker.py / eval_llm_boilerplate_checker.py.

Outputs (per mode):
  - demographics_metrics_{mode}.csv  : tidy long table, one row per
                                       eval x demographic x category
  - demographics_metrics_{mode}.pdf  : bar charts (AUC / MAP@K) per demographic

Usage:
    python eval_demographics.py --mode patient_centric \
        --data-dir /path/to/data --output-dir /path/to/output
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import roc_auc_score

from eval_utils import (
    bootstrap_metric_ci,
    binary_auroc_score,
    positive_rate_metric,
    calculate_map_at_k,
    mae_metric,
    rmse_metric,
    r2_metric,
    bias_metric,
    pearson_metric,
    spearman_metric,
)
from demographics_utils import (
    DEFAULT_STRUCTURED_DATA_DIR,
    DEMOGRAPHIC_COLS,
    AGE_LABELS,
    UNKNOWN,
    load_demographics,
    load_summary_dates,
    assign_demographics,
)

# Reuse the existing per-eval candidate-frame builders.
import eval_llm_trial_checker as trial_checker
import eval_llm_boilerplate_checker as boilerplate_checker

# Minimum samples required before a stratified metric is reported.
MIN_SAMPLES = 20
# Bootstrap resamples for every CI, matching eval_utils' default so the overall
# (non-stratified) rows line up exactly with the standard eval reports.
BOOTSTRAP_N = 1000

OVERALL_DEMO = 'overall'
OVERALL_CAT = 'all'


def parse_args():
    parser = argparse.ArgumentParser(
        description="Demographic-stratified evaluation of trial-matching models"
    )
    parser.add_argument("--mode", type=str, required=True,
                        choices=["patient_centric", "trial_centric"],
                        help="Evaluation mode")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory containing candidate / result / summary files")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save evaluation outputs")
    parser.add_argument("--eval-dir", type=str, default=None,
                        help="Directory with existing model eval outputs "
                             "(ModernBERT *_with_predictions CSVs); "
                             "defaults to <data-dir>/evaluation")
    parser.add_argument("--structured-data-dir", type=str,
                        default=DEFAULT_STRUCTURED_DATA_DIR,
                        help="Directory with the pan-DFCI registration tables")
    parser.add_argument("--k", type=int, default=None,
                        help="K value for MAP@K calculation (default: 20)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Score threshold for ranking filter (default: 0.5)")
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Per-eval candidate-frame builders
# --------------------------------------------------------------------------- #

def build_baseline_frame(data_dir: Path, mode: str):
    """Baseline ranking frame: the consolidated eligibility file in rank order."""
    path = data_dir / f"consolidated_eligibility_{mode}.csv"
    if not path.exists():
        print(f"[baseline] missing {path}; skipping")
        return None
    df = pd.read_csv(path)
    df = df[~df.patient_summary.isnull()]
    # Drop gold PARSE_FAILED sentinels (eligibility_result == -1).
    df = df[df.eligibility_result >= 0].reset_index(drop=True)
    return df


def build_trial_checker_frame(data_dir: Path, mode: str):
    """Merged predictions+gold frame for the LLM trial checker (rank order)."""
    results = data_dir / f"oncoreasoning_trialcheck_{mode}.csv"
    gold_path = data_dir / f"consolidated_eligibility_{mode}.csv"
    if not results.exists() or not gold_path.exists():
        print(f"[trial_checker] missing inputs near {results}; skipping")
        return None
    predictions = trial_checker.load_predictions(results)
    gold = trial_checker.load_gold(gold_path)
    merged = trial_checker.align_predictions_to_gold(predictions, gold)
    if len(merged) == 0:
        return None
    return merged


def build_boilerplate_frame(data_dir: Path, mode: str):
    """Merged predictions+gold frame for the LLM boilerplate checker."""
    results = data_dir / f"oncoreasoning_boilerplate_{mode}.csv"
    candidates = data_dir / f"{mode.split('_')[0]}_centric_candidates.csv"
    gold_path = data_dir / f"consolidated_boilerplate_{mode}.csv"
    if not results.exists() or not gold_path.exists():
        print(f"[boilerplate] missing inputs near {results}; skipping")
        return None
    predictions = boilerplate_checker.load_predictions(results, candidates)
    gold = boilerplate_checker.load_gold(gold_path)
    merged = gold.merge(predictions, on=boilerplate_checker.KEY_COLS, how='inner')
    if len(merged) == 0:
        return None
    return merged


def _find_predictions_csv(eval_dir: Path, subdir: str, stem: str, mode: str):
    """Locate a ModernBERT *_with_predictions CSV, tolerating the _soc suffix."""
    prefix = mode.split('_')[0]  # 'patient' or 'trial'
    pattern = str(eval_dir / subdir / f"{stem}{prefix}_centric_with_predictions*.csv")
    matches = sorted(glob.glob(pattern))
    return Path(matches[0]) if matches else None


def build_modernbert_trial_frame(eval_dir: Path, mode: str):
    """ModernBERT trial-checker predictions+gold frame (in retrieval order)."""
    path = _find_predictions_csv(eval_dir, "modernbert-trial-checker", "", mode)
    if path is None:
        print(f"[modernbert_trial_checker] no with_predictions CSV under "
              f"{eval_dir / 'modernbert-trial-checker'}; skipping")
        return None
    df = pd.read_csv(path, usecols=lambda c: c in {
        'dfci_mrn', 'patient_summary', 'this_space',
        'eligibility_result', 'prediction_score'})
    df = df[~df.patient_summary.isnull()]
    # Drop gold PARSE_FAILED sentinels (eligibility_result == -1).
    df = df[df.eligibility_result >= 0].reset_index(drop=True)
    df['prediction_score'] = pd.to_numeric(df['prediction_score'], errors='coerce')
    return df


def build_modernbert_boilerplate_frame(eval_dir: Path, mode: str):
    """ModernBERT boilerplate-checker predictions+gold frame."""
    path = _find_predictions_csv(eval_dir, "modernbert-boilerplate-checker",
                                 "boilerplate_", mode)
    if path is None:
        print(f"[modernbert_boilerplate] no with_predictions CSV under "
              f"{eval_dir / 'modernbert-boilerplate-checker'}; skipping")
        return None
    df = pd.read_csv(path, usecols=lambda c: c in {
        'dfci_mrn', 'patient_summary', 'this_space',
        'exclusion_result', 'prediction_score'})
    df = df[~df.patient_summary.isnull()]
    # Drop gold PARSE_FAILED sentinels (exclusion_result == -1).
    df = df[df.exclusion_result >= 0].reset_index(drop=True)
    df['prediction_score'] = pd.to_numeric(df['prediction_score'], errors='coerce')
    return df


# --------------------------------------------------------------------------- #
# Stratified metric computation
# --------------------------------------------------------------------------- #

def _ci_bounds(ci):
    if ci is None:
        return (np.nan, np.nan)
    return (ci.get('lower', np.nan), ci.get('upper', np.nan))


def _auc_for(sub: pd.DataFrame, score_col: str, gold_col: str):
    """AUC + CI for a subset, or (nan, nan, nan) when undefined."""
    if sub is None or len(sub) < MIN_SAMPLES:
        return np.nan, np.nan, np.nan
    gold = sub[gold_col].values.astype(float)
    score = sub[score_col].values.astype(float)
    if len(np.unique(gold)) < 2:
        return np.nan, np.nan, np.nan
    try:
        auc = roc_auc_score(gold, score)
    except ValueError:
        return np.nan, np.nan, np.nan
    lo, hi = _ci_bounds(bootstrap_metric_ci(
        [gold, score], binary_auroc_score, n_bootstrap=BOOTSTRAP_N))
    return auc, lo, hi


def _map_for(sub: pd.DataFrame, group_col: str, label_col: str, k: int):
    """MAP@K + CI + group count for a subset, or nans when undefined."""
    if sub is None or len(sub) < MIN_SAMPLES:
        return np.nan, np.nan, np.nan, (0 if sub is None else sub[group_col].nunique())
    map_k, stats = calculate_map_at_k(sub, group_col, label_col, k, n_bootstrap=BOOTSTRAP_N)
    lo, hi = _ci_bounds(stats.get('map_at_k_ci'))
    return map_k, lo, hi, stats['num_groups']


# Continuous regression metrics reported per stratum (gold vs continuous prediction).
_REGRESSION_METRICS = {
    'mae': mae_metric,
    'rmse': rmse_metric,
    'r2': r2_metric,
    'bias': bias_metric,
    'pearson': pearson_metric,
    'spearman': spearman_metric,
}


def _regression_for(sub, gold_col, pred_col):
    """Continuous regression metrics + CIs for a subset (gold vs prediction).

    Returns a flat dict {<metric>, <metric>_ci_low, <metric>_ci_high} for every
    entry in _REGRESSION_METRICS; values are nan when the subset is too small or
    degenerate (e.g. constant gold makes correlation/R^2 undefined).
    """
    out = {}
    for name in _REGRESSION_METRICS:
        out[name] = np.nan
        out[f'{name}_ci_low'] = np.nan
        out[f'{name}_ci_high'] = np.nan
    if sub is None or len(sub) < MIN_SAMPLES:
        return out
    gold = sub[gold_col].values.astype(float)
    pred = sub[pred_col].values.astype(float)
    mask = np.isfinite(gold) & np.isfinite(pred)
    gold, pred = gold[mask], pred[mask]
    if len(gold) < MIN_SAMPLES:
        return out
    for name, fn in _REGRESSION_METRICS.items():
        try:
            val = float(fn(gold, pred))
        except Exception:
            val = np.nan
        lo, hi = _ci_bounds(bootstrap_metric_ci([gold, pred], fn, n_bootstrap=BOOTSTRAP_N))
        out[name] = val
        out[f'{name}_ci_low'] = lo
        out[f'{name}_ci_high'] = hi
    return out


def _positive_rate(clf_sub, rank_sub, clf_gold_col, rank_label_col):
    """Positive rate + CI, preferring the classification gold when available."""
    if clf_sub is not None and len(clf_sub) > 0:
        vals = (clf_sub[clf_gold_col].values > 0).astype(float)
    elif rank_sub is not None and len(rank_sub) > 0:
        vals = (rank_sub[rank_label_col].values > 0).astype(float)
    else:
        return np.nan, np.nan, np.nan
    rate = positive_rate_metric(vals)
    lo, hi = _ci_bounds(bootstrap_metric_ci(
        [vals], positive_rate_metric, n_bootstrap=BOOTSTRAP_N))
    return rate, lo, hi


def _order_categories(dim, values):
    """Stable display order: defined age order; else alpha with Unknown last."""
    values = list(values)
    if dim == 'age_category':
        order = [c for c in AGE_LABELS if c in values] + (
            [UNKNOWN] if UNKNOWN in values else [])
        return order
    rest = sorted(v for v in values if v != UNKNOWN)
    return rest + ([UNKNOWN] if UNKNOWN in values else [])


def stratify_eval(name, mode, k, *, clf_frame=None, rank_frame=None,
                  score_col=None, clf_gold_col=None,
                  group_col=None, rank_label_col=None,
                  reg_gold_col=None, reg_pred_col=None):
    """Compute per-demographic metric rows for one eval.

    clf_frame: frame for classification (AUC); rank_frame: frame for ranking
    (MAP@K). Either may be None. Both carry the demographic columns. When
    reg_gold_col/reg_pred_col are given, continuous regression metrics
    (MAE/RMSE/R^2/bias/Pearson/Spearman) are also computed on clf_frame subsets.
    """
    rows = []

    def emit(demographic, category, clf_sub, rank_sub):
        auc, auc_lo, auc_hi = (np.nan, np.nan, np.nan)
        if clf_frame is not None:
            auc, auc_lo, auc_hi = _auc_for(clf_sub, score_col, clf_gold_col)
        map_k = map_lo = map_hi = np.nan
        n_groups = np.nan
        if rank_frame is not None:
            map_k, map_lo, map_hi, n_groups = _map_for(
                rank_sub, group_col, rank_label_col, k)
        pos, pos_lo, pos_hi = _positive_rate(
            clf_sub, rank_sub, clf_gold_col, rank_label_col)
        row = {
            'eval': name, 'mode': mode,
            'demographic': demographic, 'category': category,
            'n_classification': (np.nan if clf_frame is None
                                 else (0 if clf_sub is None else len(clf_sub))),
            'n_ranking': (np.nan if rank_frame is None
                          else (0 if rank_sub is None else len(rank_sub))),
            'n_groups': n_groups,
            'positive_rate': pos, 'positive_rate_ci_low': pos_lo,
            'positive_rate_ci_high': pos_hi,
            'auc': auc, 'auc_ci_low': auc_lo, 'auc_ci_high': auc_hi,
            'map_at_k': map_k, 'map_at_k_ci_low': map_lo,
            'map_at_k_ci_high': map_hi, 'k': k,
        }
        if reg_gold_col is not None and reg_pred_col is not None:
            row.update(_regression_for(clf_sub, reg_gold_col, reg_pred_col))
        rows.append(row)

    # Overall row first.
    emit(OVERALL_DEMO, OVERALL_CAT, clf_frame, rank_frame)

    # Per-demographic categories.
    ref_frame = clf_frame if clf_frame is not None else rank_frame
    for dim in DEMOGRAPHIC_COLS:
        cats = _order_categories(dim, ref_frame[dim].dropna().unique())
        for cat in cats:
            clf_sub = None if clf_frame is None else clf_frame[clf_frame[dim] == cat]
            rank_sub = None if rank_frame is None else rank_frame[rank_frame[dim] == cat]
            emit(dim, cat, clf_sub, rank_sub)

    return rows


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def _bar_page(pdf, results_df, eval_name, metric, value_col, lo_col, hi_col, title):
    """One PDF page: grouped bar charts of `metric` across each demographic."""
    if value_col not in results_df.columns:
        return
    sub = results_df[(results_df['eval'] == eval_name) &
                     (results_df.demographic != OVERALL_DEMO)]
    sub = sub[np.isfinite(sub[value_col])]
    if sub.empty:
        return
    dims = [d for d in DEMOGRAPHIC_COLS if d in set(sub.demographic)]
    if not dims:
        return
    fig, axes = plt.subplots(len(dims), 1, figsize=(9, 3.2 * len(dims)))
    if len(dims) == 1:
        axes = [axes]
    overall = results_df[(results_df['eval'] == eval_name) &
                         (results_df.demographic == OVERALL_DEMO)]
    overall_val = overall[value_col].iloc[0] if len(overall) else np.nan
    for ax, dim in zip(axes, dims):
        d = sub[sub.demographic == dim]
        cats = _order_categories(dim, d.category.unique())
        d = d.set_index('category').reindex(cats)
        vals = d[value_col].values
        err = np.vstack([
            np.clip(d[value_col].values - d[lo_col].values, 0, None),
            np.clip(d[hi_col].values - d[value_col].values, 0, None),
        ])
        x = np.arange(len(cats))
        ax.bar(x, vals, yerr=err, capsize=3, alpha=0.8, edgecolor='black')
        if np.isfinite(overall_val):
            ax.axhline(overall_val, color='r', linestyle='--', alpha=0.7,
                       label=f'overall = {overall_val:.3f}')
            ax.legend(fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(cats, rotation=30, ha='right', fontsize=8)
        ax.set_ylabel(metric)
        ax.set_title(f'{dim}')
        ax.grid(True, axis='y', alpha=0.3)
    fig.suptitle(title)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def write_pdf(results_df, pdf_path, mode):
    with PdfPages(str(pdf_path)) as pdf:
        for eval_name in results_df['eval'].unique():
            _bar_page(pdf, results_df, eval_name, 'AUC',
                      'auc', 'auc_ci_low', 'auc_ci_high',
                      f'{eval_name} ({mode}) - AUC by demographic')
            _bar_page(pdf, results_df, eval_name, f'MAP@K',
                      'map_at_k', 'map_at_k_ci_low', 'map_at_k_ci_high',
                      f'{eval_name} ({mode}) - MAP@K by demographic')
            # Continuous regression metrics (only populated for the trial checkers).
            for _mkey, _mlabel in [('mae', 'MAE'), ('rmse', 'RMSE'),
                                   ('r2', 'R^2 (CoD)'), ('bias', 'Bias (pred-gold)')]:
                _bar_page(pdf, results_df, eval_name, _mlabel,
                          _mkey, f'{_mkey}_ci_low', f'{_mkey}_ci_high',
                          f'{eval_name} ({mode}) - {_mlabel} by demographic')
    print(f"PDF report saved to: {pdf_path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    eval_dir = Path(args.eval_dir) if args.eval_dir else data_dir / "evaluation"
    mode = args.mode
    # Ranking depth matches retrieval depth: 20 patient-centric, 40 trial-centric.
    k = args.k if args.k is not None else (40 if mode == "trial_centric" else 20)
    group_col = 'patient_summary' if mode == 'patient_centric' else 'this_space'

    print("=" * 60)
    print(f"DEMOGRAPHIC-STRATIFIED EVALUATION ({mode})")
    print("=" * 60)

    demo = load_demographics(args.structured_data_dir)
    summary_dates = load_summary_dates(str(data_dir / "patient_summaries.parquet"))
    print(f"Loaded demographics for {len(demo)} patients; "
          f"treatment-start dates for {len(summary_dates)} summaries")

    def with_demo(frame):
        return assign_demographics(frame, demo, summary_dates)

    all_rows = []

    baseline = build_baseline_frame(data_dir, mode)
    if baseline is not None:
        all_rows += stratify_eval(
            'baseline', mode, k,
            rank_frame=with_demo(baseline),
            group_col=group_col, rank_label_col='eligibility_result')

    tc = build_trial_checker_frame(data_dir, mode)
    if tc is not None:
        tc = with_demo(tc)
        tc['_gold_bin'] = (tc['eligibility_result'] > 0).astype(float)
        ranked = tc[tc['prediction'] >= args.threshold].copy()
        all_rows += stratify_eval(
            'trial_checker', mode, k,
            clf_frame=tc, score_col='prediction', clf_gold_col='_gold_bin',
            rank_frame=ranked, group_col=group_col,
            rank_label_col='eligibility_result',
            reg_gold_col='eligibility_result', reg_pred_col='prediction')

    bp = build_boilerplate_frame(data_dir, mode)
    if bp is not None:
        bp = with_demo(bp)
        all_rows += stratify_eval(
            'boilerplate', mode, k,
            clf_frame=bp, score_col='prediction', clf_gold_col='exclusion_result')

    mb_tc = build_modernbert_trial_frame(eval_dir, mode)
    if mb_tc is not None:
        mb_tc = with_demo(mb_tc)
        mb_tc['_gold_bin'] = (mb_tc['eligibility_result'] > 0).astype(float)
        mb_ranked = mb_tc[mb_tc['prediction_score'] >= args.threshold].copy()
        all_rows += stratify_eval(
            'modernbert_trial_checker', mode, k,
            clf_frame=mb_tc, score_col='prediction_score', clf_gold_col='_gold_bin',
            rank_frame=mb_ranked, group_col=group_col,
            rank_label_col='eligibility_result',
            reg_gold_col='eligibility_result', reg_pred_col='prediction_score')

    mb_bp = build_modernbert_boilerplate_frame(eval_dir, mode)
    if mb_bp is not None:
        mb_bp = with_demo(mb_bp)
        all_rows += stratify_eval(
            'modernbert_boilerplate', mode, k,
            clf_frame=mb_bp, score_col='prediction_score',
            clf_gold_col='exclusion_result')

    if not all_rows:
        print("No evaluable inputs found; nothing written.")
        return

    results_df = pd.DataFrame(all_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"demographics_metrics_{mode}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nMetrics table saved to: {csv_path}  ({len(results_df)} rows)")

    write_pdf(results_df, output_dir / f"demographics_metrics_{mode}.pdf", mode)
    print(f"\nEvaluation complete. Reports saved to: {output_dir}")


if __name__ == "__main__":
    main()
