#!/usr/bin/env python3
"""
Shared evaluation utilities for clinical trial matching model evaluation.

This module provides functions to:
- Calculate evaluation metrics (AUC, F1, precision-recall, etc.)
- Generate PDF reports with visualizations
- Compute MAP@K metrics for ranking evaluation
"""

import itertools
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for PDF generation
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import (
    roc_auc_score, f1_score, classification_report,
    precision_recall_curve, auc, roc_curve, confusion_matrix,
    average_precision_score, cohen_kappa_score, r2_score
)
from sklearn.calibration import calibration_curve
from scipy.stats import spearmanr, pearsonr
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, Callable, Sequence
import pandas as pd


DEFAULT_BOOTSTRAP_SAMPLES = 1000
DEFAULT_BOOTSTRAP_RANDOM_STATE = 0
DEFAULT_CONFIDENCE_LEVEL = 0.95


def sigmoid(x):
    """Apply sigmoid function to input."""
    return 1. / (1. + np.exp(-x))


def _finite_float(value: Any) -> Optional[float]:
    """Return a finite float or None when the metric cannot be used."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def _ci_from_samples(samples: Sequence[float],
                     confidence_level: float = DEFAULT_CONFIDENCE_LEVEL) -> Optional[Dict[str, Any]]:
    """Build a percentile confidence interval from bootstrap samples."""
    values = np.asarray(samples, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None

    alpha = 1.0 - confidence_level
    lower, upper = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        'lower': float(lower),
        'upper': float(upper),
        'confidence_level': confidence_level,
        'n_bootstrap': int(len(values)),
    }


def bootstrap_metric_ci(arrays: Sequence[np.ndarray],
                        metric_fn: Callable[..., float],
                        n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
                        confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
                        random_state: int = DEFAULT_BOOTSTRAP_RANDOM_STATE) -> Optional[Dict[str, Any]]:
    """
    Calculate a percentile bootstrap confidence interval for row-aligned arrays.

    Invalid bootstrap samples (for example, AUROC samples with one class) are
    skipped instead of failing the whole evaluation.
    """
    prepared = [np.asarray(array) for array in arrays]
    if not prepared:
        return None
    n = len(prepared[0])
    if n == 0 or any(len(array) != n for array in prepared):
        return None

    rng = np.random.default_rng(random_state)
    samples = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                value = metric_fn(*[array[indices] for array in prepared])
        except Exception:
            continue
        value = _finite_float(value)
        if value is not None:
            samples.append(value)

    return _ci_from_samples(samples, confidence_level)


def bootstrap_mean_ci(values: Sequence[float],
                      n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
                      confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
                      random_state: int = DEFAULT_BOOTSTRAP_RANDOM_STATE) -> Optional[Dict[str, Any]]:
    """Calculate a percentile bootstrap CI for a mean statistic."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None

    rng = np.random.default_rng(random_state)
    samples = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(values), size=len(values))
        samples.append(float(np.mean(values[indices])))

    return _ci_from_samples(samples, confidence_level)


def format_metric_with_ci(value: Any,
                          ci: Optional[Dict[str, Any]],
                          precision: int = 4) -> str:
    """Format a metric value with a 95% CI suffix."""
    value = _finite_float(value)
    if value is None:
        return "N/A"
    if ci is None:
        return f"{value:.{precision}f} (95% CI N/A)"
    level = int(round(float(ci.get('confidence_level', DEFAULT_CONFIDENCE_LEVEL)) * 100))
    lower = ci.get('lower')
    upper = ci.get('upper')
    if _finite_float(lower) is None or _finite_float(upper) is None:
        return f"{value:.{precision}f} ({level}% CI N/A)"
    return f"{value:.{precision}f} ({level}% CI [{lower:.{precision}f}, {upper:.{precision}f}])"


def binary_auroc_score(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate binary AUROC."""
    return roc_auc_score(actual, predicted)


def average_precision_metric(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate average precision/AUPRC."""
    return average_precision_score(actual, predicted)


def positive_rate_metric(actual: np.ndarray) -> float:
    """Calculate the fraction of positive labels."""
    return float(np.mean(actual))


def best_f1_metric(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate the best F1 score over precision-recall thresholds."""
    precision, recall, _ = precision_recall_curve(actual, predicted)
    f1 = 2 * ((precision * recall) / (precision + recall + 1e-10))
    return float(np.max(f1))


def pearson_metric(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Pearson correlation."""
    return float(pearsonr(actual, predicted)[0])


def spearman_metric(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Spearman correlation."""
    return float(spearmanr(actual, predicted)[0])


def mae_metric(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate mean absolute error."""
    return float(np.mean(np.abs(actual - predicted)))


def rmse_metric(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root mean squared error (same units as the target; penalizes large errors)."""
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def r2_metric(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Coefficient of determination (R^2) against the identity line.

    sklearn r2_score = 1 - SS_res / SS_tot, which penalizes bias and scale error.
    This is distinct from Pearson r squared (best-fit line) and can be negative
    when predictions are worse than always predicting the mean.
    """
    return float(r2_score(actual, predicted))


def bias_metric(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean signed error (predicted - actual); positive => systematic over-prediction."""
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    return float(np.mean(predicted - actual))


def average_precision_at_k(label_array: np.ndarray) -> float:
    """
    Calculate average precision for a ranked list.

    Args:
        label_array: Binary array of labels (1=relevant, 0=not relevant) in ranked order

    Returns:
        Average precision score
    """
    total_yes = np.sum(label_array > 0)
    if total_yes > 0:
        yes_indices = np.where(label_array > 0)[0] + 1
        precisions = []
        for index in yes_indices:
            precision = np.sum(label_array[0:index] > 0) / index
            precisions.append(precision)
        return np.sum(np.array(precisions)) / total_yes
    else:
        return 0


def plot_confusion_matrix(cm: np.ndarray, classes: list,
                          normalize: bool = False,
                          title: str = 'Confusion matrix',
                          cmap=plt.cm.Blues,
                          ax=None) -> None:
    """
    Plot a confusion matrix.

    Args:
        cm: Confusion matrix array
        classes: List of class names
        normalize: Whether to normalize the matrix
        title: Plot title
        cmap: Colormap to use
        ax: Matplotlib axes (optional)
    """
    if ax is None:
        ax = plt.gca()

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.set_title(title)
    ax.grid(False)


def eval_model(predicted: np.ndarray, actual: np.ndarray,
               pdf_path: Optional[str] = None,
               title_prefix: str = "",
               gold_continuous: Optional[np.ndarray] = None,
               n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
               random_state: int = DEFAULT_BOOTSTRAP_RANDOM_STATE) -> Optional[float]:
    """
    Evaluate model predictions and optionally save results to PDF.

    Args:
        predicted: Predicted scores/probabilities
        actual: Actual binary labels
        pdf_path: Path to save PDF report (if None, just returns metrics)
        title_prefix: Prefix for plot titles
        gold_continuous: Optional continuous gold-standard scores for regression metrics

    Returns:
        Best F1 threshold or None if calculation fails
    """
    predicted = np.asarray(predicted)
    actual = np.asarray(actual)
    outcome_counts = np.unique(actual, return_counts=True)[1]

    try:
        prob_outcome = outcome_counts[1] / (outcome_counts[0] + outcome_counts[1])
        auc_score = roc_auc_score(actual, predicted)
        auc_ci = bootstrap_metric_ci(
            [actual, predicted], binary_auroc_score,
            n_bootstrap=n_bootstrap, random_state=random_state
        )
        prob_outcome_ci = bootstrap_metric_ci(
            [actual], positive_rate_metric,
            n_bootstrap=n_bootstrap, random_state=random_state
        )

        # Calculate ROC curve
        fpr, tpr, threshold = roc_curve(actual, predicted)
        roc_auc = auc(fpr, tpr)

        # Calculate precision-recall
        avg_precision = average_precision_score(actual, predicted)
        avg_precision_ci = bootstrap_metric_ci(
            [actual, predicted], average_precision_metric,
            n_bootstrap=n_bootstrap, random_state=random_state
        )
        precision, recall, thresholds = precision_recall_curve(actual, predicted)

        # Best F1
        F1 = 2 * ((precision * recall) / (precision + recall + 1e-10))
        best_f1 = max(F1)
        best_f1_ci = bootstrap_metric_ci(
            [actual, predicted], best_f1_metric,
            n_bootstrap=n_bootstrap, random_state=random_state
        )
        best_f1_thresh = thresholds[np.argmax(F1)] if len(thresholds) > 0 else 0.5

        # Print metrics summary
        print(f"AUC: {format_metric_with_ci(auc_score, auc_ci)}")
        print(f"Outcome probability: {format_metric_with_ci(prob_outcome, prob_outcome_ci)}")
        print(f"Average precision score: {format_metric_with_ci(avg_precision, avg_precision_ci)}")
        print(f"Best F1: {format_metric_with_ci(best_f1, best_f1_ci)}")
        print(f"Best F1 threshold: {best_f1_thresh:.4f}")

        if pdf_path is None:
            return best_f1_thresh

        # Generate PDF report
        with PdfPages(pdf_path) as pdf:
            # Page 1: Summary metrics
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.axis('off')
            summary_text = f"""
{title_prefix} Evaluation Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY METRICS
===============
Total samples: {len(actual)}
Positive samples: {int(sum(actual))} ({prob_outcome:.2%})
Negative samples: {int(len(actual) - sum(actual))} ({1-prob_outcome:.2%})

AUC-ROC: {format_metric_with_ci(auc_score, auc_ci)}
Average Precision: {format_metric_with_ci(avg_precision, avg_precision_ci)}
Best F1 Score: {format_metric_with_ci(best_f1, best_f1_ci)}
Best F1 Threshold: {best_f1_thresh:.4f}
"""
            ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
                    fontsize=12, verticalalignment='top', fontfamily='monospace')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            # Page 2: ROC Curve
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_title(f'{title_prefix} ROC Curve')
            ax.plot(fpr, tpr, 'b', label=f'AUC = {roc_auc:.4f}')
            ax.legend(loc='lower right')
            ax.plot([0, 1], [0, 1], 'r--')
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_ylabel('True Positive Rate')
            ax.set_xlabel('False Positive Rate')
            ax.grid(True, alpha=0.3)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            # Page 3: Precision-Recall Curve
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(recall, precision, color='b')
            ax.plot([0, 1], [prob_outcome, prob_outcome], 'r--', label='Random baseline')
            ax.fill_between(recall, precision, alpha=0.2, color='b')
            ax.set_xlabel('Recall (Sensitivity)')
            ax.set_ylabel('Precision (PPV)')
            ax.set_ylim([0.0, 1.05])
            ax.set_xlim([0.0, 1.0])
            ax.set_title(f'{title_prefix} Precision-Recall Curve (AP={avg_precision:.4f})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            # Page 4: Confusion matrices
            pred_best_f1 = np.where(predicted >= best_f1_thresh, 1, 0)
            pred_05 = np.where(predicted >= 0.5, 1, 0)

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            cnf_matrix_best = confusion_matrix(actual, pred_best_f1)
            plot_confusion_matrix(cnf_matrix_best, classes=['No', 'Yes'],
                                  title=f'Best F1 Threshold ({best_f1_thresh:.3f})',
                                  ax=axes[0])

            cnf_matrix_05 = confusion_matrix(actual, pred_05)
            plot_confusion_matrix(cnf_matrix_05, classes=['No', 'Yes'],
                                  title='Threshold = 0.5',
                                  ax=axes[1])

            fig.suptitle(f'{title_prefix} Confusion Matrices')
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            # Page 5: Classification reports
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.axis('off')

            report_best = classification_report(actual, pred_best_f1, target_names=['No', 'Yes'])
            report_05 = classification_report(actual, pred_05, target_names=['No', 'Yes'])

            report_text = f"""
Classification Report at Best F1 Threshold ({best_f1_thresh:.3f}):
{report_best}

Classification Report at 0.5 Threshold:
{report_05}
"""
            ax.text(0.05, 0.95, report_text, transform=ax.transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            # Page 6: Threshold vs Precision + Histogram
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            axes[0].plot(thresholds, precision[:-1], color='b')
            axes[0].set_xlabel('Threshold probability')
            axes[0].set_ylabel('Precision (PPV)')
            axes[0].set_ylim([0.0, 1.0])
            axes[0].set_xlim([0.0, 1.0])
            axes[0].set_title('Threshold vs Precision')
            axes[0].grid(True, alpha=0.3)

            axes[1].hist(predicted, bins=50, edgecolor='black', alpha=0.7)
            axes[1].set_title("Prediction Distribution")
            axes[1].set_xlabel("Predicted probability")
            axes[1].set_ylabel("Frequency")
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            # Page 7: Calibration curve. calibration_curve requires predicted in
            # [0, 1]; some models (e.g. the trial-checker regression head) emit a
            # 0-5 score, so min-max normalize to [0, 1] for the reliability diagram
            # when the values fall outside that range. AUC/F1/regression metrics
            # above intentionally keep predicted on its native scale.
            fig, ax = plt.subplots(figsize=(8, 6))
            pred_min, pred_max = float(np.min(predicted)), float(np.max(predicted))
            if pred_min < 0.0 or pred_max > 1.0:
                if pred_max > pred_min:
                    predicted_cal = (predicted - pred_min) / (pred_max - pred_min)
                else:
                    predicted_cal = np.clip(predicted, 0.0, 1.0)
                cal_xlabel = 'Predicted score (min-max normalized to [0, 1])'
            else:
                predicted_cal = predicted
                cal_xlabel = 'Predicted probability'
            y_plot, x_plot = calibration_curve(actual, predicted_cal, n_bins=25)
            ax.plot(x_plot, y_plot, marker='o', linewidth=1, label='Model calibration')
            line = mlines.Line2D([0, 1], [0, 1], color='black', linestyle='--')
            transform = ax.transAxes
            line.set_transform(transform)
            ax.add_line(line)
            ax.set_title(f'{title_prefix} Calibration Plot')
            ax.set_xlabel(cal_xlabel)
            ax.set_ylabel('True probability in each bin')
            ax.legend()
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            # --- Regression metrics pages (only when continuous gold scores provided) ---
            if gold_continuous is not None:
                gold_continuous = np.asarray(gold_continuous)
                r, p_r = pearsonr(gold_continuous, predicted)
                rho, p_rho = spearmanr(gold_continuous, predicted)
                mae = np.mean(np.abs(gold_continuous - predicted))
                rmse = rmse_metric(gold_continuous, predicted)
                r2 = r2_metric(gold_continuous, predicted)
                bias = bias_metric(gold_continuous, predicted)
                r_ci = bootstrap_metric_ci(
                    [gold_continuous, predicted], pearson_metric,
                    n_bootstrap=n_bootstrap, random_state=random_state
                )
                rho_ci = bootstrap_metric_ci(
                    [gold_continuous, predicted], spearman_metric,
                    n_bootstrap=n_bootstrap, random_state=random_state
                )
                mae_ci = bootstrap_metric_ci(
                    [gold_continuous, predicted], mae_metric,
                    n_bootstrap=n_bootstrap, random_state=random_state
                )
                rmse_ci = bootstrap_metric_ci(
                    [gold_continuous, predicted], rmse_metric,
                    n_bootstrap=n_bootstrap, random_state=random_state
                )
                r2_ci = bootstrap_metric_ci(
                    [gold_continuous, predicted], r2_metric,
                    n_bootstrap=n_bootstrap, random_state=random_state
                )
                bias_ci = bootstrap_metric_ci(
                    [gold_continuous, predicted], bias_metric,
                    n_bootstrap=n_bootstrap, random_state=random_state
                )

                # Page: Regression summary metrics
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.axis('off')
                reg_text = f"""
{title_prefix} Regression Metrics
{'=' * 40}

Pearson r:    {format_metric_with_ci(r, r_ci)}  (p = {p_r:.4e})
Spearman rho: {format_metric_with_ci(rho, rho_ci)}  (p = {p_rho:.4e})
MAE:          {format_metric_with_ci(mae, mae_ci)}
RMSE:         {format_metric_with_ci(rmse, rmse_ci)}
R^2 (CoD):    {format_metric_with_ci(r2, r2_ci)}
Bias (pred-gold): {format_metric_with_ci(bias, bias_ci)}

Gold score range:      [{gold_continuous.min():.2f}, {gold_continuous.max():.2f}]
Predicted score range: [{predicted.min():.2f}, {predicted.max():.2f}]
N samples:             {len(gold_continuous)}
"""
                ax.text(0.1, 0.9, reg_text, transform=ax.transAxes,
                        fontsize=12, verticalalignment='top', fontfamily='monospace')
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

                # Page: Scatter plot of gold vs predicted
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(gold_continuous, predicted, alpha=0.15, s=10, edgecolors='none')
                score_min = min(gold_continuous.min(), predicted.min())
                score_max = max(gold_continuous.max(), predicted.max())
                ax.plot([score_min, score_max], [score_min, score_max], 'r--', label='y = x')
                ax.set_xlabel('Gold Score')
                ax.set_ylabel('Predicted Score')
                ax.set_title(f'{title_prefix} Gold vs Predicted (r={r:.3f}, ρ={rho:.3f})')
                ax.legend()
                ax.grid(True, alpha=0.3)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

        print(f"PDF report saved to: {pdf_path}")
        return best_f1_thresh

    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return None


def eval_model_categorical(predicted_probs: np.ndarray, actual_labels: np.ndarray,
                           class_names: list, pdf_path: Optional[str] = None,
                           title_prefix: str = "",
                           n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
                           random_state: int = DEFAULT_BOOTSTRAP_RANDOM_STATE) -> Optional[Dict[str, Any]]:
    """
    Evaluate multi-class model predictions and optionally save results to PDF.

    Args:
        predicted_probs: Array of shape (N, num_classes) with predicted probabilities
        actual_labels: Array of integer class labels (N,)
        class_names: List of class name strings
        pdf_path: Path to save PDF report
        title_prefix: Prefix for plot titles

    Returns:
        Dictionary with metrics or None if calculation fails
    """
    from sklearn.metrics import accuracy_score

    try:
        predicted_probs = np.asarray(predicted_probs)
        actual_labels = np.asarray(actual_labels)
        predicted_labels = np.argmax(predicted_probs, axis=1)
        accuracy = accuracy_score(actual_labels, predicted_labels)
        macro_f1 = f1_score(actual_labels, predicted_labels, average='macro')
        weighted_f1 = f1_score(actual_labels, predicted_labels, average='weighted')
        kappa = cohen_kappa_score(actual_labels, predicted_labels)
        metric_cis = {
            'accuracy': bootstrap_metric_ci(
                [actual_labels, predicted_labels],
                lambda y_true, y_pred: accuracy_score(y_true, y_pred),
                n_bootstrap=n_bootstrap, random_state=random_state
            ),
            'macro_f1': bootstrap_metric_ci(
                [actual_labels, predicted_labels],
                lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro'),
                n_bootstrap=n_bootstrap, random_state=random_state
            ),
            'weighted_f1': bootstrap_metric_ci(
                [actual_labels, predicted_labels],
                lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted'),
                n_bootstrap=n_bootstrap, random_state=random_state
            ),
            'kappa': bootstrap_metric_ci(
                [actual_labels, predicted_labels],
                lambda y_true, y_pred: cohen_kappa_score(y_true, y_pred),
                n_bootstrap=n_bootstrap, random_state=random_state
            ),
        }

        metrics = {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'kappa': kappa,
            'accuracy_ci': metric_cis['accuracy'],
            'macro_f1_ci': metric_cis['macro_f1'],
            'weighted_f1_ci': metric_cis['weighted_f1'],
            'kappa_ci': metric_cis['kappa'],
        }

        # Multiclass AUROC metrics (One-vs-Rest)
        try:
            macro_auroc = roc_auc_score(actual_labels, predicted_probs,
                                        multi_class='ovr', average='macro')
            weighted_auroc = roc_auc_score(actual_labels, predicted_probs,
                                           multi_class='ovr', average='weighted')
            macro_auroc_ci = bootstrap_metric_ci(
                [actual_labels, predicted_probs],
                lambda y_true, probs: roc_auc_score(
                    y_true, probs, multi_class='ovr', average='macro'
                ),
                n_bootstrap=n_bootstrap, random_state=random_state
            )
            weighted_auroc_ci = bootstrap_metric_ci(
                [actual_labels, predicted_probs],
                lambda y_true, probs: roc_auc_score(
                    y_true, probs, multi_class='ovr', average='weighted'
                ),
                n_bootstrap=n_bootstrap, random_state=random_state
            )
            metrics['macro_auroc'] = macro_auroc
            metrics['weighted_auroc'] = weighted_auroc
            metrics['macro_auroc_ci'] = macro_auroc_ci
            metrics['weighted_auroc_ci'] = weighted_auroc_ci
        except ValueError as e:
            macro_auroc = None
            weighted_auroc = None
            macro_auroc_ci = None
            weighted_auroc_ci = None
            print(f"Warning: Could not compute multiclass AUROC: {e}")

        # Per-class AUROC (one-vs-rest)
        per_class_auroc = {}
        per_class_auroc_ci = {}
        for i, name in enumerate(class_names):
            try:
                binary_labels = (actual_labels == i).astype(int)
                if binary_labels.sum() > 0 and binary_labels.sum() < len(binary_labels):
                    class_auroc = roc_auc_score(binary_labels, predicted_probs[:, i])
                    class_auroc_ci = bootstrap_metric_ci(
                        [binary_labels, predicted_probs[:, i]], binary_auroc_score,
                        n_bootstrap=n_bootstrap, random_state=random_state
                    )
                    per_class_auroc[name] = class_auroc
                    per_class_auroc_ci[name] = class_auroc_ci
                    metrics[f'auroc_{name}'] = class_auroc
                    metrics[f'auroc_{name}_ci'] = class_auroc_ci
            except ValueError:
                pass

        # Binary AUROC: first class vs rest (e.g., NO! vs any YES)
        binary_auroc = None
        binary_auroc_ci = None
        try:
            binary_gold = (actual_labels > 0).astype(int)
            binary_score = 1 - predicted_probs[:, 0]
            if binary_gold.sum() > 0 and binary_gold.sum() < len(binary_gold):
                binary_auroc = roc_auc_score(binary_gold, binary_score)
                binary_auroc_ci = bootstrap_metric_ci(
                    [binary_gold, binary_score], binary_auroc_score,
                    n_bootstrap=n_bootstrap, random_state=random_state
                )
                metrics['binary_auroc'] = binary_auroc
                metrics['binary_auroc_ci'] = binary_auroc_ci
        except ValueError as e:
            print(f"Warning: Could not compute binary AUROC: {e}")

        print(f"Accuracy: {format_metric_with_ci(accuracy, metric_cis['accuracy'])}")
        print(f"Macro F1: {format_metric_with_ci(macro_f1, metric_cis['macro_f1'])}")
        print(f"Weighted F1: {format_metric_with_ci(weighted_f1, metric_cis['weighted_f1'])}")
        print(f"Cohen's Kappa: {format_metric_with_ci(kappa, metric_cis['kappa'])}")
        if macro_auroc is not None:
            print(f"Macro AUROC (OvR): {format_metric_with_ci(macro_auroc, macro_auroc_ci)}")
        if weighted_auroc is not None:
            print(f"Weighted AUROC (OvR): {format_metric_with_ci(weighted_auroc, weighted_auroc_ci)}")
        if binary_auroc is not None:
            print(f"Binary AUROC ({class_names[0]} vs rest): {format_metric_with_ci(binary_auroc, binary_auroc_ci)}")
        for name, auc_val in per_class_auroc.items():
            print(f"  AUROC {name}: {format_metric_with_ci(auc_val, per_class_auroc_ci.get(name))}")

        if pdf_path is None:
            return metrics

        with PdfPages(pdf_path) as pdf:
            # Page 1: Summary metrics
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.axis('off')
            class_counts = np.bincount(actual_labels.astype(int), minlength=len(class_names))
            counts_str = "\n".join(f"  {name}: {count}" for name, count in zip(class_names, class_counts))
            auroc_str = ""
            if macro_auroc is not None:
                auroc_str += f"\nMacro AUROC (OvR): {format_metric_with_ci(macro_auroc, macro_auroc_ci)}"
            if weighted_auroc is not None:
                auroc_str += f"\nWeighted AUROC (OvR): {format_metric_with_ci(weighted_auroc, weighted_auroc_ci)}"
            if binary_auroc is not None:
                auroc_str += f"\nBinary AUROC ({class_names[0]} vs rest): {format_metric_with_ci(binary_auroc, binary_auroc_ci)}"
            if per_class_auroc:
                auroc_str += "\n\nPer-class AUROC (OvR):"
                for name, auc_val in per_class_auroc.items():
                    auroc_str += f"\n  {name}: {format_metric_with_ci(auc_val, per_class_auroc_ci.get(name))}"

            summary_text = f"""{title_prefix} Categorical Evaluation Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY METRICS
===============
Total samples: {len(actual_labels)}
Number of classes: {len(class_names)}

Accuracy: {format_metric_with_ci(accuracy, metric_cis['accuracy'])}
Macro F1: {format_metric_with_ci(macro_f1, metric_cis['macro_f1'])}
Weighted F1: {format_metric_with_ci(weighted_f1, metric_cis['weighted_f1'])}
Cohen's Kappa: {format_metric_with_ci(kappa, metric_cis['kappa'])}
{auroc_str}

CLASS DISTRIBUTION (actual):
{counts_str}
"""
            ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            # Page 2: Confusion matrix (raw)
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            cm = confusion_matrix(actual_labels, predicted_labels,
                                  labels=list(range(len(class_names))))
            plot_confusion_matrix(cm, classes=class_names,
                                  title='Confusion Matrix (counts)', ax=axes[0])
            plot_confusion_matrix(cm, classes=class_names, normalize=True,
                                  title='Confusion Matrix (normalized)', ax=axes[1])
            fig.suptitle(f'{title_prefix} Confusion Matrices')
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            # Page 3: Classification report
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.axis('off')
            report = classification_report(actual_labels, predicted_labels,
                                           target_names=class_names,
                                           labels=list(range(len(class_names))))
            report_text = f"Classification Report:\n\n{report}"
            ax.text(0.05, 0.95, report_text, transform=ax.transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            # Page 4: Per-class probability distributions
            n_classes = len(class_names)
            cols = min(n_classes, 3)
            rows = (n_classes + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
            axes_flat = np.array(axes).flatten() if n_classes > 1 else [axes]
            for i, (name, ax) in enumerate(zip(class_names, axes_flat)):
                ax.hist(predicted_probs[:, i], bins=30, edgecolor='black', alpha=0.7)
                ax.set_title(f'P({name})')
                ax.set_xlabel('Predicted probability')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
            for j in range(n_classes, len(axes_flat)):
                axes_flat[j].set_visible(False)
            fig.suptitle(f'{title_prefix} Per-Class Probability Distributions')
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            # Page 5: Per-class ROC curves (OvR)
            if per_class_auroc:
                fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
                axes_flat = np.array(axes).flatten() if n_classes > 1 else [axes]
                for i, (name, ax) in enumerate(zip(class_names, axes_flat)):
                    binary_labels = (actual_labels == i).astype(int)
                    if binary_labels.sum() > 0 and binary_labels.sum() < len(binary_labels):
                        fpr, tpr, _ = roc_curve(binary_labels, predicted_probs[:, i])
                        auc_val = per_class_auroc.get(name)
                        label = f"AUC = {format_metric_with_ci(auc_val, per_class_auroc_ci.get(name))}" if auc_val is not None else 'AUC = N/A'
                        ax.plot(fpr, tpr, 'b', label=label)
                        ax.plot([0, 1], [0, 1], 'r--', alpha=0.5)
                        ax.legend(loc='lower right', fontsize=8)
                    else:
                        ax.text(0.5, 0.5, 'N/A\n(single class)', ha='center', va='center')
                    ax.set_title(f'{name}')
                    ax.set_xlabel('FPR')
                    ax.set_ylabel('TPR')
                    ax.set_xlim([0, 1])
                    ax.set_ylim([0, 1])
                    ax.grid(True, alpha=0.3)
                for j in range(n_classes, len(axes_flat)):
                    axes_flat[j].set_visible(False)
                fig.suptitle(f'{title_prefix} Per-Class ROC Curves (OvR)')
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

        print(f"PDF report saved to: {pdf_path}")
        return metrics

    except Exception as e:
        print(f"Error calculating categorical metrics: {e}")
        import traceback
        traceback.print_exc()
        return None


def calculate_map_at_k(df: pd.DataFrame,
                       group_col: str,
                       label_col: str,
                       k: int = 20,
                       n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
                       random_state: int = DEFAULT_BOOTSTRAP_RANDOM_STATE) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate Mean Average Precision at K.

    Args:
        df: DataFrame with predictions (assumed to be sorted by rank within groups)
        group_col: Column to group by (e.g., 'patient_summary' or 'this_space')
        label_col: Column with binary labels
        k: Number of top results to consider

    Returns:
        Tuple of (MAP@K score, dictionary with additional stats)
    """
    # Take top k per group
    top_k = df.groupby(group_col).head(k)

    # Calculate AP for each group
    ap_scores = top_k.groupby(group_col)[label_col].apply(
        lambda x: average_precision_at_k(x.values)
    )

    map_k = ap_scores.mean()
    map_k_ci = bootstrap_mean_ci(
        ap_scores.values, n_bootstrap=n_bootstrap, random_state=random_state
    )
    positive_rate = top_k[label_col].mean()
    positive_rate_ci = bootstrap_metric_ci(
        [top_k[label_col].values], positive_rate_metric,
        n_bootstrap=n_bootstrap, random_state=random_state
    )

    stats = {
        'map_at_k': map_k,
        'map_at_k_ci': map_k_ci,
        'k': k,
        'num_groups': len(ap_scores),
        'total_samples': len(top_k),
        'positive_rate': positive_rate,
        'positive_rate_ci': positive_rate_ci,
        'median_group_size': top_k.groupby(group_col).size().median(),
        'mean_group_size': top_k.groupby(group_col).size().mean(),
    }

    return map_k, stats


def generate_ranking_report(df: pd.DataFrame,
                           group_col: str,
                           label_col: str,
                           pdf_path: str,
                           title_prefix: str = "",
                           k: int = 20,
                           n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
                           random_state: int = DEFAULT_BOOTSTRAP_RANDOM_STATE) -> Dict[str, Any]:
    """
    Generate a PDF report with ranking metrics.

    Args:
        df: DataFrame with predictions
        group_col: Column to group by
        label_col: Column with binary labels
        pdf_path: Path to save PDF report
        title_prefix: Prefix for titles
        k: Number of top results for MAP@K

    Returns:
        Dictionary with computed statistics
    """
    map_k, stats = calculate_map_at_k(
        df, group_col, label_col, k,
        n_bootstrap=n_bootstrap, random_state=random_state
    )

    with PdfPages(pdf_path) as pdf:
        # Summary page
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('off')

        summary_text = f"""
{title_prefix} Ranking Evaluation Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

RANKING METRICS
===============
MAP@{k}: {format_metric_with_ci(map_k, stats.get('map_at_k_ci'))}
Number of {group_col}s: {stats['num_groups']}
Total samples (top {k} per group): {stats['total_samples']}
Positive rate: {format_metric_with_ci(stats['positive_rate'], stats.get('positive_rate_ci'))}
Median results per {group_col}: {stats['median_group_size']:.1f}
Mean results per {group_col}: {stats['mean_group_size']:.1f}
"""
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
                fontsize=12, verticalalignment='top', fontfamily='monospace')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # Distribution of AP scores
        top_k = df.groupby(group_col).head(k)
        ap_scores = top_k.groupby(group_col)[label_col].apply(
            lambda x: average_precision_at_k(x.values)
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(ap_scores, bins=20, edgecolor='black', alpha=0.7)
        ax.axvline(
            map_k, color='r', linestyle='--',
            label=f"MAP@{k} = {format_metric_with_ci(map_k, stats.get('map_at_k_ci'))}"
        )
        ax.set_xlabel('Average Precision')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{title_prefix} Distribution of AP@{k} Scores')
        ax.legend()
        ax.grid(True, alpha=0.3)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # Results per group distribution
        fig, ax = plt.subplots(figsize=(8, 6))
        group_sizes = top_k.groupby(group_col).size()
        ax.hist(group_sizes, bins=min(k, 20), edgecolor='black', alpha=0.7)
        ax.set_xlabel(f'Number of results per {group_col}')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{title_prefix} Distribution of Results per {group_col}')
        ax.grid(True, alpha=0.3)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    print(f"Ranking report saved to: {pdf_path}")
    return stats


def load_and_combine_csv_files(directory: str, pattern: str = "*.csv") -> pd.DataFrame:
    """
    Load and combine multiple CSV files from a directory.

    Args:
        directory: Directory path
        pattern: Glob pattern for files

    Returns:
        Combined DataFrame
    """
    import glob

    all_files = glob.glob(str(Path(directory) / pattern))

    if not all_files:
        raise FileNotFoundError(f"No CSV files found in: {directory}")

    dfs = []
    for file_path in all_files:
        try:
            df = pd.read_csv(file_path)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

    if not dfs:
        raise ValueError("No DataFrames could be successfully loaded")

    return pd.concat(dfs, ignore_index=True)
