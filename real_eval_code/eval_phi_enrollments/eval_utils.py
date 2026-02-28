#!/usr/bin/env python3
"""
Shared evaluation utilities for clinical trial matching model evaluation.

This module provides functions to:
- Calculate evaluation metrics (AUC, F1, precision-recall, etc.)
- Generate PDF reports with visualizations
- Compute MAP@K metrics for ranking evaluation
"""

import itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for PDF generation
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import (
    roc_auc_score, f1_score, classification_report,
    precision_recall_curve, auc, roc_curve, confusion_matrix,
    average_precision_score, cohen_kappa_score
)
from sklearn.calibration import calibration_curve
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, Any
import pandas as pd


def sigmoid(x):
    """Apply sigmoid function to input."""
    return 1. / (1. + np.exp(-x))


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
               title_prefix: str = "") -> Optional[float]:
    """
    Evaluate model predictions and optionally save results to PDF.

    Args:
        predicted: Predicted scores/probabilities
        actual: Actual binary labels
        pdf_path: Path to save PDF report (if None, just returns metrics)
        title_prefix: Prefix for plot titles

    Returns:
        Best F1 threshold or None if calculation fails
    """
    outcome_counts = np.unique(actual, return_counts=True)[1]

    try:
        prob_outcome = outcome_counts[1] / (outcome_counts[0] + outcome_counts[1])
        auc_score = roc_auc_score(actual, predicted)

        # Calculate ROC curve
        fpr, tpr, threshold = roc_curve(actual, predicted)
        roc_auc = auc(fpr, tpr)

        # Calculate precision-recall
        avg_precision = average_precision_score(actual, predicted)
        precision, recall, thresholds = precision_recall_curve(actual, predicted)

        # Best F1
        F1 = 2 * ((precision * recall) / (precision + recall + 1e-10))
        best_f1 = max(F1)
        best_f1_thresh = thresholds[np.argmax(F1)] if len(thresholds) > 0 else 0.5

        # Print metrics summary
        print(f"AUC: {auc_score:.4f}")
        print(f"Outcome probability: {prob_outcome:.4f}")
        print(f"Average precision score: {avg_precision:.4f}")
        print(f"Best F1: {best_f1:.4f}")
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

AUC-ROC: {auc_score:.4f}
Average Precision: {avg_precision:.4f}
Best F1 Score: {best_f1:.4f}
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

            # Page 7: Calibration curve
            fig, ax = plt.subplots(figsize=(8, 6))
            y_plot, x_plot = calibration_curve(actual, predicted, n_bins=25)
            ax.plot(x_plot, y_plot, marker='o', linewidth=1, label='Model calibration')
            line = mlines.Line2D([0, 1], [0, 1], color='black', linestyle='--')
            transform = ax.transAxes
            line.set_transform(transform)
            ax.add_line(line)
            ax.set_title(f'{title_prefix} Calibration Plot')
            ax.set_xlabel('Predicted probability')
            ax.set_ylabel('True probability in each bin')
            ax.legend()
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
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
                           title_prefix: str = "") -> Optional[Dict[str, Any]]:
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
        predicted_labels = np.argmax(predicted_probs, axis=1)
        accuracy = accuracy_score(actual_labels, predicted_labels)
        macro_f1 = f1_score(actual_labels, predicted_labels, average='macro')
        weighted_f1 = f1_score(actual_labels, predicted_labels, average='weighted')
        kappa = cohen_kappa_score(actual_labels, predicted_labels)

        metrics = {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'kappa': kappa,
        }

        # Multiclass AUROC metrics (One-vs-Rest)
        try:
            macro_auroc = roc_auc_score(actual_labels, predicted_probs,
                                        multi_class='ovr', average='macro')
            weighted_auroc = roc_auc_score(actual_labels, predicted_probs,
                                           multi_class='ovr', average='weighted')
            metrics['macro_auroc'] = macro_auroc
            metrics['weighted_auroc'] = weighted_auroc
        except ValueError as e:
            macro_auroc = None
            weighted_auroc = None
            print(f"Warning: Could not compute multiclass AUROC: {e}")

        # Per-class AUROC (one-vs-rest)
        per_class_auroc = {}
        for i, name in enumerate(class_names):
            try:
                binary_labels = (actual_labels == i).astype(int)
                if binary_labels.sum() > 0 and binary_labels.sum() < len(binary_labels):
                    class_auroc = roc_auc_score(binary_labels, predicted_probs[:, i])
                    per_class_auroc[name] = class_auroc
                    metrics[f'auroc_{name}'] = class_auroc
            except ValueError:
                pass

        # Binary AUROC: first class vs rest (e.g., NO! vs any YES)
        binary_auroc = None
        try:
            binary_gold = (actual_labels > 0).astype(int)
            binary_score = 1 - predicted_probs[:, 0]
            if binary_gold.sum() > 0 and binary_gold.sum() < len(binary_gold):
                binary_auroc = roc_auc_score(binary_gold, binary_score)
                metrics['binary_auroc'] = binary_auroc
        except ValueError as e:
            print(f"Warning: Could not compute binary AUROC: {e}")

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
        print(f"Weighted F1: {weighted_f1:.4f}")
        print(f"Cohen's Kappa: {kappa:.4f}")
        if macro_auroc is not None:
            print(f"Macro AUROC (OvR): {macro_auroc:.4f}")
        if weighted_auroc is not None:
            print(f"Weighted AUROC (OvR): {weighted_auroc:.4f}")
        if binary_auroc is not None:
            print(f"Binary AUROC ({class_names[0]} vs rest): {binary_auroc:.4f}")
        for name, auc_val in per_class_auroc.items():
            print(f"  AUROC {name}: {auc_val:.4f}")

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
                auroc_str += f"\nMacro AUROC (OvR): {macro_auroc:.4f}"
            if weighted_auroc is not None:
                auroc_str += f"\nWeighted AUROC (OvR): {weighted_auroc:.4f}"
            if binary_auroc is not None:
                auroc_str += f"\nBinary AUROC ({class_names[0]} vs rest): {binary_auroc:.4f}"
            if per_class_auroc:
                auroc_str += "\n\nPer-class AUROC (OvR):"
                for name, auc_val in per_class_auroc.items():
                    auroc_str += f"\n  {name}: {auc_val:.4f}"

            summary_text = f"""{title_prefix} Categorical Evaluation Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY METRICS
===============
Total samples: {len(actual_labels)}
Number of classes: {len(class_names)}

Accuracy: {accuracy:.4f}
Macro F1: {macro_f1:.4f}
Weighted F1: {weighted_f1:.4f}
Cohen's Kappa: {kappa:.4f}
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
                        label = f'AUC = {auc_val:.4f}' if auc_val else 'AUC = N/A'
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
                       k: int = 20) -> Tuple[float, Dict[str, Any]]:
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

    stats = {
        'map_at_k': map_k,
        'k': k,
        'num_groups': len(ap_scores),
        'total_samples': len(top_k),
        'positive_rate': top_k[label_col].mean(),
        'median_group_size': top_k.groupby(group_col).size().median(),
        'mean_group_size': top_k.groupby(group_col).size().mean(),
    }

    return map_k, stats


def generate_ranking_report(df: pd.DataFrame,
                           group_col: str,
                           label_col: str,
                           pdf_path: str,
                           title_prefix: str = "",
                           k: int = 20) -> Dict[str, Any]:
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
    map_k, stats = calculate_map_at_k(df, group_col, label_col, k)

    with PdfPages(pdf_path) as pdf:
        # Summary page
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('off')

        summary_text = f"""
{title_prefix} Ranking Evaluation Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

RANKING METRICS
===============
MAP@{k}: {map_k:.4f}
Number of {group_col}s: {stats['num_groups']}
Total samples (top {k} per group): {stats['total_samples']}
Positive rate: {stats['positive_rate']:.4f}
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
        ax.axvline(map_k, color='r', linestyle='--', label=f'MAP@{k} = {map_k:.4f}')
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
