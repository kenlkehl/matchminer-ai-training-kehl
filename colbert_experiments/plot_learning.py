"""Save standalone loss curves and a numerical summary for a completed fit."""
import argparse
import json
from pathlib import Path

import numpy as np

from .data import write_json


def plot(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    directory = Path(directory)
    metrics = json.loads((directory / 'metrics.json').read_text())
    manifest = json.loads((directory / 'training_manifest.json').read_text())
    regression = manifest['loss'] == 'mse_normalized_gated_score'
    loss_label = 'MSE on normalized score' if regression else 'Pairwise ranking loss'
    history, evaluations = metrics['history'], metrics['evaluations']
    if not history or not evaluations:
        raise ValueError('Training history and fixed-probe evaluations are required')
    steps = [row['step'] for row in history]
    losses = np.array([row['loss'] for row in history])
    window = min(50, len(losses))
    smooth = np.convolve(losses, np.ones(window) / window, mode='valid')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    axes[0].plot(steps, losses, alpha=.18, color='#1f77b4', label='Minibatch loss')
    axes[0].plot(steps[window - 1:], smooth, color='#1f77b4', label=f'{window}-update mean')
    axes[0].set(title='Training updates', ylabel=loss_label)
    for split, color, label in [('train', '#1f77b4', 'Fixed train groups'), ('val', '#d55e00', 'Fixed validation groups')]:
        available = [r for r in evaluations if r[split]['loss'] is not None]
        axes[1].plot([r['step'] for r in available], [r[split]['loss'] for r in available],
                     marker='o', color=color, label=label)
        axes[2].plot([r['step'] for r in available], [r[split]['pairwise_accuracy'] for r in available],
                     marker='o', color=color, label=label)
    axes[1].set(title='Same labeled comparisons at each checkpoint', ylabel=loss_label)
    axes[2].set(title='Higher label ranked first' if regression else 'Positive ranked above negative',
                ylabel='Pairwise accuracy', ylim=(0, 1))
    for ax in axes:
        ax.set_xlabel('Optimizer updates')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle(('Gated score regression' if regression else 'Five-task ColBERT') +
                 ': v22 synthetic summaries and extracted trial cohorts')
    fig.tight_layout()
    fig.savefig(directory / 'learning_curves.png', dpi=180)
    plt.close(fig)
    summary = dict(steps=steps[-1], first_50_mean_loss=float(losses[:window].mean()),
                   last_50_mean_loss=float(losses[-window:].mean()),
                   baseline=evaluations[0], final=evaluations[-1],
                   evaluation_scope='Fixed labeled candidate pools, macro-averaged over task/direction strata; not full-corpus recall')
    write_json(directory / 'learning_summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    print(json.dumps(plot(parser.parse_args().directory), indent=2))
