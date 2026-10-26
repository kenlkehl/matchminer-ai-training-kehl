#!/usr/bin/env python3
"""One-off single-GB300 training on all completed production labels, no new LLM calls."""
import argparse
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE = ROOT.parent / 'data/no_phi/synthetic_cohorts_5000_gemma4_31b_20261024/post_notes_colbert_20261025'
OUTPUT = ROOT.parent / 'data/no_phi/colbert_production_gb300_one_epoch_20261026'
EXPECTED_PAIRS = 2435540


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', type=Path, default=SOURCE)
    p.add_argument('--output', type=Path, default=OUTPUT)
    device = p.add_mutually_exclusive_group()
    device.add_argument('--gpu', help='Physical CUDA GPU index or UUID; explicitly overrides CUDA_VISIBLE_DEVICES')
    device.add_argument('--visible-gpu', type=int,
                        help='Index within inherited CUDA_VISIBLE_DEVICES; default is its first GPU, or GPU 0 if unset')
    p.add_argument('--cache-dir', type=Path, help='Local Hugging Face cache; otherwise use the environment default')
    p.add_argument('--documents-per-query', type=int, default=16)
    p.add_argument('--accumulation', type=int, default=64)
    p.add_argument('--epochs', type=float, default=1)
    p.add_argument('--gradient-checkpointing', action=argparse.BooleanOptionalAction, default=False,
                   help='Recompute encoder activations to save VRAM at the cost of speed')
    p.add_argument('--learning-rate', type=float, default=2e-5)
    p.add_argument('--eval-every', type=int, default=100)
    p.add_argument('--dry-run', action='store_true', help='Print the command without waiting, using GPUs or writing outputs')
    return p


def command(args):
    output, source = args.output.resolve(), args.source_run.resolve()
    result = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=1',
        str(ROOT / 'trialspace_training.py'), 'train',
        '--corpus', str(source / 'corpus_all'), '--labels', str(source / 'labels/scores.jsonl'),
        '--output', str(output / 'fit'), '--output-model', str(output / 'model'),
        '--model', 'lightonai/GTE-ModernColBERT-v1', '--query-length', '8192', '--document-length', '8192',
        '--objective', 'gated_regression', '--loss', 'legacy_ranking', '--prediction-link', 'identity',
        '--training-splits', 'train', 'val', 'test', '--require-complete-coverage',
        '--epochs', str(args.epochs), '--documents-per-query', str(args.documents_per_query),
        '--accumulation', str(args.accumulation), '--learning-rate', str(args.learning_rate),
        '--lr-scheduler', 'linear', '--warmup-ratio', '.01',
        '--ranking-scale', '20', '--contrastive-weight', '1',
        '--eval-every', str(args.eval_every), '--eval-groups-per-stratum', '32',
        '--save-steps', '25', '--save-total-limit', '2', '--pause-file', str(output / 'PAUSE')]
    result += ['--gradient-checkpointing' if args.gradient_checkpointing else '--no-gradient-checkpointing']
    if args.cache_dir:
        result += ['--cache-dir', str(args.cache_dir.resolve())]
    return result


def selected_gpu(gpu, visible, *, visible_index=None):
    """Explicit physical selection wins; visible indexing is a separate option."""
    if gpu is not None:
        gpu = gpu.strip()
        if not gpu or ',' in gpu or not (gpu.isdigit() or gpu.startswith(('GPU-', 'MIG-'))):
            raise ValueError('Select one nonnegative physical GPU index or GPU/MIG UUID')
        return gpu
    index = 0 if visible_index is None else visible_index
    if index < 0:
        raise ValueError('--visible-gpu must be nonnegative')
    if visible is None:
        return str(index)
    devices = [item.strip() for item in visible.split(',')]
    if any(not item or item == '-1' for item in devices):
        raise ValueError('CUDA_VISIBLE_DEVICES disables CUDA or has an invalid device list')
    if index >= len(devices):
        raise ValueError(f'--visible-gpu {index} is outside CUDA_VISIBLE_DEVICES={visible}; '
                         f'use a visible index from 0 to {len(devices) - 1}')
    return devices[index]


def ready(source):
    """Verify compact training files by content; never scan reasoning checkpoints."""
    from colbert_experiments.score_publication import ready_scores
    publication = ready_scores(source / 'labels', corpus=source / 'corpus_all',
                               candidates=source / 'mining_all/candidates.jsonl', expected_pairs=EXPECTED_PAIRS)
    if publication is not None:
        return True, dict(state='training_scores_ready', scored_pairs=publication['pairs'])
    workflow = json.loads((source / 'status.json').read_text())
    labels = json.loads((source / 'labels/status.json').read_text())
    if labels.get('failures'):
        raise RuntimeError('Source labeling failed with unresolved question labels')
    if workflow.get('phase') == 'failed' or labels.get('state') == 'failed':
        raise RuntimeError('Source workflow failed before publishing verified training scores: '
                           + str(workflow.get('error', labels.get('error', 'unknown error')))
                           + f". Saved pairs: {labels.get('scored_pairs')}/{labels.get('total_pairs')}. "
                           'Resume CPU-only label finalization; this is not evidence that saved labels are missing.')
    if labels.get('total_pairs') != EXPECTED_PAIRS:
        raise ValueError('This one-off launcher expects the repaired 2,435,540-pair production run')
    if workflow.get('phase') != 'completed' or labels.get('state') != 'complete':
        return False, labels
    if labels.get('scored_pairs') != EXPECTED_PAIRS:
        raise ValueError('Source has incomplete score coverage')
    required = ['completed/labeling.json', 'labels/scores.jsonl',
                'labels/labels.jsonl', 'corpus_all/patients.jsonl', 'corpus_all/cohorts.jsonl']
    if not all((source / item).is_file() for item in required):
        raise FileNotFoundError('Completed production label/corpus files are missing')
    return True, labels


def main():
    args = parser().parse_args()
    if min(args.documents_per_query, args.accumulation) < 1:
        raise ValueError('Select one GPU and positive microbatch/accumulation sizes')
    gpu = selected_gpu(args.gpu, os.environ.get('CUDA_VISIBLE_DEVICES'), visible_index=args.visible_gpu)
    launch = command(args)
    print('CUDA_VISIBLE_DEVICES=' + shlex.quote(gpu) + ' ' + shlex.join(launch), flush=True)
    selection = 'explicit --gpu' if args.gpu is not None else 'visible GPU selection'
    print(f'Training subprocess will see physical CUDA device {gpu} as cuda:0 ({selection}).', flush=True)
    print(f'Both directions; all {EXPECTED_PAIRS:,} pairs; {args.epochs:g} complete epochs; '
          f'up to {args.documents_per_query * args.accumulation:,} directional pairs per update.', flush=True)
    if args.dry_run:
        return
    if socket.gethostname().split('.')[0].lower().startswith('camus'):
        raise RuntimeError('This run was moved off Camus. Run this launcher on the GB300 host.')
    source = args.source_run.resolve()
    print('Verifying training-file content hashes with sequential reads; no reasoning checkpoints are scanned.', flush=True)
    while True:
        done, status = ready(source)
        if done:
            break
        print('Waiting for validated production training scores: ' + json.dumps({key: status.get(key) for key in
              ('state', 'scored_pairs', 'exported_questions', 'total_questions')}), flush=True)
        time.sleep(30)
    environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu, 'PYTHONUNBUFFERED': '1',
                   'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': '4',
                   'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4'}
    paths = [str(ROOT)]
    sibling = ROOT.parent / 'matchminer-ai-inference-kehl/src'
    if sibling.exists():
        paths.insert(0, str(sibling))
    if environment.get('PYTHONPATH'):
        paths.append(environment['PYTHONPATH'])
    environment['PYTHONPATH'] = os.pathsep.join(paths)
    args.output.mkdir(parents=True, exist_ok=True)
    print('Complete production training scores are ready. Starting training; resume by repeating this command.', flush=True)
    result = subprocess.run(launch, cwd=ROOT, env=environment)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
