#!/usr/bin/env python3
"""Single-GB300 retrieval, fresh teacher labels, and training on promoted cohorts."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT.parent / 'data/no_phi'
PATIENTS = DATA / 'synthetic_cohorts_5000_gemma4_31b_20261024/post_notes_colbert_20261025/prepared/patient_summaries_with_spaces.parquet'
COHORTS = DATA / 'sample_trial_cohort_lineitems.csv'
SOURCE = DATA / 'colbert_namedcohorts_gb300_20261026'
OUTPUT = DATA / 'colbert_production_gb300_namedcohorts_20261026'


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', type=Path, default=SOURCE)
    p.add_argument('--output', type=Path, default=OUTPUT)
    p.add_argument('--cohorts', type=Path, default=COHORTS)
    p.add_argument('--patients', type=Path, default=PATIENTS,
                   help='Existing repaired synthetic summaries; no patient generation or summarization')
    teacher = p.add_mutually_exclusive_group()
    teacher.add_argument('--teacher-url', default='http://127.0.0.1:8000/v1')
    teacher.add_argument('--teacher-servers-file', type=Path,
                         help='Live endpoint registry; healthy servers may join and leave during labeling')
    p.add_argument('--teacher-concurrency-file', type=Path,
                   help='Live global/per-endpoint adaptive concurrency ceilings')
    p.add_argument('--teacher-model', default='gemma4-31b')
    p.add_argument('--teacher-tokenizer', default='nvidia/Gemma-4-31B-IT-NVFP4')
    p.add_argument('--label-concurrency', type=int, default=320)
    p.add_argument('--mining-batch-size', type=int, default=16)
    p.add_argument('--mining-pair-batch-size', type=int, default=512)
    p.add_argument('--mining-device-cache-gb', type=float, default=8)
    p.add_argument('--mining-device-cache-reserve-gb', type=float, default=8)
    p.add_argument('--mining-device-memory-limit-gb', type=float, default=0,
                   help='Optional PyTorch allocator limit when retrieval shares a GPU with vLLM')
    p.add_argument('--stop-after', choices=['prepare', 'mine', 'label', 'train'], default='train',
                   help='Run the resumable stages through this point; default includes training')
    device = p.add_mutually_exclusive_group()
    device.add_argument('--gpu', help='Physical CUDA GPU index or UUID; explicitly overrides CUDA_VISIBLE_DEVICES')
    device.add_argument('--visible-gpu', type=int,
                        help='Index within inherited CUDA_VISIBLE_DEVICES; default is its first GPU, or GPU 0 if unset')
    p.add_argument('--cache-dir', type=Path, help='Local Hugging Face cache; otherwise use the environment default')
    p.add_argument('--documents-per-query', type=int, default=16)
    p.add_argument('--accumulation', type=int, default=64)
    p.add_argument('--query-groups-per-forward', type=int, default=8)
    p.add_argument('--encoder-token-budget', type=int, default=131072)
    p.add_argument('--token-cache-tokens', type=int, default=8000000)
    p.add_argument('--ranking-backward', choices=['replay', 'retained'], default='replay')
    p.add_argument('--activation-storage', choices=['auto', 'gpu', 'cpu', 'direct'], default='auto')
    p.add_argument('--activation-gpu-budget-gb', type=float, default=0,
                   help='Auto/direct retention soft GPU limit in GiB; zero reserves half initially free VRAM')
    p.add_argument('--epochs', type=float, default=1)
    p.add_argument('--gradient-checkpointing', action=argparse.BooleanOptionalAction, default=False,
                   help='Recompute encoder activations to save VRAM at the cost of speed')
    p.add_argument('--learning-rate', type=float, default=2e-5)
    p.add_argument('--eval-every', type=int, default=100)
    p.add_argument('--dry-run', action='store_true', help='Print the command without waiting, using GPUs or writing outputs')
    return p


def preparation_commands(args):
    """Use production stages with one shared, generation-specific corpus."""
    source = args.source_run.resolve()
    program = [sys.executable, str(ROOT / 'trialspace_training.py')]
    corpus = str(source / 'corpus_all')
    prepare = program + ['prepare', '--patients', str(args.patients.resolve()),
        '--cohorts', str(args.cohorts.resolve()), '--output', corpus, '--merge-source-splits']
    mine = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=1',
        *program[1:], 'mine', '--corpus', corpus, '--output', str(source / 'mining_all'),
        '--model', 'lightonai/GTE-ModernColBERT-v1', '--query-length', '8192', '--document-length', '8192',
        '--candidate-limit', '256', '--proposal-strategy', 'lexical_random',
        '--top-k-patient-to-cohort', '20', '--top-k-cohort-to-patient', '40',
        '--lexical-k', '5', '--random-k', '5', '--document-batch-size', str(args.mining_batch_size),
        '--query-batch-size', str(args.mining_batch_size), '--score-batch-size', str(args.mining_batch_size),
        '--cache-batches', '256', '--pair-batch-size', str(args.mining_pair_batch_size),
        '--score-query-batches', '4', '--device-cache-gb', str(args.mining_device_cache_gb),
        '--device-cache-reserve-gb', str(args.mining_device_cache_reserve_gb),
        '--device-memory-limit-gb', str(args.mining_device_memory_limit_gb),
        '--splits', 'train', 'val', 'test']
    label = program + ['label', '--corpus', corpus,
        '--candidates', str(source / 'mining_all/candidates.jsonl'), '--output', str(source / 'labels'),
        '--teacher-model', args.teacher_model, '--teacher-tokenizer', args.teacher_tokenizer,
        '--teacher-context', '250000',
        '--max_concurrent_per_server', str(args.label_concurrency),
        '--initial-concurrent-per-server', str(min(96, args.label_concurrency)),
        '--pending-requests', '0', '--max_attempts', '16', '--request_timeout', '7200',
        '--gated-regression', '--scheduler-adaptive-concurrency']
    label += (['--server_urls_file', str(args.teacher_servers_file.resolve())] if args.teacher_servers_file
              else ['--server_urls', args.teacher_url])
    if args.teacher_concurrency_file:
        label += ['--concurrency-control-file', str(args.teacher_concurrency_file.resolve())]
    if args.cache_dir:
        for stage in (mine, label):
            stage += ['--cache-dir', str(args.cache_dir.resolve())]
    return [('prepare', prepare), ('mine', mine), ('label', label)]


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
        '--query-groups-per-forward', str(args.query_groups_per_forward),
        '--encoder-token-budget', str(args.encoder_token_budget),
        '--token-cache-tokens', str(args.token_cache_tokens),
        '--ranking-backward', args.ranking_backward, '--activation-storage', args.activation_storage,
        '--activation-gpu-budget-gb', str(args.activation_gpu_budget_gb),
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
    completion = source / 'mining_all/complete.json'
    if not completion.exists():
        return False, dict(state='retrieval_not_complete')
    expected_pairs = json.loads(completion.read_text())['pairs']
    if not isinstance(expected_pairs, int) or expected_pairs < 1:
        raise ValueError('Completed retrieval must contain a positive candidate count')
    publication = ready_scores(source / 'labels', corpus=source / 'corpus_all',
                               candidates=source / 'mining_all/candidates.jsonl', expected_pairs=expected_pairs)
    if publication is not None:
        return True, dict(state='training_scores_ready', scored_pairs=publication['pairs'])
    return False, dict(state='training_scores_not_published', total_pairs=expected_pairs)


def label_settings(args):
    """Only semantic label settings belong in the resume identity."""
    return dict(teacher_model=args.teacher_model, teacher_tokenizer=args.teacher_tokenizer,
                teacher_context=250000, max_attempts=16, request_timeout=7200)


def run_process(command, *, environment, status_path, phase):
    """Expose the actual child identity to the fleet's inference-complete guard."""
    from colbert_experiments.data import write_json
    with subprocess.Popen(command, cwd=ROOT, env=environment) as child:
        write_json(status_path, dict(phase=phase, pid=os.getpid(), child_pid=child.pid, command=command))
        try:
            return child.wait()
        except BaseException:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            raise


def run_stages(args, environment):
    from colbert_experiments.data import write_json
    source = args.source_run.resolve()
    source.mkdir(parents=True, exist_ok=True)
    with (source / 'gb300_workflow.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        settings_path = source / 'gb300_label_settings.json'
        settings = label_settings(args)
        if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
            raise ValueError('Teacher settings changed; use a new --source-run and --output')
        write_json(settings_path, settings)
        status_path = source / 'gb300_status.json'
        for name, stage in preparation_commands(args):
            # Prepare and mine validate their own source/settings manifests even
            # on resume. Only a complete content-verified score publication skips
            # label replay (which would reopen the large reasoning archive).
            done, status = ready(source) if name == 'label' else (False, {})
            phase = {'prepare': 'preparing_corpus', 'mine': 'mining', 'label': 'labeling'}[name]
            write_json(status_path, dict(phase=phase, pid=os.getpid(), command=stage))
            if not done:
                returncode = run_process(stage, environment=environment, status_path=status_path, phase=phase)
                if returncode:
                    write_json(status_path, dict(phase='failed', stage=name, returncode=returncode))
                    raise SystemExit(returncode)
            if name == 'label':
                done, status = ready(source)
                if not done:
                    raise RuntimeError('Labeling did not publish complete training scores: ' + json.dumps(status))
                print(f"Verified {status['scored_pairs']:,} freshly retrieved training pairs.", flush=True)
            if args.stop_after == name:
                write_json(status_path, dict(phase='completed' if name == 'label' else 'stopped', completed_stage=name))
                return 0
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(status_path, dict(phase='training', pid=os.getpid(), command=command(args)))
        print('Complete production training scores are ready. Starting training; resume by repeating this command.', flush=True)
        returncode = run_process(command(args), environment=environment, status_path=status_path, phase='training')
        write_json(status_path, dict(phase={0: 'completed', 75: 'paused'}.get(returncode, 'failed'),
                                    stage='train', returncode=returncode))
        return returncode


def main():
    args = parser().parse_args()
    if min(args.documents_per_query, args.accumulation) < 1:
        raise ValueError('Select one GPU and positive microbatch/accumulation sizes')
    if min(args.query_groups_per_forward, args.encoder_token_budget) < 1 or args.token_cache_tokens < 0:
        raise ValueError('Use positive encoder batch limits and nonnegative token cache capacity')
    if args.ranking_backward == 'retained' and args.gradient_checkpointing:
        raise ValueError('Retained mode uses one forward; leave gradient checkpointing disabled')
    if not 1 <= args.label_concurrency <= 1024:
        raise ValueError('--label-concurrency must be between 1 and 1024')
    if min(args.mining_batch_size, args.mining_pair_batch_size) < 1 or args.mining_device_cache_gb < 0:
        raise ValueError('Use positive mining batches and a nonnegative device cache budget')
    gpu = selected_gpu(args.gpu, os.environ.get('CUDA_VISIBLE_DEVICES'), visible_index=args.visible_gpu)
    launch = command(args)
    if args.stop_after == 'train':
        print('CUDA_VISIBLE_DEVICES=' + shlex.quote(gpu) + ' ' + shlex.join(launch), flush=True)
    else:
        print(f'CUDA_VISIBLE_DEVICES={shlex.quote(gpu)}; plan stops after {args.stop_after}.', flush=True)
    selection = 'explicit --gpu' if args.gpu is not None else 'visible GPU selection'
    print(f'Training subprocess will see physical CUDA device {gpu} as cuda:0 ({selection}).', flush=True)
    print(f'Both directions; all freshly retrieved/labeled pairs; {args.epochs:g} complete epochs; '
          f'up to {args.documents_per_query * args.accumulation:,} directional pairs per update.', flush=True)
    print(f'Up to {args.query_groups_per_forward} query groups per encoder forward; '
          f'{args.encoder_token_budget:,} padded-token packing budget; CPU token cache enabled '
          f'for up to {args.token_cache_tokens:,} tokens.', flush=True)
    print(f'Ranking backward: {args.ranking_backward}; activation storage: {args.activation_storage}.', flush=True)
    print(f'Promoted cohorts: {args.cohorts.resolve()}', flush=True)
    print(f'Existing patient summaries: {args.patients.resolve()}', flush=True)
    for name, stage in preparation_commands(args):
        print(f'{name}: ' + shlex.join(stage), flush=True)
        if name == args.stop_after:
            break
    if args.dry_run:
        return
    if socket.gethostname().split('.')[0].lower().startswith('camus'):
        raise RuntimeError('This run was moved off Camus. Run this launcher on the GB300 host.')
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
    raise SystemExit(run_stages(args, environment))


if __name__ == '__main__':
    main()
