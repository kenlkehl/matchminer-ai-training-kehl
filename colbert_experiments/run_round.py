"""One resumable mining -> five-task labeling -> training round, isolated from production."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .data import write_json
from .model import BASE_MODEL
from .label_concurrency import DEFAULT_CEILING, DEFAULT_INITIAL, DEFAULT_AUTO_WORKERS_MAX


def semantic_commands(commands):
    """Endpoint routing and concurrency do not change the clinical label contract."""
    runtime = {'--max_concurrent_per_server', '--initial-concurrent-per-server',
               '--pending-requests', '--max-auto-pending-requests', '--concurrency-control-file', '--server_urls',
               '--server_urls_file', '--server_urls_refresh', '--teacher-served-model'}
    normalized = {}
    for stage, command in commands.items():
        result, index = [], 0
        while index < len(command):
            if stage == 'labeling' and command[index] in runtime:
                index += 2
            else:
                result.append(command[index])
                index += 1
        normalized[stage] = result
    return normalized


def compatible_commands(previous, current, *, training_started):
    old, new = semantic_commands(previous), semantic_commands(current)
    if old == new:
        return True
    if training_started:
        return False

    def training_execution(command):
        command = list(command)
        world, accumulation = 1, 1
        for flag in ('--nproc_per_node', '--accumulation'):
            if flag in command:
                index = command.index(flag)
                value = int(command[index + 1])
                del command[index:index + 2]
                if flag == '--nproc_per_node':
                    world = value
                else:
                    accumulation = value
        return command, world * accumulation

    old_training, new_training = old.pop('training'), new.pop('training')
    return old == new and training_execution(old_training) == training_execution(new_training)


def busy_training_gpus(gpus):
    """Only inspect the selected physical GPUs; never terminate another GPU user."""
    catalog = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid',
                                       '--format=csv,noheader,nounits'], text=True)
    devices = {uuid.strip(): index.strip() for index, uuid in
               (line.split(',') for line in catalog.splitlines() if line.strip())}
    if set(gpus) - set(devices.values()):
        raise ValueError('Selected training GPU is unavailable')
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                         '--format=csv,noheader,nounits'], text=True)
    return [dict(gpu=devices[uuid.strip()], pid=int(pid.strip()))
            for uuid, pid in (line.split(',') for line in processes.splitlines() if line.strip())
            if devices.get(uuid.strip()) in gpus]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--model', default=BASE_MODEL, help='Base checkpoint or previous round model directory')
    p.add_argument('--revision')
    p.add_argument('--cache-dir')
    p.add_argument('--gpus', required=True, help='Explicit physical GPU IDs, e.g. 2,3')
    p.add_argument('--training-gpus', help='Subset of --gpus for fitting; accumulation scales to preserve the effective query-group batch')
    p.add_argument('--stop-teacher-before-training', help='Explicit local-teacher process registration authorized for shutdown after successful labeling')
    p.add_argument('--teacher-url', action='append', required=True)
    p.add_argument('--teacher-model', required=True)
    p.add_argument('--teacher-served-model')
    p.add_argument('--teacher-tokenizer', required=True)
    p.add_argument('--teacher-concurrency', type=int, default=DEFAULT_CEILING)
    p.add_argument('--teacher-initial-concurrency', type=int, default=DEFAULT_INITIAL)
    p.add_argument('--teacher-pending-requests', type=int, default=0)
    p.add_argument('--teacher-max-auto-pending-requests', type=int, default=DEFAULT_AUTO_WORKERS_MAX)
    p.add_argument('--teacher-concurrency-file')
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--accumulation', type=int, default=8)
    p.add_argument('--top-k', type=int, default=10)
    p.add_argument('--lexical-k', type=int, default=5)
    p.add_argument('--random-k', type=int, default=5)
    p.add_argument('--candidate-limit', type=int)
    p.add_argument('--proposal-strategy', choices=['random', 'lexical_random'], default='random')
    p.add_argument('--cohort-query-limit', type=int)
    p.add_argument('--eval-every', type=int, default=100)
    p.add_argument('--extra-pairs')
    p.add_argument('--context-length', type=int, default=8192)
    p.add_argument('--objective', choices=['binary', 'gated_regression'], default='binary')
    p.add_argument('--documents-per-query', type=int, default=4)
    p.add_argument('--prediction-link', choices=['identity', 'sigmoid'], default='identity')
    args = p.parse_args()
    gpus = args.gpus.split(',')
    if not gpus or len(set(gpus)) != len(gpus) or not all(g.isdigit() for g in gpus):
        p.error('--gpus must contain distinct physical GPU numbers')
    training_gpus = args.training_gpus.split(',') if args.training_gpus else gpus
    if not training_gpus or len(set(training_gpus)) != len(training_gpus) or not set(training_gpus) <= set(gpus):
        p.error('--training-gpus must be a nonempty subset of --gpus')
    effective_batch = args.accumulation * len(gpus)
    if effective_batch % len(training_gpus):
        p.error('The training GPU count must divide the effective query-group batch')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': args.gpus, 'TOKENIZERS_PARALLELISM': 'false',
           'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS', '4')}
    common = ['--corpus', str(Path(args.corpus).resolve())]
    model_name = str(Path(args.model).resolve()) if Path(args.model).is_dir() else args.model
    model = ['--model', model_name, '--query-length', str(args.context_length),
             '--document-length', str(args.context_length)]
    if args.revision:
        model += ['--revision', args.revision]
    if args.cache_dir:
        model += ['--cache-dir', args.cache_dir]
    mine = [sys.executable, '-m', 'colbert_experiments', 'mine', *common, *model,
            '--output', str(output / 'mining'), '--top-k', str(args.top_k),
            '--lexical-k', str(args.lexical_k), '--random-k', str(args.random_k)]
    if args.candidate_limit:
        mine += ['--candidate-limit', str(args.candidate_limit)]
    mine += ['--proposal-strategy', args.proposal_strategy]
    if args.cohort_query_limit:
        mine += ['--cohort-query-limit', str(args.cohort_query_limit)]
    if args.extra_pairs:
        mine += ['--extra-pairs', str(Path(args.extra_pairs).resolve())]
    label = [sys.executable, '-m', 'colbert_experiments', 'label', *common,
        '--candidates', str(output / 'mining/candidates.jsonl'), '--output', str(output / 'teacher'),
        '--teacher-model', args.teacher_model, '--teacher-tokenizer', args.teacher_tokenizer,
        '--server_urls', ','.join(args.teacher_url), '--max_concurrent_per_server', str(args.teacher_concurrency),
        '--initial-concurrent-per-server', str(args.teacher_initial_concurrency),
        '--pending-requests', str(args.teacher_pending_requests),
        '--max-auto-pending-requests', str(args.teacher_max_auto_pending_requests)]
    if args.teacher_concurrency_file:
        label += ['--concurrency-control-file', str(Path(args.teacher_concurrency_file).resolve())]
    if args.teacher_served_model:
        label += ['--teacher-served-model', args.teacher_served_model]
    if args.cache_dir:
        label += ['--cache-dir', args.cache_dir]
    train = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
        '--nproc_per_node', str(len(training_gpus)), '--module', 'colbert_experiments', 'train', *common, *model,
        '--labels', str(output / 'teacher/labels.jsonl'), '--output', str(output / 'fit'),
        '--steps', str(args.steps), '--accumulation', str(effective_batch // len(training_gpus)), '--eval-every', str(args.eval_every)]
    if args.objective == 'gated_regression':
        mine += ['--objective', args.objective]
        label += ['--gated-regression']
        train[train.index('--labels') + 1] = str(output / 'teacher/scores.jsonl')
        train += ['--objective', args.objective, '--documents-per-query', str(args.documents_per_query),
                  '--prediction-link', args.prediction_link]
    with (output / 'round.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        commands = dict(mining=mine, labeling=label, training=train)
        path = output / 'commands.json'
        if path.exists() and json.loads(path.read_text()) != commands:
            previous = json.loads(path.read_text())
            if not compatible_commands(previous, commands,
                    training_started=(output / 'fit/training_manifest.json').exists()):
                raise ValueError('Round arguments changed; use a new output directory')
            history_path = output / 'execution_history.json'
            history = json.loads(history_path.read_text()) if history_path.exists() else []
            history.append(previous)
            write_json(history_path, history)
        write_json(path, commands)
        write_json(output / 'execution_devices.json', dict(mining_gpus=gpus, training_gpus=training_gpus,
                   effective_query_group_batch=effective_batch))
        for stage, command in commands.items():
            if stage == 'training':
                if args.stop_teacher_before_training:
                    from .local_teacher import stop_registered_teacher
                    write_json(output / 'status.json', dict(state='releasing_teacher_gpu', stage=stage))
                    stop_registered_teacher(args.stop_teacher_before_training,
                        output / 'teacher_shutdown.json', training_gpus=training_gpus)
                while busy := busy_training_gpus(training_gpus):
                    write_json(output / 'status.json', dict(state='waiting_for_gpus', stage=stage, busy=busy))
                    time.sleep(30)
            write_json(output / 'status.json', dict(state='running', stage=stage))
            try:
                with (output / (stage + '.log')).open('a') as log:
                    stage_env = {**env, 'CUDA_VISIBLE_DEVICES': ','.join(training_gpus)} if stage == 'training' else env
                    subprocess.run(command, env=stage_env, cwd=Path(__file__).resolve().parents[1],
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
            except subprocess.CalledProcessError:
                write_json(output / 'status.json', dict(state='failed', stage=stage))
                raise
        write_json(output / 'status.json', dict(state='complete', model=str(output / 'fit/model')))


if __name__ == '__main__':
    main()
