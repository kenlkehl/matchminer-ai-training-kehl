"""Durable eight-GPU TrialSpace job on the existing persistent-disk Spot VM."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def atomic(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def teachers(action):
    subprocess.run(['sudo', 'systemctl', *action,
                    *[f'mmai-cloud-teacher@{i}.service' for i in range(8)]], check=True)


def validate(spec):
    root = Path('/home/klkehl/mmai')
    if spec['remote_root'] != str(root):
        raise ValueError('Unexpected cloud training configuration')
    if spec['job'] not in ['trialspace_round1', 'trialspace_round2'] or spec['gpu_count'] != 8:
        raise ValueError('Unexpected TrialSpace stage')
    if not Path(spec['output']).is_relative_to(root / 'models/gemma_pipeline_response_v3'):
        raise ValueError('Unexpected model output directory')
    if not Path(spec['checkpoints']).is_relative_to(root / 'models/gemma_pipeline_response_v3'):
        raise ValueError('Unexpected checkpoint directory')
    expected_base = ('google/embeddinggemma-2' if spec['job'] == 'trialspace_round1'
                     else str(Path(spec['output']).parent / 'trialspace_round1'))
    if spec['base_model'] != expected_base:
        raise ValueError('Round 2 must initialize from the trained round 1 model')
    if len(spec['inputs']) != 2 or len(spec['projection']) != 2:
        raise ValueError('Each fit requires only its own two directional label files')
    round_number = spec['job'][-1]
    expected_labels = {f'top_{direction}_tocheck_round{round_number}_labels'
                       for direction in ('cohorts', 'patients')}
    if {Path(item['source']).parent.name for item in spec['projection']} != expected_labels:
        raise ValueError('Training labels belong to another refinement round')
    for path in spec['inputs']:
        if not Path(path).is_relative_to(root / 'cloud_workers/jobs' / spec['job']):
            raise ValueError('Unexpected training label input')


def submit(spec_path, spec):
    unit_name = 'mmai-cloud-' + spec['job'] + '.service'
    status_path = spec_path.parent / 'status.json'
    if status_path.exists():
        status = json.loads(status_path.read_text())
        if status.get('operation') != spec['operation']:
            raise ValueError('Refusing to reuse a job directory from another run')
        if status['status'] == 'complete':
            print(json.dumps(status));return
        active = subprocess.run(['systemctl', 'is-active', '--quiet', unit_name]).returncode == 0
        if active:
            print('Existing remote training job is active');return
    unit = f'''[Unit]
Description=Checkpointed cloud {spec['job']}
After=network-online.target

[Service]
Type=simple
User=klkehl
WorkingDirectory=/home/klkehl/mmai/matchminer-ai-training-kehl
Environment=HF_HOME=/home/klkehl/mmai/cache/huggingface
Environment=PATH=/home/klkehl/thisenv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
Environment=PYTHONPATH=/home/klkehl/mmai/matchminer-ai-inference-kehl/src:/home/klkehl/mmai/matchminer-ai-training-kehl
Environment=TOKENIZERS_PARALLELISM=false
Environment=OMP_NUM_THREADS=8
Environment=PYTHONUNBUFFERED=1
ExecStart=/home/klkehl/thisenv/bin/python /home/klkehl/mmai/matchminer-ai-training-kehl/scripts/cloud_trialspace_job.py run --spec {spec_path}
Restart=no
TimeoutStopSec=120
KillMode=control-group
LimitNOFILE=8192:1048576
StandardOutput=append:{spec_path.parent}/training.log
StandardError=append:{spec_path.parent}/training.log

[Install]
WantedBy=multi-user.target
'''
    path = spec_path.parent / unit_name
    path.write_text(unit)
    subprocess.run(['sudo', 'install', '-m', '644', str(path), '/etc/systemd/system/' + unit_name], check=True)
    # Enabled on boot so a Spot stop resumes the checkpointed job. Teachers stay
    # disabled while training owns all eight GPUs, including across a reboot.
    teachers(['disable', '--now'])
    atomic(status_path, {'status': 'queued', 'operation': spec['operation'], 'job': spec['job']})
    subprocess.run(['sudo', 'systemctl', 'daemon-reload'], check=True)
    subprocess.run(['sudo', 'systemctl', 'enable', '--now', unit_name], check=True)
    print('Submitted ' + unit_name, flush=True)


def run(spec_path, spec):
    root = Path(spec['remote_root'])
    status_path = spec_path.parent / 'status.json'
    if status_path.exists():
        previous = json.loads(status_path.read_text())
        if previous.get('operation') != spec['operation']:
            raise ValueError('Existing job status belongs to another run')
        if previous['status'] == 'complete':
            output = Path(spec['output'])
            if all((output / name).is_file() and (output / name).stat().st_size == size
                   for name, size in previous['model_files'].items()):
                # Recover a Spot stop after saving the model/receipt but before
                # reenabling teachers and disabling this boot-resume unit.
                teachers(['enable', '--now'])
                subprocess.run(['sudo', 'systemctl', 'disable', 'mmai-cloud-' + spec['job'] + '.service'], check=True)
                return 0
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    teachers(['disable', '--now'])
    atomic(status_path, {'status': 'running', 'operation': spec['operation'],
                        'job': spec['job'], 'started_at': started,
                        'checkpoint_interval_steps': spec['save_steps']})
    outcome = 1
    try:
        if spec['job'] == 'trialspace_round2':
            previous_model = Path(spec['base_model'])
            previous_config = json.loads((previous_model / 'config.json').read_text())
            if (not (previous_model / 'modules.json').is_file()
                    or previous_config.get('model_type') != 'embedding_gemma2'
                    or previous_config.get('vision_config') is not None
                    or previous_config.get('audio_config') is not None):
                raise ValueError('Round 1 is not a complete text-only EmbeddingGemma 2 model')
        import pyarrow.parquet as pq
        for path, transfer in zip(spec['inputs'], spec['projection']):
            if pq.ParquetFile(path).metadata.num_rows != transfer['rows']:
                raise ValueError('Incomplete transferred label file')
        for repo, revision in [('matchminer-ai-training-kehl', spec['training_commit']),
                                ('matchminer-ai-inference-kehl', spec['inference_commit'])]:
            head = subprocess.check_output(['git', '-C', str(root / repo), 'rev-parse', 'HEAD'], text=True).strip()
            if head != revision:
                raise ValueError('Cloud code revision differs from the approved run')
        for filename, checksum in spec['execution_overlay'].items():
            deployed = root / 'matchminer-ai-training-kehl' / filename
            if hashlib.sha256(deployed.read_bytes()).hexdigest() != checksum:
                raise ValueError('Cloud training execution code differs: ' + filename)
        environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7'}
        command = ['/home/klkehl/thisenv/bin/python', '-m', 'torch.distributed.run',
                   '--standalone', '--nproc_per_node=8', 'finetune_embedder.py',
                   '--base-model', spec['base_model'], '--ckpt-dir', spec['checkpoints'],
                   '--output-model', spec['output'], '--save-steps', str(spec['save_steps'])]
        for path in spec['inputs']:
            command += ['--input-parquet', path]
        subprocess.run(command, cwd=root / 'matchminer-ai-training-kehl', env=environment, check=True)
        output = Path(spec['output'])
        files = {str(p.relative_to(output)): p.stat().st_size for p in output.rglob('*') if p.is_file()}
        if 'modules.json' not in files or not any(name.endswith('.safetensors') for name in files):
            raise ValueError('Training did not produce a complete model artifact')
        config = json.loads((output / 'config.json').read_text())
        if config.get('model_type') != 'embedding_gemma2' or config.get('vision_config') is not None or config.get('audio_config') is not None:
            raise ValueError('Training did not produce the approved text-only model')
        atomic(status_path, {'status': 'complete', 'operation': spec['operation'],
                            'job': spec['job'], 'started_at': started,
                            'completed_at': dt.datetime.now(dt.timezone.utc).isoformat(),
                            'model_files': files})
        outcome = 0
    except Exception as exc:
        atomic(status_path, {'status': 'failed', 'operation': spec['operation'],
                            'job': spec['job'], 'error': str(exc)})
        raise
    finally:
        # This finally block is intentionally NOT executed on a Spot VM stop;
        # the enabled job and disabled teachers then survive on persistent disk.
        teachers(['enable', '--now'])
        subprocess.run(['sudo', 'systemctl', 'disable', 'mmai-cloud-' + spec['job'] + '.service'], check=True)
    return outcome


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['submit', 'run'])
    parser.add_argument('--spec', type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    validate(spec)
    if args.action == 'submit':
        submit(args.spec, spec)
    else:
        sys.exit(run(args.spec, spec))
