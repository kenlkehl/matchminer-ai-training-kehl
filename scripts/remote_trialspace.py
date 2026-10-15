"""Optional, durable GCP execution of a TrialSpace stage in a local pipeline.

Activation is explicit: .mmai_cache/remote_trialspace.json must name the exact
local output directory. Teachers and checkpointed training share the cloud GPUs
sequentially. The local pipeline waits and keeps ownership of stage completion.
"""
from datetime import timedelta
import json
import os
from pathlib import Path
import shlex
import subprocess
import time


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def remote_path(path, local_root, remote_root):
    return str(Path(remote_root) / Path(path).resolve().relative_to(Path(local_root).resolve()))


def remote_base_model(base_model, output, local_root, remote_root):
    """Keep refinement weights on the cloud persistent disk, in the same run."""
    output = Path(output).resolve()
    if output.name == 'trialspace_round1' and base_model == 'google/embeddinggemma-2':
        return base_model
    previous = output.parent / 'trialspace_round1'
    if output.name != 'trialspace_round2' or Path(base_model).resolve() != previous:
        raise ValueError('Round 2 must initialize from this run\'s trained round 1 model')
    if not (previous / 'modules.json').is_file():
        raise ValueError('The completed round 1 model is missing')
    return remote_path(previous, local_root, remote_root)


def project_labels(source, target):
    """Copy only the unchanged columns consumed by finetune_embedder.

    Teacher traces stay in the authoritative local labeling artifacts. No source
    patient/note/trial files are hashed or revalidated by this execution adapter.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(source)
    required = ['patient_summary', 'this_space', 'eligibility_result']
    missing = set(required) - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f'Labels lack training columns: {sorted(missing)}')
    columns = required + [name for name in ['split', 'patient_long_text']
                          if name in parquet.schema_arrow.names]
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.partial.parquet')
    rows = 0
    schema = pa.schema([parquet.schema_arrow.field(name) for name in columns])
    with pq.ParquetWriter(temporary, schema,
                         compression='zstd') as writer:
        for batch in parquet.iter_batches(batch_size=8192, columns=columns):
            writer.write_table(pa.Table.from_batches([batch]).replace_schema_metadata(None))
            rows += batch.num_rows
    temporary.replace(target)
    return {'source': str(source), 'rows': rows, 'columns': columns,
            'bytes': target.stat().st_size}


class Cloud:
    def __init__(self, config):
        self.config = config
        self.host = config['ssh_user'] + '@' + config['vm']
        self.flags = ['--project=' + config['project'], '--zone=' + config['zone'],
                      '--quiet', '--ssh-flag=-oBatchMode=yes',
                      '--ssh-flag=-oConnectTimeout=30']

    def ssh(self, command, timeout=120):
        result = subprocess.run(['/usr/bin/gcloud', 'compute', 'ssh', self.host,
                                 *self.flags, '--command=' + command],
                                capture_output=True, text=True, timeout=timeout)
        if result.returncode:
            raise RuntimeError(result.stderr[-2000:])
        return result.stdout

    def scp(self, source, destination, recursive=False):
        flags = ['--project=' + self.config['project'], '--zone=' + self.config['zone'],
                 '--quiet', '--scp-flag=-oBatchMode=yes', '--scp-flag=-oConnectTimeout=30']
        command = ['/usr/bin/gcloud', 'compute', 'scp', *flags]
        if recursive:
            command.append('--recurse')
        command += [str(source), str(destination)]
        subprocess.run(command, check=True, timeout=3600)

    def retry(self, operation, description):
        failures = 0
        while True:
            try:
                return operation()
            except (RuntimeError, subprocess.SubprocessError, OSError) as exc:
                failures += 1
                print(f'[cloud] {description} unavailable; waiting for Spot recovery '
                      f'(attempt {failures}): {str(exc)[-500:]}', flush=True)
                if failures >= 1440:
                    raise RuntimeError('Cloud unavailable for more than a day') from exc
                time.sleep(60)


def execute(args, config, root):
    cloud = Cloud(config)
    local_parent = root.parent
    remote_root = config['remote_root']
    output = Path(args.output_model).resolve()
    base_model = remote_base_model(args.base_model, output, local_parent, remote_root)
    job = output.name
    operation = config['run_id'] + '/' + job
    job_dir = remote_root + '/cloud_workers/jobs/' + job
    local_job = root / '.mmai_cache/remote_trialspace' / job
    local_job.mkdir(parents=True, exist_ok=True)
    control_path = root / '.mmai_cache/local-monitor/gcp_pool_config.json'
    control = json.loads(control_path.read_text())
    atomic_json(control_path, {**control, 'enabled': False, 'training_job': job})
    try:
        probe = ('from pathlib import Path; import json; p=Path(' + repr(job_dir)
                 + '); print(json.dumps({n:json.loads((p/(n+".json")).read_text()) '
                 + 'if (p/(n+".json")).exists() else None for n in ["status","spec"]}))')
        previous = json.loads(cloud.retry(lambda: cloud.ssh('python3 -c ' + shlex.quote(probe)),
                                         'existing job inspection'))
        prior_state = previous.get('status')
        if prior_state and prior_state.get('operation') != operation:
            raise RuntimeError('Existing cloud job belongs to another run')
        resuming = bool(prior_state and prior_state['status'] in ['queued', 'running', 'complete'])
        if resuming:
            spec = previous['spec']
            if (spec['operation'] != operation or spec['execution_overlay'] != config['execution_overlay']
                    or spec['base_model'] != base_model
                    or spec['training_commit'] != config['training_commit']
                    or spec['inference_commit'] != config['inference_commit']
                    or [p['source'] for p in spec['projection']] !=
                       [str(Path(p).resolve()) for p in args.input_parquet]):
                raise RuntimeError('Existing cloud job has a different execution configuration')
            print('[cloud] Reusing durable job and uploaded labels after interruption', flush=True)
        else:
            projected = []
            inputs = []
            for index, source in enumerate(args.input_parquet or []):
                source = Path(source).resolve()
                source.relative_to(Path(config['run_dir']).resolve())
                target = local_job / f'labels-{index}.parquet'
                projected.append(project_labels(source, target))
                inputs.append(job_dir + '/' + target.name)
            if not inputs:
                raise ValueError('Remote training requires explicit label artifacts')
            spec = {'operation': operation, 'job': job, 'remote_root': remote_root,
                    'inputs': inputs, 'output': remote_path(output, local_parent, remote_root),
                    'checkpoints': remote_path(args.ckpt_dir, local_parent, remote_root),
                    'base_model': base_model, 'gpu_count': 8, 'save_steps': 500,
                    'training_commit': config['training_commit'],
                    'inference_commit': config['inference_commit'], 'projection': projected,
                    'execution_overlay': config['execution_overlay']}
        atomic_json(local_job / 'spec.json', spec)
        if not resuming:
            cloud.retry(lambda: cloud.ssh('mkdir -p ' + shlex.quote(job_dir)), 'job directory')
            for target in [*[local_job / Path(path).name for path in inputs], local_job / 'spec.json']:
                cloud.retry(lambda p=target: cloud.scp(p, cloud.host + ':' + job_dir + '/'),
                            'upload ' + target.name)
        driver = remote_root + '/matchminer-ai-training-kehl/scripts/cloud_trialspace_job.py'
        cloud.retry(lambda: cloud.ssh('/home/klkehl/thisenv/bin/python '
                    + shlex.quote(driver) + ' submit --spec ' + shlex.quote(job_dir + '/spec.json')),
                    'training submission')
        while True:
            text = cloud.retry(lambda: cloud.ssh('cat ' + shlex.quote(job_dir + '/status.json')),
                               'training status')
            state = json.loads(text)
            if state.get('operation') != operation:
                raise RuntimeError('Cloud training receipt belongs to another run')
            atomic_json(local_job / 'remote_status.json', state)
            print('[cloud] ' + json.dumps(state), flush=True)
            if state['status'] == 'complete':
                break
            if state['status'] == 'failed':
                raise RuntimeError('Cloud TrialSpace training failed: ' + state.get('error', 'see remote log'))
            time.sleep(30)
        if output.exists():
            # Recover a crash after the atomic copy-back but before the local
            # runner wrote its stage marker. Existing artifacts are verified,
            # never overwritten.
            copied = output
        else:
            staging = local_job / 'download'
            staging.mkdir(exist_ok=True)
            cloud.retry(lambda: cloud.scp(cloud.host + ':' + spec['output'], staging, recursive=True),
                        'trained model download')
            copied = staging / output.name
        for name, size in state['model_files'].items():
            path = copied / name
            if not path.is_file() or path.stat().st_size != size:
                raise RuntimeError('Incomplete trained model download: ' + name)
        if not (copied / 'modules.json').is_file():
            raise RuntimeError('Downloaded model lacks SentenceTransformer modules')
        output.parent.mkdir(parents=True, exist_ok=True)
        if copied != output:
            copied.replace(output)
        atomic_json(local_job / 'receipt.json', {**spec, 'status': 'copied_back',
                                               'remote_status': state})
        print('[cloud] Verified trained artifact copied back to ' + str(output), flush=True)
    finally:
        latest = json.loads(control_path.read_text())
        atomic_json(control_path, {**latest, 'enabled': bool(latest.get('desired_running')), 'training_job': None})


def maybe_run_remote(args, root):
    root = Path(root)
    config_path = root / '.mmai_cache/remote_trialspace.json'
    if not config_path.is_file():
        return False
    config = json.loads(config_path.read_text())
    if not config.get('enabled'):
        return False
    if not Path(args.output_model).resolve().is_relative_to(Path(config['models_dir']).resolve()):
        return False
    # The existing pipeline launches two local ranks. They coordinate on CPU;
    # only rank zero uploads data and waits for the eight-GPU remote job.
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    distributed = None
    if world_size > 1:
        import torch.distributed as distributed
        distributed.init_process_group('gloo', timeout=timedelta(days=2))
    outcome = [None]
    try:
        if rank == 0:
            try:
                execute(args, config, root)
                outcome[0] = {'ok': True}
            except Exception as exc:
                outcome[0] = {'ok': False, 'error': str(exc)}
        if distributed is not None:
            distributed.broadcast_object_list(outcome, src=0)
        if not outcome[0]['ok']:
            raise RuntimeError(outcome[0]['error'])
    finally:
        if distributed is not None:
            distributed.destroy_process_group()
    return True
