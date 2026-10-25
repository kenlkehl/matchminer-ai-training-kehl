"""Durable notes → summaries → cloud shutdown → local mining → teacher labels."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.gcp_note_fleet import Fleet, now, process_identity, save, ensure_ssh_descriptor_capacity
from colbert_experiments.local_teacher import (process_identity as teacher_identity,
                                               open_pidfd, send_pidfd_signal)


def generation_ready(status, alive):
    if status.get('stage') == 'failed':
        raise RuntimeError('Note generation hard-failed')
    if status.get('stage') == 'completed':
        return True
    if not alive:
        raise RuntimeError('Note generator exited before completing its outputs')
    return False


def stop_teacher(target, receipt):
    """A PID/start-time/command match is mandatory before signaling a server."""
    expected = target['process']
    if str(target['gpu']) not in ('2', '3'):
        raise ValueError('This workflow may release only Camus GPUs 2 and 3')
    current = teacher_identity(expected['pid'])
    if current is None:
        save(receipt, dict(state='already_stopped', target=target))
        return
    fd = open_pidfd(expected['pid'])
    try:
        if teacher_identity(expected['pid']) != expected:
            raise ValueError('Registered vLLM process changed; refusing to signal it')
        command = current['command']
        if 'serve' not in command or '--port' not in command or command[command.index('--port') + 1] != str(target['port']):
            raise ValueError('Registered process is not the expected vLLM server')
        environment = dict(x.split('=', 1) for x in Path(f'/proc/{expected["pid"]}/environ').read_bytes().decode().split('\0') if '=' in x)
        if environment.get('CUDA_VISIBLE_DEVICES') != str(target['gpu']):
            raise ValueError('Registered vLLM GPU changed')
        send_pidfd_signal(fd, signal.SIGTERM)
        save(receipt, dict(state='termination_requested', at=now(), target=target))
    finally:
        os.close(fd)


def busy_gpus(selected):
    lines = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True).splitlines()
    indices = {uuid.strip(): index.strip() for index, uuid in (line.split(',', 1) for line in lines)}
    rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True).splitlines()
    return [dict(gpu=indices.get(uuid.strip()), pid=int(pid.strip())) for uuid, pid in (line.split(',', 1) for line in rows)
            if indices.get(uuid.strip()) in selected]


def healthy_teacher(target):
    try:
        base = f'http://127.0.0.1:{target["port"]}'
        with urllib.request.urlopen(base + '/health', timeout=5) as response:
            response.read()
        with urllib.request.urlopen(base + '/v1/models', timeout=5) as response:
            models = json.load(response)['data']
        return any(m.get('root', m['id']) == target['model_root'] and m.get('max_model_len', 0) >= 250000 for m in models)
    except Exception:
        return False


class Workflow:
    def __init__(self, config, directory):
        self.config, self.root = config, Path(directory)
        self.root.mkdir(parents=True, exist_ok=True)
        self.status = dict(pid=os.getpid(), started_at=now(), phase='waiting_notes')
        self.child = None
        self.owned_teachers = {}
        self.cancelled = False

    def update(self, phase=None, **values):
        self.status.update(values, updated_at=now())
        if phase:
            self.status['phase'] = phase
        save(self.root / 'status.json', self.status)

    def interrupted(self):
        if self.cancelled or (self.root / 'STOP').exists():
            raise RuntimeError('Post-note workflow stopped by operator')

    def wait_notes(self):
        while True:
            self.interrupted()
            status = json.loads(Path(self.config['notes_status']).read_text())
            alive = process_identity(self.config['generator_pid']) == self.config['generator_start_ticks']
            if generation_ready(status, alive):
                return
            self.update('waiting_notes', saved_notes=status.get('stage_completed'), expected_notes=status.get('expected_notes'))
            time.sleep(10)

    def run_stage(self, name):
        stage = self.config['stages'][name]
        receipt = self.root / 'completed' / (name + '.json')
        if receipt.exists():
            prior = json.loads(receipt.read_text())
            if prior['command'] != stage['command'] or not all(Path(p).exists() for p in stage['outputs']):
                raise ValueError('Completed workflow stage changed: ' + name)
            return
        self.update(name, child_pid=None)
        log = self.root / 'logs' / (name + '.log')
        log.parent.mkdir(exist_ok=True)
        env = {**os.environ, **self.config.get('environment', {}), **stage.get('environment', {})}
        with log.open('ab') as stream:
            self.child = subprocess.Popen(stage['command'], cwd=self.config['repo'], env=env,
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            self.update(child_pid=self.child.pid, command=stage['command'], log=str(log))
            while self.child.poll() is None:
                self.interrupted()
                if name == 'labeling':
                    self.restart_dead_teachers()
                self.update()
                time.sleep(10)
            if self.child.returncode:
                raise RuntimeError(f'{name} exited with code {self.child.returncode}; see {log}')
        self.child = None
        if not all(Path(p).exists() for p in stage['outputs']):
            raise RuntimeError('Missing outputs after ' + name)
        save(receipt, dict(at=now(), command=stage['command'], outputs=stage['outputs']))
        self.update(child_pid=None)

    def stop_cloud(self):
        self.update('stopping_cloud')
        config_path = Path(self.config['fleet_config'])
        config = json.loads(config_path.read_text())
        fleet_root = config_path.parent
        # The attached fleet supervisor joins pending maintenance before stopping
        # VMs. If it exited unexpectedly, take over the bounded allowlist cleanup.
        while True:
            self.interrupted()
            complete = fleet_root / 'shutdown_complete.json'
            if complete.exists() and json.loads(complete.read_text()).get('all_terminated'):
                return
            report = json.loads((fleet_root / 'status.json').read_text())
            if process_identity(report['supervisor_pid']) is None:
                Fleet(config, fleet_root).shutdown('post_notes_workflow_cleanup')
                return
            time.sleep(10)

    def release_gpus(self):
        self.update('releasing_gpus')
        for target in self.config['local_teachers']:
            stop_teacher(target, self.root / f'stopped_gpu_{target["gpu"]}.json')
        deadline = time.monotonic() + 180
        while busy := busy_gpus(['2', '3']):
            self.interrupted()
            if time.monotonic() >= deadline:
                raise RuntimeError('GPUs 2 and 3 are still occupied: ' + str(busy))
            time.sleep(5)

    def launch_teacher(self, target):
        gpu = str(target['gpu'])
        if busy_gpus([gpu]):
            raise RuntimeError('Refusing to start a second server on busy GPU ' + gpu)
        log = self.root / 'logs' / f'label_teacher_gpu_{gpu}.log'
        env = {**os.environ, **target.get('environment', {}), 'CUDA_VISIBLE_DEVICES': gpu}
        with log.open('ab') as stream:
            process = subprocess.Popen(target['command'], cwd=self.config['repo'], env=env,
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        self.owned_teachers[gpu] = process
        save(self.root / f'label_teacher_gpu_{gpu}.json', dict(pid=process.pid, at=now(), gpu=gpu, command=target['command']))

    def restart_dead_teachers(self):
        for target in self.config['local_teachers']:
            process = self.owned_teachers.get(str(target['gpu']))
            if process is not None and process.poll() is not None:
                self.launch_teacher(target)

    def start_labelers(self):
        self.update('starting_labelers')
        for target in self.config['local_teachers']:
            if not healthy_teacher(target):
                self.launch_teacher(target)
        deadline = time.monotonic() + 1800
        while not all(healthy_teacher(target) for target in self.config['local_teachers']):
            self.interrupted()
            self.restart_dead_teachers()
            if time.monotonic() >= deadline:
                raise TimeoutError('Local label teachers did not become healthy')
            time.sleep(10)
        # Do not rewrite this registry during labeling; additional authorized
        # servers can be admitted by editing it while the live pool is running.
        registry = Path(self.config['label_registry'])
        if not registry.exists():
            save(registry, dict(ready=True, servers=self.config['label_servers']))

    def run(self):
        def halt(signum, frame):
            self.cancelled = True
        signal.signal(signal.SIGTERM, halt)
        signal.signal(signal.SIGINT, halt)
        try:
            self.update()
            self.wait_notes()
            self.run_stage('summarizing')
            self.stop_cloud()
            self.release_gpus()
            self.run_stage('preparing_corpus')
            self.run_stage('mining')
            self.start_labelers()
            self.run_stage('labeling')
            self.update('completed', completed_at=now())
        except BaseException as error:
            if self.child is not None and self.child.poll() is None:
                os.killpg(self.child.pid, signal.SIGTERM)
            try:
                self.update('failed', error=repr(error))
            finally:
                config_path = Path(self.config['fleet_config'])
                # STOP also prevents a late pending Spot start from being retried.
                try:
                    (config_path.parent / 'STOP').touch()
                finally:
                    if not (config_path.parent / 'shutdown_complete.json').exists():
                        Fleet(json.loads(config_path.read_text()), config_path.parent).shutdown('downstream_workflow_failed')
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    ensure_ssh_descriptor_capacity()
    config = json.loads(args.config.read_text())
    with (args.config.parent / 'workflow.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        Workflow(config, args.config.parent).run()


if __name__ == '__main__':
    main()
