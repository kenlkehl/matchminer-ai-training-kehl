"""One TrialSpace round's GPU work on the GB300, while its vLLM teacher is stopped.

Started on the GB300 over SSH by scripts/trialspace_rounds_workflow.py, which
polls --status on shared storage. Steps (each skipped when its output exists):

1. Find the vLLM server listening on the configured port, record its command
   line, working directory, GPU UUID and /v1/models answer (never its
   environment), and stop it.
2. Fit the round's student (finetune_embedder.py), then mine the next round
   with it (trialspace_comparison.py dense-mine).
3. Restart the teacher with the same command, directory and environment and
   wait until /health and /v1/models answer as before. This also runs when a
   step fails, so a failed fit never leaves the GB300 without its teacher.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def start_ticks(pid):
    try:
        return int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19])
    except (OSError, IndexError, ValueError):
        return None


def descendants(pid):
    children = {}
    for proc in Path('/proc').iterdir():
        if proc.name.isdigit():
            try:
                parent = int((proc / 'stat').read_text().rsplit(')', 1)[1].split()[1])
            except (OSError, IndexError, ValueError):
                continue
            children.setdefault(parent, []).append(int(proc.name))
    family, stack = {pid}, [pid]
    while stack:
        for child in children.get(stack.pop(), []):
            if child not in family:
                family.add(child)
                stack.append(child)
    return family


def find_teacher(port):
    found = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = [part.decode() for part in (proc / 'cmdline').read_bytes().split(b'\0')[:-1]]
        except OSError:
            continue
        if 'serve' in argv and any(Path(part).name == 'vllm' for part in argv[:argv.index('serve')]):
            if (argv[argv.index('--port') + 1] if '--port' in argv else '8000') == str(port):
                found.append(int(proc.name))
    if len(found) > 1:
        raise RuntimeError(f'Several vLLM servers claim port {port}: {found}')
    return found[0] if found else None


def models(url):
    with urllib.request.urlopen(url + '/v1/models', timeout=20) as response:
        return [dict(id=m['id'], root=m.get('root'), max_model_len=m.get('max_model_len'))
                for m in json.load(response)['data']]


def healthy(url):
    try:
        with urllib.request.urlopen(url + '/health', timeout=10) as response:
            return response.status == 200
    except OSError:
        return False


def gpu_apps():
    rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory',
                                    '--format=csv,noheader,nounits'], text=True).splitlines()
    return [dict(pid=int(p), gpu=g.strip(), mib=int(m)) for p, g, m in (r.split(',') for r in rows if r.strip())]


class Job:
    def __init__(self, config, round_number):
        self.config, self.round = config, round_number
        self.status_path = Path(config['status_dir']) / f'gb300_round{round_number}.json'
        self.status = json.loads(self.status_path.read_text()) if self.status_path.exists() else {}
        self.url = f"http://127.0.0.1:{config['teacher_port']}"
        self.teacher_env = None

    def update(self, **values):
        self.status.update(values, updated_at=now(), job_pid=os.getpid())
        save(self.status_path, self.status)

    def release(self):
        pid = find_teacher(self.config['teacher_port'])
        if pid is None:
            if not self.status.get('teacher'):
                raise RuntimeError('No vLLM teacher found and none recorded; refusing to guess its command')
            return
        argv = [part.decode() for part in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')[:-1]]
        expected = self.config['teacher_model_root']
        served = models(self.url)
        if not any(m['root'] == expected or m['id'] == expected for m in served):
            raise RuntimeError(f'Port {self.config["teacher_port"]} does not serve {expected}: {served}')
        family = descendants(pid)
        uuids = sorted({app['gpu'] for app in gpu_apps() if app['pid'] in family})
        if len(uuids) != 1:
            raise RuntimeError(f'Expected the teacher on one GPU, found {uuids}')
        # Held in memory only, to restart the server exactly as it ran.
        self.teacher_env = dict(part.split('=', 1) for part in
                                Path(f'/proc/{pid}/environ').read_bytes().decode().split('\0') if '=' in part)
        self.update(phase='releasing_teacher', teacher=dict(pid=pid, start_ticks=start_ticks(pid), command=argv,
                    cwd=os.readlink(f'/proc/{pid}/cwd'), gpu_uuid=uuids[0], models=served, stopped_at=now()))
        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + 600
        while start_ticks(pid) == self.status['teacher']['start_ticks'] or any(
                app['gpu'] == uuids[0] for app in gpu_apps()):
            if time.monotonic() > deadline:
                raise RuntimeError('The GB300 GPU is still occupied after stopping the teacher: ' + repr(gpu_apps()))
            time.sleep(5)
        self.update(phase='teacher_stopped')

    def run_step(self, name, command, outputs):
        if all(Path(p).exists() for p in outputs):
            self.update(**{name: dict(state='complete', skipped=True)})
            return
        env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONPATH=os.pathsep.join(self.config['pythonpath']),
                   CUDA_VISIBLE_DEVICES=self.status['teacher']['gpu_uuid'],
                   PYTORCH_ALLOC_CONF='expandable_segments:True')
        log = Path(self.config['log_dir']) / f'gb300_round{self.round}_{name}.log'
        with log.open('ab') as stream:
            child = subprocess.Popen(command, cwd=self.config['repo'], env=env, stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            self.update(phase=name, **{name: dict(state='running', pid=child.pid, started_at=now(), log=str(log))})
            code = child.wait()
        if code != 0 or not all(Path(p).exists() for p in outputs):
            self.update(**{name: dict(state='failed', returncode=code, log=str(log), at=now())})
            raise RuntimeError(f'{name} failed with exit code {code}; see {log}')
        self.update(**{name: dict(state='complete', returncode=0, finished_at=now(), log=str(log))})

    def restore(self):
        teacher = self.status.get('teacher')
        if not teacher or healthy(self.url):
            return
        env = self.teacher_env
        if env is None:  # A restarted job: rebuild the virtualenv PATH and GPU only.
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=teacher['gpu_uuid'])
            env['PATH'] = str(Path(teacher['command'][0]).parent) + os.pathsep + env.get('PATH', os.defpath)
        log = Path(self.config['log_dir']) / f'gb300_teacher_after_round{self.round}.log'
        with log.open('ab') as stream:
            child = subprocess.Popen(teacher['command'], cwd=teacher['cwd'], env=env, stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        self.update(phase='restoring_teacher', restored=dict(pid=child.pid, started_at=now(), log=str(log)))
        deadline = time.monotonic() + 2400
        while not healthy(self.url):
            if child.poll() is not None:
                raise RuntimeError(f'The restarted teacher exited with {child.returncode}; see {log}')
            if time.monotonic() > deadline:
                raise TimeoutError('The restarted teacher did not become healthy in 40 minutes')
            time.sleep(15)
        if models(self.url) != teacher['models']:
            raise RuntimeError('The restarted teacher serves different models: ' + repr(models(self.url)))
        self.update(restored=dict(self.status['restored'], healthy_at=now()))

    def run(self):
        self.update(phase='starting', state='running', round=self.round, host=os.uname().nodename)
        steps = self.config['rounds'][str(self.round)]
        failure = None
        try:
            if not all(all(Path(p).exists() for p in steps[name]['outputs']) for name in ('train', 'mine')):
                self.release()
                for name in ('train', 'mine'):
                    self.run_step(name, steps[name]['command'], steps[name]['outputs'])
        except BaseException as error:  # Restore the teacher before reporting any failure.
            failure = error
        try:
            self.restore()
        except BaseException as error:
            failure = failure or error
            self.update(restore_error=repr(error))
        if failure is not None:
            self.update(state='failed', phase='failed', error=repr(failure), failed_at=now())
            raise failure
        self.update(state='complete', phase='complete', completed_at=now())


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--round', type=int, choices=(1, 2), required=True)
    args = parser.parse_args()
    # One job per GB300 at a time (a host-local lock; a second launch exits here).
    with open('/tmp/mmai-gb300-round-job.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        Job(json.loads(args.config.read_text())['gb300'], args.round).run()


if __name__ == '__main__':
    main()
