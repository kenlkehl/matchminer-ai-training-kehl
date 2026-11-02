"""Rounds 2 and 3 of an EmbeddingGemma 2 TrialSpace run, after its round-1 labeling.

The configuration holds the run's train_from_summaries stage commands (see
README "Training"), remapped onto the run's round-1 directories. For each
round r in 1, 2:

1. project-trialspace-r on this host (labels -> pair table);
2. on the GB300 (scripts/gb300_round_job.py over SSH): stop its vLLM teacher,
   fit train-trialspace-r, mine round r+1, restart the teacher;
3. label-trialspace-(r+1) on this host with the live registry, plus a cloud
   labeling fleet (scripts/gcp_note_fleet.py, labeling mode) that watches this
   workflow's status and stops the VMs when label inference ends.

Round 1's labeler and fleet are only observed. Touch STOP in the status
directory to fail the workflow (the fleet then stops its VMs).
"""
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def start_ticks(pid):
    try:
        return int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19])
    except (OSError, IndexError, ValueError):
        return None


def complete(paths):
    return all(Path(p).exists() for p in paths)


class Workflow:
    def __init__(self, config, path):
        self.config, self.path = config, path
        self.root = Path(config['status']).parent
        self.status = json.loads(Path(config['status']).read_text()) if Path(config['status']).exists() else {}
        self.env = dict(os.environ, **config['environment'], PYTHONUNBUFFERED='1')
        self.watch = None

    def update(self, phase=None, **values):
        if phase:
            self.status['phase'] = phase
        self.status.update(values, updated_at=now(), pid=os.getpid(), start_ticks=start_ticks(os.getpid()))
        save(self.config['status'], self.status)

    def check(self):
        if (self.root / 'STOP').exists():
            raise RuntimeError('Operator STOP')

    def wait_process(self, pid, ticks, label, timeout=None):
        deadline = timeout and time.monotonic() + timeout
        while start_ticks(pid) == ticks:
            self.check()
            if deadline and time.monotonic() > deadline:
                raise TimeoutError(f'{label} (pid {pid}) is still running')
            time.sleep(30)

    def vms_stopped(self, fleet_dir, timeout=3600):
        """The fleet records shutdown_complete.json once gcloud reports every VM TERMINATED."""
        record, deadline = Path(fleet_dir) / 'shutdown_complete.json', time.monotonic() + timeout
        while not (record.exists() and json.loads(record.read_text()).get('all_terminated') is True):
            if time.monotonic() > deadline:
                raise TimeoutError(f'Cloud VMs not confirmed TERMINATED in {fleet_dir}')
            time.sleep(60)

    def wait_round1(self):
        first = self.config['round1']
        self.update('waiting_round1_labels')
        self.wait_process(first['labeler_pid'], first['labeler_start_ticks'], 'round-1 labeler')
        labels = Path(first['labels'])
        if not complete([labels / 'storage_complete.json', labels / 'scores.jsonl', labels / 'labeling_manifest.json']):
            raise RuntimeError('Round-1 labeling exited without publishing complete labels')
        self.update('waiting_round1_fleet')
        self.wait_process(first['fleet_pid'], first['fleet_start_ticks'], 'round-1 fleet controller', 3600)
        self.vms_stopped(first['fleet_dir'])

    def local(self, name):
        stage = self.config['stages'][name]
        if complete(stage['outputs']):
            return
        log = Path(self.config['log_dir']) / f'{name}.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('ab') as stream:
            child = subprocess.Popen(stage['command'], cwd=self.config['repo'], env=self.env,
                                     stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
            self.update(name, child_pid=child.pid, **{name: dict(state='running', started_at=now(), log=str(log))})
            while child.poll() is None:
                self.check()
                time.sleep(30)
        if child.returncode != 0 or not complete(stage['outputs']):
            raise RuntimeError(f'{name} failed ({child.returncode}); see {log}')
        self.update(child_pid=None, **{name: dict(state='complete', finished_at=now(), log=str(log))})

    def gb300(self, round_number):
        name = f'gb300-round{round_number}'
        status_path = Path(self.config['gb300']['status_dir']) / f'gb300_round{round_number}.json'
        read = lambda: json.loads(status_path.read_text()) if status_path.exists() else {}
        if read().get('state') == 'complete':
            return
        ssh = self.config['ssh']
        job = (f"cd {self.config['repo']} && setsid nohup {self.config['gb300_python']} scripts/gb300_round_job.py "
               f"--config {self.path} --round {round_number} >> {self.config['log_dir']}/{name}.log 2>&1 < /dev/null &")
        while True:  # Until SSH to the GB300 works (it needs a known host key and key login).
            self.check()
            try:
                launched = subprocess.run([*ssh, job], timeout=120, capture_output=True, text=True)
            except subprocess.TimeoutExpired as error:
                launched = subprocess.CompletedProcess(error.cmd, 255, '', 'timeout')
            if launched.returncode == 0:
                break
            self.update(f'{name}-waiting-for-ssh', ssh_error=launched.stderr[-500:], ssh_checked_at=now())
            time.sleep(300)
        self.update(name, child_pid=None, ssh_error=None, **{name: dict(state='started', at=now())})
        misses = 0
        while True:
            self.check()
            time.sleep(60)
            status = read()
            if status.get('state') == 'complete':
                break
            if status.get('state') == 'failed':
                raise RuntimeError(f'{name} failed: {status.get("error")}')
            if status.get('job_pid'):
                probe = subprocess.run([*ssh, f"kill -0 {status['job_pid']}"], timeout=120, capture_output=True)
                if probe.returncode == 255:  # SSH itself failed; ask again next minute.
                    continue
                misses = 0 if probe.returncode == 0 else misses + 1
                if misses >= 3 and read().get('state') not in ('complete', 'failed'):
                    raise RuntimeError(f'{name} job {status["job_pid"]} exited without completing')
        self.update(**{name: dict(state='complete', at=now())})

    def label(self, round_number):
        name = f'label-trialspace-{round_number}'
        stage = self.config['stages'][name]
        labels = Path(stage['command'][stage['command'].index('--output') + 1])
        if complete([*stage['outputs'], labels / 'storage_complete.json']):
            return
        # A new labeler applies the current tuning revision once; drop round-specific pins.
        control_path = Path(self.config['label_concurrency'])
        control = json.loads(control_path.read_text())
        if control.pop('tuning', None) is not None:
            save(control_path, control)
        log = Path(self.config['log_dir']) / f'{name}.log'
        with log.open('ab') as stream:
            child = subprocess.Popen(stage['command'], cwd=self.config['repo'], env=self.env,
                                     stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
        self.update('labeling', child_pid=child.pid, label_round=round_number,
                    **{name: dict(state='running', pid=child.pid, started_at=now(), log=str(log))})
        fleet_dir = self.start_fleet(round_number, labels, child.pid)
        while child.poll() is None:
            self.check()
            time.sleep(60)
        if child.returncode != 0 or not complete([*stage['outputs'], labels / 'storage_complete.json']):
            raise RuntimeError(f'{name} failed ({child.returncode}); see {log}')
        save(self.watch, dict(phase='completed', child_pid=child.pid, at=now()))
        self.watch = None
        self.update(f'{name}-complete', child_pid=None, **{name: dict(state='complete', finished_at=now(), log=str(log))})
        fleet = self.status[f'fleet-round{round_number}']
        self.wait_process(fleet['pid'], fleet['start_ticks'], f'round-{round_number} fleet controller', 3600)
        self.vms_stopped(fleet_dir)

    def start_fleet(self, round_number, labels, labeler_pid):
        settings = self.config['fleet']
        fleet_dir = Path(settings['directory_prefix'] + str(round_number))
        fleet_dir.mkdir(parents=True, exist_ok=True)
        # The fleet stops at this round's label inference end, its 'completed', or 'failed'.
        self.watch = fleet_dir / 'labeler_watch.json'
        save(self.watch, dict(phase='labeling', child_pid=labeler_pid, at=now()))
        fleet = copy.deepcopy(json.loads(Path(settings['template']).read_text()))
        fleet.update(fleet_id=f"{fleet['fleet_id']}-r{round_number}", controller=settings['controller'],
                     purpose=settings['purpose'], workflow_mode='labeling',
                     workflow=dict(status=str(self.watch), pid=os.getpid(), start_ticks=str(start_ticks(os.getpid())),
                                   label_status=str(labels / 'status.json')))
        fleet.pop('launched_at', None)
        save(fleet_dir / 'config.json', fleet)
        with (fleet_dir / 'controller.log').open('ab') as stream:
            child = subprocess.Popen([settings['python'], '-I', settings['controller'], '--config', str(fleet_dir / 'config.json')],
                                     cwd=fleet_dir, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        self.update(**{f'fleet-round{round_number}': dict(pid=child.pid, start_ticks=start_ticks(child.pid),
                                                          directory=str(fleet_dir), started_at=now())})
        return fleet_dir

    def run(self):
        try:
            self.wait_round1()
            for round_number in (1, 2):
                self.local(f'project-trialspace-{round_number}')
                self.gb300(round_number)
                self.label(round_number + 1)
            self.update('completed', completed_at=now())
        except BaseException as error:
            if self.watch is not None:
                save(self.watch, dict(phase='failed', error=repr(error), at=now()))
            self.update('failed', error=repr(error), failed_at=now())
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    with open(Path(config['status']).parent / 'workflow.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        Workflow(config, args.config.resolve()).run()


if __name__ == '__main__':
    sys.exit(main())
