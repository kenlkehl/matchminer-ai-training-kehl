"""Wait for complete labels, release teachers, then run one bounded DDP fit.

This does not restart labeling or infer ownership from GPU use. Local servers
must match the workflow's exact commands, owner and CUDA device. A separate
watchdog and the training service's RuntimeMaxSec enforce the GPU deadline.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from colbert_experiments.local_teacher import process_identity, open_pidfd, send_pidfd_signal


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def timestamp(value):
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError('Deadline must have an explicit timezone offset')
    return parsed.timestamp()


def signal_owned(expected, sig):
    if process_identity(expected['pid']) != expected:
        return False
    try:
        fd = open_pidfd(expected['pid'])
    except ProcessLookupError:
        return False
    try:
        if process_identity(expected['pid']) != expected:
            return False
        send_pidfd_signal(fd, sig)
        return True
    except ProcessLookupError:
        return False
    finally:
        os.close(fd)


def user_processes():
    for folder in Path('/proc').iterdir():
        if not folder.name.isdigit():
            continue
        try:
            if folder.stat().st_uid == os.getuid():
                identity = process_identity(int(folder.name))
                if identity:
                    yield identity
        except (ProcessLookupError, FileNotFoundError, PermissionError):
            continue


def session_members(sid):
    found = []
    for identity in user_processes():
        try:
            if os.getsid(identity['pid']) == sid:
                found.append(identity)
        except ProcessLookupError:
            pass
    return found


def discover_teachers(workflow):
    targets = []
    for target in workflow['local_teachers']:
        matches = []
        for identity in user_processes():
            if identity['command'] != target['command']:
                continue
            try:
                environment = dict(item.split('=', 1) for item in
                    Path(f'/proc/{identity["pid"]}/environ').read_bytes().decode().split('\0') if '=' in item)
                if environment.get('CUDA_VISIBLE_DEVICES') != str(target['gpu']):
                    continue
                if os.getsid(identity['pid']) != identity['pid']:
                    raise ValueError('Teacher is not an isolated owned session')
                matches.append(dict(gpu=target['gpu'], port=target['port'], leader=identity,
                                    members=session_members(identity['pid'])))
            except (ProcessLookupError, FileNotFoundError):
                continue
        if len(matches) > 1:
            raise ValueError('Multiple teachers match one registered GPU/port')
        targets.extend(matches)
    return targets


def labels_ready(workflow, labels, receipt, expected_pairs):
    if workflow.get('phase') == 'failed' or labels.get('state') == 'failed':
        raise RuntimeError('Labeling hard-failed; training will not start')
    if workflow.get('phase') != 'completed' or labels.get('state') != 'complete' or not receipt:
        return False
    if labels.get('failures') or labels.get('scored_pairs') != expected_pairs or labels.get('total_pairs') != expected_pairs:
        raise ValueError('Completed labeling has failures or incomplete candidate coverage')
    if not all(Path(path).is_file() for path in receipt['outputs']):
        raise ValueError('Completed labeling outputs are missing')
    return True


class Handoff:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.config = read(self.path)
        self.root = self.path.parent
        self.workflow_root = Path(self.config['workflow_config']).parent
        self.workflow = read(self.config['workflow_config'])
        self.fleet_root = Path(self.config['fleet_config']).parent
        self.deadline = timestamp(self.config['deadline'])
        self.pause_at = timestamp(self.config['pause_at'])
        if self.pause_at >= self.deadline:
            raise ValueError('Graceful pause must precede the hard deadline')
        if sorted(str(t['gpu']) for t in self.workflow['local_teachers']) != ['0', '1', '2', '3']:
            raise ValueError('This handoff requires the four explicitly registered Camus teachers')

    def update(self, phase, **values):
        save(self.root / 'status.json', dict(at=now(), pid=os.getpid(), phase=phase, **values))

    def lock(self, name):
        stream = (self.root / (name + '.lock')).open('a')
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return stream

    def control(self, *args):
        return subprocess.run(['systemctl', '--user', *args], text=True, capture_output=True, timeout=20)

    def service(self):
        result = self.control('show', self.config['unit'], '--property=ActiveState,SubState,Result,ExecMainStatus')
        return dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)

    def cloud_stop(self, wait):
        (self.fleet_root / 'STOP').touch()
        # The fleet joins any pending Spot-start maintenance before stopping.
        # Only take over if the registered supervisor is gone.
        while True:
            done = read(self.fleet_root / 'shutdown_complete.json', {})
            if done.get('all_terminated'):
                return
            expected = self.config['fleet_process']
            if process_identity(expected['pid']) != expected:
                cleanup_unit = self.config['unit'].removesuffix('.service') + '-cloud-cleanup.service'
                state = self.control('show', cleanup_unit, '--property=ActiveState', '--value')
                if state.stdout.strip() not in ('active', 'activating'):
                    result = subprocess.run(['systemd-run', '--user', '--unit=' + cleanup_unit,
                        '--property=Restart=on-failure', '--property=RestartSec=20',
                        '--property=StandardOutput=append:' + str(self.root / 'cloud_cleanup.log'),
                        '--property=StandardError=append:' + str(self.root / 'cloud_cleanup.log'),
                        '--', sys.executable, str(Path(__file__).resolve()), '--config', str(self.path),
                        '--mode', 'cloud-cleanup'], text=True, capture_output=True, timeout=30)
                    if result.returncode:
                        raise RuntimeError('Cloud cleanup service failed: ' + result.stderr)
            if not wait:
                return
            if time.time() >= self.pause_at or (self.root / 'DEADLINE').exists():
                raise TimeoutError('Cloud shutdown has not completed before the training cutoff')
            time.sleep(10)

    def stop_local(self, grace=90):
        # Saved session membership permits cleanup of orphaned engines even if
        # their server parent already exited. Never signal an unregistered GPU PID.
        ledger_path = self.root / 'local_sessions.json'
        sessions = read(ledger_path, [])
        known = {s['leader']['pid'] for s in sessions}
        if not read(self.root / 'local_shutdown.json', {}).get('all_owned_processes_stopped'):
            sessions += [s for s in discover_teachers(self.workflow) if s['leader']['pid'] not in known]
        save(ledger_path, sessions)
        started = time.monotonic()
        while True:
            live = []
            for session in sessions:
                anchors = [p for p in session['members'] if process_identity(p['pid']) == p]
                if anchors:
                    members = session_members(session['leader']['pid'])
                    saved = {p['pid']: p for p in session['members']}
                    saved.update({p['pid']: p for p in members})
                    session['members'] = list(saved.values())
                    live.extend(p for p in session['members'] if process_identity(p['pid']) == p)
            save(ledger_path, sessions)
            if not live:
                save(self.root / 'local_shutdown.json', dict(at=now(), all_owned_processes_stopped=True))
                return
            sig = signal.SIGKILL if time.monotonic() - started >= grace else signal.SIGTERM
            for identity in live:
                signal_owned(identity, sig)
            if time.monotonic() - started >= grace + 20:
                raise RuntimeError('Owned teacher processes did not terminate')
            time.sleep(2)

    def check_gpu_availability(self):
        from scripts.post_notes_workflow import teacher_blockers
        blockers = [b for target in self.workflow['local_teachers'] for b in teacher_blockers(target)]
        if blockers:
            raise RuntimeError(f'Training GPUs have unexpected occupants: {blockers}')

    def launch_training(self):
        # Serialize launch with the independent deadline guard to prevent a late
        # launch after that guard has already released the GPUs.
        with self.lock('launch'):
            if time.time() >= self.pause_at or (self.root / 'DEADLINE').exists():
                self.update('paused_before_training', reason='Deadline reached before fit could start')
                return False
            if (self.root / 'training_launch.json').exists():
                return True
            runtime = int(self.deadline - time.time()) - 30
            command = ['systemd-run', '--user', '--unit=' + self.config['unit'],
                '--property=WorkingDirectory=' + self.config['repo'],
                '--property=RuntimeMaxSec=' + str(runtime), '--property=TimeoutStopSec=10',
                '--property=KillMode=control-group', '--property=SendSIGKILL=yes',
                '--property=StandardOutput=append:' + str(self.root / 'training.log'),
                '--property=StandardError=append:' + str(self.root / 'training.log')]
            command += ['--setenv=' + key + '=' + str(value) for key, value in self.config['environment'].items()]
            command += ['--', *self.config['training_command']]
            result = subprocess.run(command, text=True, capture_output=True, timeout=30)
            if result.returncode:
                raise RuntimeError('Training service launch failed: ' + result.stderr)
            save(self.root / 'training_launch.json', dict(at=now(), command=command,
                runtime_max_seconds=runtime, deadline=self.config['deadline'], output=result.stderr))
            return True

    def run(self):
        with self.lock('handoff'):
            try:
                if read(self.root / 'status.json', {}).get('phase') in ('complete', 'paused', 'failed'):
                    return
                while not (self.root / 'training_launch.json').exists():
                    if time.time() >= self.pause_at or (self.root / 'DEADLINE').exists():
                        self.update('paused_before_training')
                        return
                    status = read(self.workflow_root / 'status.json', {})
                    labels = read(self.config['label_status'], {})
                    receipt = read(self.workflow_root / 'completed/labeling.json')
                    if labels_ready(status, labels, receipt, self.config['expected_pairs']):
                        self.update('stopping_cloud')
                        self.cloud_stop(wait=True)
                        self.update('stopping_local_teachers')
                        self.stop_local()
                        self.check_gpu_availability()
                        if not self.launch_training():
                            return
                        break
                    expected = self.config['workflow_process']
                    if process_identity(expected['pid']) != expected:
                        raise RuntimeError('Label workflow exited without complete labeling outputs')
                    self.update('waiting_labels', scored_pairs=labels.get('scored_pairs'),
                                total_pairs=self.config['expected_pairs'])
                    time.sleep(15)
                while True:
                    unit = self.service()
                    fit = read(self.config['fit_status'], {})
                    if unit.get('ActiveState') not in ('active', 'activating', 'deactivating'):
                        if fit.get('state') == 'complete' and unit.get('ExecMainStatus') == '0':
                            self.update('complete', fit=fit, service=unit)
                        elif fit.get('state') == 'paused' or (self.root / 'DEADLINE').exists():
                            self.update('paused', fit=fit, service=unit)
                        else:
                            raise RuntimeError('Training service stopped unexpectedly: ' + json.dumps(unit))
                        return
                    self.update('training', fit=fit, service=unit)
                    time.sleep(15)
            except BaseException as error:
                self.update('failed', error=repr(error))
                self.cloud_stop(wait=False)
                self.control('stop', '--no-block', self.config['unit'])
                # Stop the workflow before its teachers so it cannot respawn them.
                signal_owned(self.config['workflow_process'], signal.SIGTERM)
                signal_owned(self.config['label_process'], signal.SIGTERM)
                time.sleep(12)
                signal_owned(self.config['workflow_process'], signal.SIGKILL)
                self.stop_local()
                raise

    def watchdog(self, immediate=False):
        with self.lock('enforcer' if immediate else 'watchdog'):
            save(self.root / ('enforcer_status.json' if immediate else 'watchdog_status.json'),
                dict(at=now(), pid=os.getpid(), state='armed', deadline=self.config['deadline'],
                     pause_at=self.config['pause_at']))
            if not immediate:
                while time.time() < self.pause_at:
                    time.sleep(max(0, min(10, self.pause_at - time.time())))
            (self.root / 'DEADLINE').touch()
            Path(self.config['pause_file']).touch()
            save(self.root / 'pause_requested.json', dict(at=now(), deadline=self.config['deadline']))
            # If labeling overruns the reserved GPU window, pause its controller
            # too. Question-level checkpoints remain available for later resume.
            self.request_cloud_stop()
            signal_owned(self.config['workflow_process'], signal.SIGTERM)
            if not immediate:
                while time.time() < self.deadline - 45:
                    time.sleep(max(0, min(5, self.deadline - 45 - time.time())))
            with self.lock('launch'):
                self.control('stop', '--no-block', self.config['unit'])
            signal_owned(self.config['workflow_process'], signal.SIGKILL)
            signal_owned(self.config['label_process'], signal.SIGTERM)
            self.stop_local(grace=10)
            # RuntimeMaxSec + TimeoutStopSec independently cover a dead watchdog.
            until = time.monotonic() + 15
            while self.service().get('ActiveState') in ('active', 'activating', 'deactivating'):
                if time.monotonic() >= until:
                    self.control('kill', '--signal=SIGKILL', '--kill-whom=all', self.config['unit'])
                    break
                time.sleep(1)
            signal_owned(self.config['label_process'], signal.SIGKILL)
            self.request_cloud_stop()
            save(self.root / 'deadline_enforced.json', dict(at=now(), service=self.service(),
                fit=read(self.config['fit_status'], {}), cloud_shutdown=read(self.fleet_root / 'shutdown_complete.json', {})))

    def request_cloud_stop(self):
        # A cloud API/service-manager outage must not block the local GPU deadline.
        try:
            self.cloud_stop(wait=False)
        except Exception as error:
            save(self.root / f'cloud_stop_error_{os.getpid()}.json', dict(at=now(), error=repr(error)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--mode', choices=['run', 'watchdog', 'enforce', 'cloud-cleanup', 'inspect'], default='run')
    args = parser.parse_args()
    handoff = Handoff(args.config)
    if args.mode == 'inspect':
        print(json.dumps(dict(deadline=handoff.config['deadline'], pause_at=handoff.config['pause_at'],
            teachers=discover_teachers(handoff.workflow), training_command=handoff.config['training_command']), indent=2))
    elif args.mode == 'cloud-cleanup':
        with handoff.lock('cloud-cleanup'):
            from scripts.gcp_note_fleet import Fleet
            Fleet(read(handoff.config['fleet_config']), handoff.fleet_root).shutdown('label_to_trialspace_cleanup')
    elif args.mode in ('watchdog', 'enforce'):
        handoff.watchdog(immediate=args.mode == 'enforce')
    else:
        handoff.run()


if __name__ == '__main__':
    main()
