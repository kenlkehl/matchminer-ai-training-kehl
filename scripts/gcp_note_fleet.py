#!/usr/bin/env python3
"""Attach Spot GPU teachers to notes, summaries, or labels; stop with the job."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import argparse
import base64
import fcntl
import hashlib
import json
import os
import resource
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

REMOTE = '/opt/mmai-gemma-note-fleet'


def ensure_ssh_descriptor_capacity():
    """Each VM tunnel carries connections for eight independent GPU servers."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    required = 65536
    if hard != resource.RLIM_INFINITY and hard < required:
        raise RuntimeError('The fleet controller needs an open-file hard limit of at least 65536')
    if soft != resource.RLIM_INFINITY and soft < required:
        resource.setrlimit(resource.RLIMIT_NOFILE, (required, hard))


def remote_ssh_limit_setup():
    # Apply to live sshd processes without disconnecting model requests, and
    # retain the higher limit for the next Spot boot. Ubuntu uses ssh.service.
    return ('from pathlib import Path\nimport resource\n'
        'for name in ("ssh", "sshd"):\n'
        ' p=Path("/etc/systemd/system")/(name+".service.d");p.mkdir(exist_ok=True)\n'
        ' (p/"mmai-note-fleet.conf").write_text("[Service]\\nLimitNOFILE=65536\\n")\n'
        'for p in Path("/proc").iterdir():\n'
        ' if not p.name.isdigit():continue\n'
        ' try:\n'
        '  if (p/"comm").read_text().strip().startswith("sshd"):\n'
        '   soft,hard=resource.prlimit(int(p.name),resource.RLIMIT_NOFILE)\n'
        '   resource.prlimit(int(p.name),resource.RLIMIT_NOFILE,(max(soft,65536),max(hard,65536)))\n'
        ' except (FileNotFoundError,ProcessLookupError):pass\n')


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def process_identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return fields[19] if fields[0] != 'Z' else None
    except FileNotFoundError:
        return None


def terminal_reason(status, alive):
    if status.get('stage') in ('completed', 'assembling'):
        return 'note_inference_finished'
    if status.get('stage') == 'failed':
        return 'note_generation_failed'
    if not alive:
        return 'note_generator_exited'
    return None


def workflow_terminal_reason(generation, generator_alive, workflow, workflow_alive, *, mode='summarization',
                             label_status=None):
    """Watch the explicitly selected inference stage, including its preparation."""
    if mode not in ('summarization', 'labeling'):
        raise ValueError('Unknown fleet workflow mode')
    if workflow.get('phase') == 'failed':
        return 'downstream_workflow_failed'
    if mode == 'labeling' and workflow.get('phase') == 'completed':
        return 'labeling_finished'
    if (mode == 'labeling' and label_status and label_status.get('inference_complete') is True
            and workflow.get('phase') == 'labeling' and workflow.get('child_pid') is not None
            and label_status.get('pid') == workflow['child_pid']
            and label_status.get('state') in ('exporting_labels', 'complete')
            and not label_status.get('failures')
            and label_status.get('scored_pairs') == label_status.get('total_pairs')
            and label_status.get('total_pairs', 0) > 0):
        return 'label_inference_finished'
    if not workflow_alive:
        return 'downstream_controller_exited'
    if mode == 'labeling':
        # The note generator has already completed. Keep teachers warming during
        # retrieval, then serve the separate live labeling registry until done.
        return None
    if workflow.get('phase') in ('stopping_cloud', 'releasing_gpus', 'preparing_corpus',
                                  'mining', 'starting_labelers', 'labeling', 'completed'):
        return 'patient_summarization_finished'
    if generation.get('stage') == 'failed':
        return 'note_generation_failed'
    if generation.get('stage') != 'completed' and not generator_alive:
        return 'note_generator_exited'
    return None


def merge_servers(registry, fleet_id, servers):
    return {**registry, 'ready': True,
            'servers': [s for s in registry['servers'] if s.get('fleet_id') != fleet_id] + servers}


def startup_script(config, worker_source):
    worker = config['worker']
    service = ('[Unit]\nDescription=Gemma note workers\nAfter=network-online.target docker.service\n'
        'Wants=network-online.target\nRequires=docker.service\nStartLimitIntervalSec=0\n'
        '[Service]\nType=oneshot\nExecStart=/usr/bin/python3 ' + REMOTE + '/worker.py\n'
        'RemainAfterExit=yes\nTimeoutStartSec=7200\nRestart=on-failure\nRestartSec=60\n'
        '[Install]\nWantedBy=multi-user.target\n')
    guard = ('[Unit]\nDescription=Stop note VM if controller lease expires\nAfter=network-online.target\n'
        # A preempted VM can boot with an expired on-disk lease. Give startup
        # metadata and SSH time to renew it before enforcing the lease again.
        f'[Service]\nExecStartPre=/usr/bin/sleep {config["lease_seconds"]}\nTimeoutStartSec=0\n'
        'ExecStart=/usr/bin/python3 ' + REMOTE + '/worker.py --guard\nRestart=always\nRestartSec=5\n'
        '[Install]\nWantedBy=multi-user.target\n')
    installer = (remote_ssh_limit_setup() + 'import json,time,subprocess\n'
        f'r=Path({REMOTE!r});r.mkdir(exist_ok=True)\n'
        f'(r/"worker.py").write_text({worker_source!r})\n'
        f'(r/"config.json").write_text({json.dumps(worker)!r})\n'
        f'(r/"lease.json").write_text(json.dumps(dict(expires_at=time.time()+{config["lease_seconds"]})))\n'
        f'Path("/etc/systemd/system/mmai-gemma-note-worker.service").write_text({service!r})\n'
        f'Path("/etc/systemd/system/mmai-gemma-note-guard.service").write_text({guard!r})\n'
        'subprocess.run(["systemctl","daemon-reload"],check=True)\n'
        'subprocess.run(["systemctl","enable","mmai-gemma-note-guard.service"],check=True)\n'
        'subprocess.run(["systemctl","start","--no-block","mmai-gemma-note-guard.service"],check=True)\n')
    encoded = base64.b64encode(installer.encode()).decode()
    return ('#!/bin/bash\nset -euo pipefail\n'
        f'printf %s {encoded} | base64 -d | python3\n'
        'systemctl disable --now mmai-nemotron-fleet.service mmai-beam-bootstrap.service 2>/dev/null || true\n'
        'while read -r container; do\n'
        ' case "$container" in\n'
        ' mmai-nemotron*|mmai-beam*) docker update --restart=no "$container" >/dev/null; '
        'docker stop --time 15 "$container" >/dev/null ;;\n'
        ' esac\ndone < <(docker ps -a --format "{{.Names}}")\n'
        'systemctl enable mmai-gemma-note-worker.service\n'
        'systemctl start --no-block mmai-gemma-note-worker.service\n')


class Fleet:
    def __init__(self, config, directory):
        self.config, self.root = config, Path(directory)
        mode = config.get('workflow_mode', 'summarization')
        if mode not in ('summarization', 'labeling') or (mode == 'labeling' and not config.get('workflow')):
            raise ValueError('Labeling fleets require an attached workflow process identity')
        self.vms = config['vms']
        if (len(self.vms) != 8 or len({(v['project'], v['name']) for v in self.vms}) != 8
                or any('rtx' not in v['name'] or v['project'] not in ('profile-notes', 'kehllab-caia-v2') for v in self.vms)):
            raise ValueError('This job requires the explicit eight-RTX-VM allowlist')
        self.root.mkdir(parents=True, exist_ok=True)
        self.last_start, self.metadata_ready = {}, set()
        self.worker_source = Path(config['worker_source']).read_text()
        if hashlib.sha256(self.worker_source.encode()).hexdigest() != config['worker_source_sha256']:
            raise ValueError('Frozen worker source changed')
        self.startup = startup_script(config, self.worker_source)
        (self.root / 'startup.sh').write_text(self.startup)
        self.stop = False

    def run(self, command, timeout=45, **kwargs):
        return subprocess.run(command, text=True, capture_output=True, timeout=timeout, **kwargs)

    def cloud(self, args, vm=None, timeout=45):
        if vm:
            args = [*args, '--project', vm['project'], '--zone', vm['zone']]
        return self.run([self.config['gcloud'], 'compute', *args], timeout=timeout)

    def states(self):
        def one(project):
            result = self.cloud(['instances', 'list', '--project', project, '--filter=name~rtx',
                '--format=json(name,status,zone,machineType,scheduling,networkInterfaces)'])
            if result.returncode:
                raise RuntimeError(result.stderr[-1500:])
            return [(project, row) for row in json.loads(result.stdout)]
        with ThreadPoolExecutor(max_workers=2) as pool:
            return {(project, row['name']): row for group in pool.map(one, sorted({v['project'] for v in self.vms}))
                    for project, row in group}

    def ssh(self, vm):
        return ['ssh', '-i', vm['key'], '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes',
            '-o', 'StrictHostKeyChecking=yes', '-o', 'UserKnownHostsFile=' + vm['known'],
            '-o', 'HostKeyAlias=' + vm['alias'], '-o', 'ConnectTimeout=10',
            '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3']

    def remote(self, vm, ip, code, shell='sudo -n python3 -', timeout=45):
        result = self.run([*self.ssh(vm), 'kenneth_kehl@' + ip, shell], input=code, timeout=timeout)
        if result.returncode:
            raise RuntimeError(result.stderr[-1500:])
        return result.stdout

    def connect(self, vm, ip):
        receipt = self.root / (vm['name'] + '-tunnel.json')
        base = [*self.ssh(vm), '-S', vm['socket']]
        if receipt.exists() and json.loads(receipt.read_text()).get('ip') != ip:
            self.run([*base, '-O', 'exit', 'kenneth_kehl@' + ip])
        if self.run([*base, '-O', 'check', 'kenneth_kehl@' + ip]).returncode == 0:
            return
        command = [*base, '-M', '-fNT', '-o', 'ExitOnForwardFailure=yes']
        for gpu in range(8):
            command += ['-L', f'127.0.0.1:{vm["base_port"] + gpu}:127.0.0.1:{8000 + gpu}']
        result = self.run([*command, 'kenneth_kehl@' + ip])
        if result.returncode:
            raise RuntimeError(result.stderr[-1500:])
        save(receipt, dict(ip=ip, at=now()))

    def prepare_metadata(self, vm):
        key = (vm['project'], vm['name'])
        if key in self.metadata_ready:
            return
        before = self.root / 'metadata_before' / (vm['name'] + '.json')
        if not before.exists():
            result = self.cloud(['instances', 'describe', vm['name'], '--format=json'], vm)
            if result.returncode:
                raise RuntimeError(result.stderr[-1500:])
            save(before, json.loads(result.stdout))
        result = self.cloud(['instances', 'add-metadata', vm['name'], '--metadata-from-file',
            'startup-script=' + str(self.root / 'startup.sh'), '--quiet'], vm)
        if result.returncode:
            raise RuntimeError(result.stderr[-1500:])
        self.metadata_ready.add(key)

    def maintain_vm(self, vm, state):
        result = dict(name=vm['name'], project=vm['project'], status=state.get('status'), at=now())
        try:
            if self.stop or (self.root / 'STOP').exists():
                return result
            if state.get('scheduling', {}).get('provisioningModel') != 'SPOT':
                raise ValueError('VM no longer has Spot provisioning')
            if not state.get('machineType', '').endswith('/g4-standard-384'):
                raise ValueError('VM no longer matches the eight-GPU machine configuration')
            self.prepare_metadata(vm)
            if self.stop or (self.root / 'STOP').exists():
                return result
            if state['status'] in ('TERMINATED', 'SUSPENDED'):
                if time.monotonic() - self.last_start.get(vm['name'], -10000) >= self.config['start_retry_seconds']:
                    self.last_start[vm['name']] = time.monotonic()
                    action = 'resume' if state['status'] == 'SUSPENDED' else 'start'
                    response = self.cloud(['instances', action, vm['name'], '--async', '--quiet'], vm)
                    result.update(start_exit=response.returncode, start_error=response.stderr[-1500:])
                return result
            if state['status'] != 'RUNNING':
                return result
            ip = state['networkInterfaces'][0]['accessConfigs'][0]['natIP']
            result['ip'] = ip
            code = ('from pathlib import Path\nimport json,time,subprocess\n'
                f'r=Path({REMOTE!r});p=r/"config.json"\n'
                f'valid=p.exists() and json.loads(p.read_text()).get("config_hash")=={self.config["worker"]["config_hash"]!r}\n'
                'if valid:\n'
                f' q=r/"lease.tmp";q.write_text(json.dumps(dict(expires_at=time.time()+{self.config["lease_seconds"]})));q.replace(r/"lease.json")\n'
                ' subprocess.run(["systemctl","start","--no-block","mmai-gemma-note-worker.service"],check=True)\n'
                'p=r/"status.json"\n'
                'print(json.dumps(dict(installed=valid,bootstrap=json.loads(p.read_text()) if p.exists() else {})))\n')
            result.update(json.loads(self.remote(vm, ip, code)))
            if not result['installed']:
                # Startup metadata is normally sufficient; repair a missed startup.
                self.remote(vm, ip, self.startup, shell='sudo -n bash -s', timeout=180)
            self.connect(vm, ip)
        except Exception as error:
            result['error'] = repr(error)
        return result

    def maintain(self):
        states = self.states()
        with ThreadPoolExecutor(max_workers=8) as pool:
            result = list(pool.map(lambda v: self.maintain_vm(v, states.get((v['project'], v['name']), {})), self.vms))
        save(self.root / 'vm_status.json', dict(at=now(), vms=result))
        return result

    def endpoints(self):
        info = self.root / 'vm_status.json'
        reports = {v['name']: v for v in json.loads(info.read_text())['vms']} if info.exists() else {}
        candidates = [(v, gpu) for v in self.vms
                      for gpu in reports.get(v['name'], {}).get('bootstrap', {}).get('healthy_gpus', [])]
        def check(item):
            vm, gpu = item
            url = f'http://127.0.0.1:{vm["base_port"] + gpu}/v1'
            try:
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(url.removesuffix('/v1') + '/health', timeout=3) as response:
                    response.read()
                with opener.open(url + '/models', timeout=3) as response:
                    models = json.load(response)['data']
                target = self.config['worker']
                if not any(m['id'] == target['served_model'] and m.get('root') == target['model']
                           and m.get('max_model_len', 0) >= target['context'] for m in models):
                    return None
                return dict(url=url, model=target['served_model'], model_root=target['model'],
                    instance=vm['name'], project=vm['project'], gpu=gpu, kind='gcp_spot_ssh',
                    fleet_id=self.config['fleet_id'], upstream_concurrency=target['max_num_seqs'])
            except Exception:
                return None
        with ThreadPoolExecutor(max_workers=64) as pool:
            return [row for row in pool.map(check, candidates) if row]

    def publish(self, endpoints):
        path = Path(self.config['registry'])
        with path.with_suffix('.fleet.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            current = json.loads(path.read_text())
            result = merge_servers(current, self.config['fleet_id'], endpoints)
            if result != current:
                save(path, result)

    def shutdown_record(self, filename, value):
        # A full/unavailable checkpoint filesystem must not prevent cloud stops.
        try:
            save(self.root / filename, value)
        except Exception as error:
            print(json.dumps(dict(shutdown_record=filename, error=repr(error), value=value)), flush=True)

    def shutdown(self, reason):
        self.stop = True
        self.shutdown_record('shutdown_requested.json', dict(reason=reason, at=now()))
        try:
            self.publish([])
        except Exception as error:
            self.shutdown_record('shutdown_registry_error.json', dict(at=now(), error=repr(error)))
        while True:
            try:
                states = self.states()
                active = [v for v in self.vms if states.get((v['project'], v['name']), {}).get('status') != 'TERMINATED']
                if not active:
                    self.shutdown_record('shutdown_complete.json', dict(at=now(), reason=reason, all_terminated=True))
                    return
                def stop(vm):
                    try:
                        state = states.get((vm['project'], vm['name']), {}).get('status')
                        if state in ('STOPPING', 'PROVISIONING', 'STAGING'):
                            return dict(name=vm['name'], status=state)
                        result = self.cloud(['instances', 'stop', vm['name'], '--async', '--quiet'], vm)
                        return dict(name=vm['name'], returncode=result.returncode, error=result.stderr[-1000:])
                    except Exception as error:
                        return dict(name=vm['name'], error=repr(error))
                with ThreadPoolExecutor(max_workers=8) as pool:
                    self.shutdown_record('shutdown_progress.json', dict(at=now(), results=list(pool.map(stop, active))))
            except Exception as error:
                self.shutdown_record('shutdown_progress.json', dict(at=now(), error=repr(error)))
            time.sleep(20)

    def serve(self):
        reason, future, next_maintenance = 'supervisor_error', None, 0
        def halt(signum, frame):
            self.stop = True
        signal.signal(signal.SIGTERM, halt)
        signal.signal(signal.SIGINT, halt)
        try:
            # Join maintenance before shutdown so a pending start cannot race it.
            with ThreadPoolExecutor(max_workers=1) as executor:
                while True:
                    status = json.loads(Path(self.config['generation_status']).read_text())
                    alive = process_identity(self.config['generator_pid']) == self.config['generator_start_ticks']
                    workflow_config = self.config.get('workflow')
                    workflow = None
                    if workflow_config:
                        workflow = json.loads(Path(workflow_config['status']).read_text())
                        workflow_alive = process_identity(workflow_config['pid']) == workflow_config['start_ticks']
                        label_path = Path(workflow_config.get('label_status',
                            Path(workflow_config['status']).parent / 'labels/status.json'))
                        label_status = (json.loads(label_path.read_text())
                            if self.config.get('workflow_mode') == 'labeling' and label_path.exists() else None)
                        reason = workflow_terminal_reason(status, alive, workflow, workflow_alive,
                            mode=self.config.get('workflow_mode', 'summarization'), label_status=label_status)
                    else:
                        reason = terminal_reason(status, alive)
                    if self.stop or (self.root / 'STOP').exists():
                        reason = reason or 'operator_stop'
                    if reason:
                        self.stop = True
                        break
                    if future is not None and future.done():
                        try:
                            future.result()
                        except Exception as error:
                            save(self.root / 'maintenance_error.json', dict(at=now(), error=repr(error)))
                        future = None
                    if future is None and time.monotonic() >= next_maintenance:
                        future = executor.submit(self.maintain)
                        next_maintenance = time.monotonic() + 45
                    endpoints = self.endpoints()
                    self.publish(endpoints)
                    save(self.root / 'status.json', dict(at=now(), state='running', supervisor_pid=os.getpid(),
                        generator_pid=self.config['generator_pid'], generation_stage=status['stage'],
                        workflow_phase=workflow.get('phase') if workflow else None,
                        completed_notes=(status.get('stage_completed') if status['stage'] in ('notes', 'assembling', 'completed') else None),
                        expected_notes=status.get('expected_notes'),
                        healthy_gpu_servers=len(endpoints), expected_gpu_servers=64,
                        automatic_vm_shutdown=True, workflow_mode=self.config.get('workflow_mode', 'summarization'),
                        shutdown_on=(['workflow_failure', 'labeling_complete', 'controller_lease_expiry']
                            if self.config.get('workflow_mode') == 'labeling' else
                            ['generator_or_workflow_failure', 'patient_summarization_complete', 'controller_lease_expiry']
                            if workflow_config else ['generator_failure', 'note_inference_complete', 'controller_lease_expiry'])))
                    time.sleep(10)
        except BaseException as error:
            reason = 'supervisor_error'
            save(self.root / 'supervisor_error.json', dict(at=now(), error=repr(error)))
            raise
        finally:
            self.shutdown(reason)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    ensure_ssh_descriptor_capacity()
    config = json.loads(args.config.read_text())
    with (args.config.parent / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        Fleet(config, args.config.parent).serve()


if __name__ == '__main__':
    main()
