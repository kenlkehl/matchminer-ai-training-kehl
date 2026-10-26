import hashlib
import base64
import json
import io
from types import SimpleNamespace

import pytest

from scripts import gcp_note_fleet as fleet
from scripts import gcp_note_worker as worker


def instance(tmp_path):
    source = tmp_path / 'worker.py'
    source.write_text('print("worker")')
    registry = tmp_path / 'servers.json'
    registry.write_text(json.dumps(dict(ready=True, servers=[dict(url='http://existing/v1')])) )
    config = dict(vms=[dict(name=f'rtx-{i}', project='profile-notes', zone='us-central1-c') for i in range(8)],
        worker_source=str(source), worker_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        worker=dict(config_hash='fixed'), lease_seconds=900, start_retry_seconds=300,
        registry=str(registry), fleet_id='test')
    return fleet.Fleet(config, tmp_path)


@pytest.mark.parametrize('stage,alive,reason', [
    ('notes', True, None), ('failed', True, 'note_generation_failed'),
    ('notes', False, 'note_generator_exited'), ('completed', False, 'note_inference_finished'),
    ('assembling', True, 'note_inference_finished')])
def test_shutdown_tracks_actual_job_lifecycle(stage, alive, reason):
    assert fleet.terminal_reason(dict(stage=stage), alive) == reason


@pytest.mark.parametrize('phase', ['preparing_corpus', 'mining', 'starting_labelers', 'labeling'])
def test_label_fleet_keeps_teachers_during_retrieval_and_labeling(phase):
    assert fleet.workflow_terminal_reason(dict(stage='completed'), False, dict(phase=phase), True,
                                          mode='labeling') is None


@pytest.mark.parametrize('phase,alive,reason', [
    ('completed', False, 'labeling_finished'), ('completed', True, 'labeling_finished'),
    ('failed', True, 'downstream_workflow_failed'), ('mining', False, 'downstream_controller_exited'),
    ('labeling', False, 'downstream_controller_exited')])
def test_label_fleet_stops_when_attached_workflow_finishes_or_fails(phase, alive, reason):
    assert fleet.workflow_terminal_reason(dict(stage='completed'), False, dict(phase=phase), alive,
                                          mode='labeling') == reason


def test_label_fleet_stops_before_cpu_only_reasoning_export_finishes():
    workflow = dict(phase='labeling', child_pid=42)
    labels = dict(pid=42, state='exporting_labels', inference_complete=True,
                  total_pairs=10, scored_pairs=10, failures={})
    assert fleet.workflow_terminal_reason({}, False, workflow, True, mode='labeling',
        label_status=labels) == 'label_inference_finished'
    for change in [dict(pid=41), dict(inference_complete=False), dict(scored_pairs=9),
                   dict(failures={'key': 'error'}), dict(state='labeling')]:
        assert fleet.workflow_terminal_reason({}, False, workflow, True, mode='labeling',
            label_status={**labels, **change}) is None


def test_label_fleet_requires_attached_workflow(tmp_path):
    obj = instance(tmp_path)
    obj.config['workflow_mode'] = 'labeling'
    with pytest.raises(ValueError, match='attached workflow'):
        fleet.Fleet(obj.config, tmp_path)


def test_registry_merge_preserves_existing_servers_and_removes_only_owned_fleet():
    other = [dict(url='existing'), dict(url='unrelated-fleet', fleet_id='other')]
    current = dict(ready=True, servers=other + [dict(url='old-worker', fleet_id='this')])
    assert fleet.merge_servers(current, 'this', [])['servers'] == other
    new = dict(url='healthy-worker', fleet_id='this')
    assert fleet.merge_servers(current, 'this', [new])['servers'] == other + [new]


def test_spot_start_failure_is_retried_with_backoff(tmp_path, monkeypatch):
    obj = instance(tmp_path)
    monkeypatch.setattr(obj, 'prepare_metadata', lambda vm: None)
    clock = [1000]
    monkeypatch.setattr(fleet.time, 'monotonic', lambda: clock[0])
    calls = []
    def cloud(command, vm):
        calls.append((command, vm))
        return SimpleNamespace(returncode=1, stderr='Spot capacity unavailable')
    monkeypatch.setattr(obj, 'cloud', cloud)
    state = dict(status='TERMINATED', scheduling=dict(provisioningModel='SPOT'), machineType='/g4-standard-384')
    assert obj.maintain_vm(obj.vms[0], state)['start_exit'] == 1
    obj.maintain_vm(obj.vms[0], state)
    assert len(calls) == 1
    clock[0] += 301
    obj.maintain_vm(obj.vms[0], state)
    assert len(calls) == 2
    obj.stop = True
    clock[0] += 301
    obj.maintain_vm(obj.vms[0], state)
    assert len(calls) == 2


def test_shutdown_retries_until_allowlisted_vms_are_stopped(tmp_path, monkeypatch):
    obj = instance(tmp_path)
    active = [True]
    def states():
        return {(v['project'],v['name']):dict(status='RUNNING' if active[0] else 'TERMINATED') for v in obj.vms}
    monkeypatch.setattr(obj, 'states', states)
    calls = []
    monkeypatch.setattr(obj, 'cloud', lambda command, vm: calls.append((command,vm)) or SimpleNamespace(returncode=0,stderr=''))
    monkeypatch.setattr(fleet.time, 'sleep', lambda seconds: active.__setitem__(0,False))
    obj.shutdown('note_generation_failed')
    assert len(calls) == 8 and all(command[:2] == ['instances','stop'] for command,vm in calls)
    assert json.loads((tmp_path/'shutdown_complete.json').read_text())['all_terminated']
    assert json.loads((tmp_path/'servers.json').read_text())['servers'] == [dict(url='http://existing/v1')]


def test_remote_guard_powers_off_after_lease_expiry(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, 'ROOT', tmp_path)
    (tmp_path/'lease.json').write_text(json.dumps(dict(expires_at=0)))
    commands = []
    monkeypatch.setattr(worker, 'run', lambda command: commands.append(command))
    worker.guard()
    assert commands == [['systemctl', 'poweroff']]
    assert json.loads((tmp_path/'shutdown.json').read_text())['reason'] == 'controller lease expired'


def test_startup_arms_guard_before_replacing_old_workers(tmp_path):
    obj = instance(tmp_path)
    script = obj.startup
    assert script.index('base64 -d | python3') < script.index('systemctl disable --now')
    assert 'mmai-nemotron-fleet.service' in script
    assert 'mmai-gemma-note-worker.service' in script


def test_reboot_with_expired_lease_has_time_to_reconnect(tmp_path):
    script = instance(tmp_path).startup
    encoded = script.split('printf %s ', 1)[1].split(' |', 1)[0]
    installer = base64.b64decode(encoded).decode()
    assert 'ExecStartPre=/usr/bin/sleep 900' in installer
    assert 'TimeoutStartSec=0' in installer
    assert '["systemctl","start","--no-block","mmai-gemma-note-guard.service"]' in installer
    assert '["systemctl","enable","--now"' not in installer


@pytest.mark.parametrize('healthy', [True, False])
def test_endpoint_admission_requires_a_live_engine(tmp_path, monkeypatch, healthy):
    obj = instance(tmp_path)
    obj.vms[0]['base_port'] = 21000
    obj.config['worker'].update(served_model='gemma', model='nvidia/gemma', context=250000, max_num_seqs=256)
    (tmp_path/'vm_status.json').write_text(json.dumps(dict(vms=[dict(name='rtx-0', bootstrap=dict(healthy_gpus=[0]))])))
    calls = []
    def open_url(url, timeout):
        calls.append(url)
        if url.endswith('/health'):
            if not healthy:
                raise OSError('engine died')
            return io.BytesIO(b'')
        return io.BytesIO(json.dumps(dict(data=[dict(id='gemma', root='nvidia/gemma', max_model_len=250000)])).encode())
    monkeypatch.setattr(fleet.urllib.request, 'build_opener', lambda *args: SimpleNamespace(open=open_url))
    assert len(obj.endpoints()) == int(healthy)
    assert calls[0].endswith('/health')
    assert len(calls) == (2 if healthy else 1)


def test_shutdown_still_stops_cloud_when_checkpoint_filesystem_fails(tmp_path, monkeypatch):
    obj = instance(tmp_path)
    active = [True]
    monkeypatch.setattr(obj, 'states', lambda: {
        (v['project'], v['name']): dict(status='RUNNING' if active[0] else 'TERMINATED') for v in obj.vms})
    def broken(*args):
        raise OSError('checkpoint filesystem unavailable')
    monkeypatch.setattr(fleet, 'save', broken)
    monkeypatch.setattr(obj, 'publish', broken)
    calls = []
    monkeypatch.setattr(obj, 'cloud', lambda command, vm: calls.append(vm['name']) or SimpleNamespace(returncode=0, stderr=''))
    monkeypatch.setattr(fleet.time, 'sleep', lambda seconds: active.__setitem__(0, False))
    obj.shutdown('checkpoint_write_failed')
    assert sorted(calls) == sorted(v['name'] for v in obj.vms)


def test_tunnels_reserve_fds_for_eight_servers(tmp_path, monkeypatch):
    monkeypatch.setattr(fleet.resource, 'getrlimit', lambda limit: (4096, 1048576))
    calls = []
    monkeypatch.setattr(fleet.resource, 'setrlimit', lambda limit, value: calls.append(value))
    fleet.ensure_ssh_descriptor_capacity()
    assert calls == [(65536, 1048576)]
    script = instance(tmp_path).startup
    installer = base64.b64decode(script.split('printf %s ', 1)[1].split(' |', 1)[0]).decode()
    assert 'LimitNOFILE=65536' in installer
    assert 'resource.prlimit' in installer


def test_tunnels_reject_insufficient_hard_fd_limit(monkeypatch):
    monkeypatch.setattr(fleet.resource, 'getrlimit', lambda limit: (1024, 4096))
    with pytest.raises(RuntimeError, match='hard limit'):
        fleet.ensure_ssh_descriptor_capacity()


def test_new_controller_loads_host_keys_only_from_authenticated_gcp(tmp_path, monkeypatch):
    obj = instance(tmp_path)
    vm = dict(name='rtx-one', project='profile-notes', zone='us-central1-c',
              alias='trusted-vm', known=str(tmp_path / 'known'))
    calls = []
    def cloud(command, selected):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout=json.dumps([
            dict(namespace='hostkeys', key='ssh-ed25519', value='public-key-data')]))
    monkeypatch.setattr(obj, 'cloud', cloud)
    obj.ensure_host_keys(vm)
    assert (tmp_path / 'known').read_text() == 'trusted-vm ssh-ed25519 public-key-data\n'
    obj.ensure_host_keys(vm)
    assert len(calls) == 1


def test_missing_gcp_host_keys_do_not_disable_ssh_verification(tmp_path, monkeypatch):
    obj = instance(tmp_path)
    vm = dict(name='rtx-one', known=str(tmp_path / 'known'), alias='trusted-vm')
    monkeypatch.setattr(obj, 'cloud', lambda *a: SimpleNamespace(returncode=1, stderr='not published yet'))
    with pytest.raises(RuntimeError, match='not available yet'):
        obj.ensure_host_keys(vm)
    assert not (tmp_path / 'known').exists()
