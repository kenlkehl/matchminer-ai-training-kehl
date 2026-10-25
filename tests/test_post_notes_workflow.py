import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import gcp_note_fleet as fleet
from scripts import post_notes_workflow as workflow
import summary_pool as summary


@pytest.mark.parametrize('phase', ['waiting_notes', 'summarizing'])
@pytest.mark.parametrize('stage,alive', [('notes', True), ('assembling', True), ('completed', False)])
def test_fleet_keeps_workers_through_summary_handoff(phase, stage, alive):
    assert fleet.workflow_terminal_reason(dict(stage=stage), alive, dict(phase=phase), True) is None


@pytest.mark.parametrize('phase', ['stopping_cloud', 'mining', 'labeling', 'completed'])
def test_fleet_stops_before_student_work(phase):
    assert fleet.workflow_terminal_reason(dict(stage='completed'), False, dict(phase=phase), True) == 'patient_summarization_finished'


def test_fleet_stops_on_generator_or_downstream_failure():
    assert fleet.workflow_terminal_reason(dict(stage='failed'), True, dict(phase='waiting_notes'), True) == 'note_generation_failed'
    assert fleet.workflow_terminal_reason(dict(stage='notes'), False, dict(phase='waiting_notes'), True) == 'note_generator_exited'
    assert fleet.workflow_terminal_reason(dict(stage='completed'), False, dict(phase='failed'), True) == 'downstream_workflow_failed'
    assert fleet.workflow_terminal_reason(dict(stage='completed'), False, dict(phase='summarizing'), False) == 'downstream_controller_exited'


def test_workflow_preserves_required_stage_order(tmp_path, monkeypatch):
    runner = workflow.Workflow({}, tmp_path)
    calls = []
    for method in ('wait_notes', 'stop_cloud', 'release_gpus', 'start_labelers'):
        monkeypatch.setattr(runner, method, lambda method=method: calls.append(method))
    monkeypatch.setattr(runner, 'run_stage', calls.append)
    monkeypatch.setattr(workflow.signal, 'signal', lambda *args: None)
    runner.run()
    assert calls == ['wait_notes', 'summarizing', 'stop_cloud', 'release_gpus',
                     'preparing_corpus', 'mining', 'start_labelers', 'labeling']
    assert runner.status['phase'] == 'completed'


def test_mining_failure_never_starts_labeling(tmp_path, monkeypatch):
    config = tmp_path / 'fleet/config.json'
    config.parent.mkdir()
    config.write_text('{}')
    runner = workflow.Workflow(dict(fleet_config=str(config)), tmp_path)
    calls = []
    for method in ('wait_notes', 'stop_cloud', 'release_gpus', 'start_labelers'):
        monkeypatch.setattr(runner, method, lambda method=method: calls.append(method))
    def stage(name):
        calls.append(name)
        if name == 'mining':
            raise RuntimeError('GPU failure')
    monkeypatch.setattr(runner, 'run_stage', stage)
    monkeypatch.setattr(workflow.signal, 'signal', lambda *args: None)
    monkeypatch.setattr(workflow, 'Fleet', lambda *args: SimpleNamespace(shutdown=lambda reason: calls.append('cloud_cleanup')))
    with pytest.raises(RuntimeError, match='GPU failure'):
        runner.run()
    assert 'start_labelers' not in calls and 'labeling' not in calls
    assert 'cloud_cleanup' in calls and runner.status['phase'] == 'failed'


def test_teacher_pid_reuse_is_never_signaled(tmp_path, monkeypatch):
    target = dict(gpu=2, port=8003, process=dict(pid=999, start_ticks=100, command=['vllm', 'serve', 'model', '--port', '8003']))
    monkeypatch.setattr(workflow, 'teacher_identity', lambda pid: {**target['process'], 'start_ticks': 101})
    monkeypatch.setattr(workflow, 'open_pidfd', lambda pid: 123)
    monkeypatch.setattr(workflow.os, 'close', lambda fd: None)
    signals = []
    monkeypatch.setattr(workflow, 'send_pidfd_signal', lambda *args: signals.append(args))
    with pytest.raises(ValueError, match='process changed'):
        workflow.stop_teacher(target, tmp_path/'receipt.json')
    assert signals == []


def test_teacher_can_share_only_registered_unchanged_allocation(monkeypatch):
    allocation = dict(gpu='0', pid=123)
    monkeypatch.setattr(workflow, 'busy_gpus', lambda selected: [allocation])
    monkeypatch.setattr(workflow, 'teacher_identity', lambda pid: dict(pid=pid, start_ticks=44))
    target = dict(gpu=0, allowed_cooccupants=[dict(pid=123, start_ticks=44)])
    assert workflow.teacher_blockers(target) == []
    assert workflow.teacher_blockers(dict(gpu=0)) == [allocation]
    target['allowed_cooccupants'][0]['start_ticks'] = 43
    assert workflow.teacher_blockers(target) == [allocation]


def test_four_label_teachers_do_not_expand_registered_shutdown_scope(tmp_path, monkeypatch):
    old = [dict(gpu=2), dict(gpu=3)]
    runner = workflow.Workflow(dict(release_teachers=old,
        local_teachers=[dict(gpu=gpu) for gpu in range(4)]), tmp_path)
    stopped, checked = [], []
    monkeypatch.setattr(workflow, 'stop_teacher', lambda target, receipt, **kwargs: stopped.append(target['gpu']))
    monkeypatch.setattr(workflow, 'busy_gpus', lambda selected: checked.append(selected) or [])
    runner.release_gpus()
    assert stopped == [2, 3] and checked == [['2'], ['3']]


def test_release_all_gpus_requires_explicit_allowlist_and_preserves_cooccupants(tmp_path, monkeypatch):
    target = dict(gpu=0, process=dict(pid=123, start_ticks=44),
                  allowed_cooccupants=[dict(pid=999, start_ticks=22)])
    with pytest.raises(ValueError, match='explicitly selected'):
        workflow.stop_teacher(target, tmp_path / 'rejected.json')
    monkeypatch.setattr(workflow, 'teacher_identity', lambda pid: None if pid == 123 else dict(pid=999, start_ticks=22))
    monkeypatch.setattr(workflow, 'busy_gpus', lambda selected: [dict(gpu='0', pid=999)])
    runner = workflow.Workflow(dict(release_teachers=[target], local_teachers=[], release_gpu_allowlist=[0]), tmp_path)
    runner.release_gpus()
    assert json.loads((tmp_path / 'stopped_gpu_0.json').read_text())['state'] == 'already_stopped'


def test_truncated_summary_attempt_keeps_raw_reasoning(tmp_path):
    async def base(client, payload):
        return 'Thinking about the input', 'Partial summary', dict(finish_reason='length', raw_text='RAW REASONING AND PARTIAL ANSWER')
    async def scenario():
        call = summary.traced_work(base, tmp_path, summary.CompletionSampling(model='synthetic'))
        with pytest.raises(summary.OutputValidationError):
            await call(SimpleNamespace(base_url='http://synthetic/v1'), dict(chunk_index=0, prompt='SOURCE INPUT'))
    asyncio.run(scenario())
    record = json.loads(next((tmp_path/'attempts').glob('*/*.json')).read_text())
    assert record['reasoning'] == 'Thinking about the input'
    assert record['raw_text'] == 'RAW REASONING AND PARTIAL ANSWER'
    assert record['prompt'] == 'SOURCE INPUT'


def test_summary_chunk_resume_preserves_serial_dependency(tmp_path, monkeypatch):
    calls, saved_rounds = [], []
    class Session:
        def __init__(self, registry, work, **kw): self.work = work
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def submit(self, payload): return await self.work(SimpleNamespace(base_url='http://synthetic/v1'), payload)
    def make_work(*args):
        async def work(client, payload):
            calls.append(payload['prompt'])
            return 'REASONING', f'Summary {payload["chunk_index"]}', dict(finish_reason='stop', raw_text='RAW COMPLETE')
        return work
    monkeypatch.setattr(summary, 'RequestSession', Session)
    monkeypatch.setattr(summary, 'make_completion_work_fn', make_work)
    monkeypatch.setattr(summary, 'build_registry_from_args', lambda args: None)
    monkeypatch.setattr(summary.resource, 'getrlimit', lambda limit: (65536, 65536))
    pool = SimpleNamespace(_processes=1, map=lambda fn, items, **kw: list(map(fn, items)))
    def build(item):
        idx, prior, first, last, text, context = item
        return idx, str(prior) + ' NEXT ' + text, 20
    args = SimpleNamespace(shard_dir=str(tmp_path), pending_requests=2, max_concurrent_per_server=2,
        model='synthetic', request_timeout=60, temperature=1, top_k=64, top_p=.95,
        presence_penalty=0, min_p=0, repetition_penalty=1, max_model_len=1000, max_tokens=100, max_attempts=2)
    async def run():
        summaries, results = {}, {}
        await summary.run_rounds(rounds=[[('p',0,'first','last','FIRST')], [('p',1,'first','last','SECOND')]],
            patient_chunk_order={'p':[0,1]}, patient_summaries=summaries, all_results=results,
            completed_rounds=0, args=args, prompt_pool=pool, parser_name='synthetic',
            tokenizer=SimpleNamespace(chat_template='TEMPLATE'), build_prompt=build,
            save_round=lambda *args: saved_rounds.append(args))
        return summaries
    assert asyncio.run(run()) == {'p':'Summary 1\n\nOther relevant history:\nNot documented.'}
    assert calls == ['None NEXT FIRST', 'Summary 0\n\nOther relevant history:\nNot documented. NEXT SECOND']
    assert len(list((tmp_path/'chunks').glob('*/*.json'))) == 2
    assert asyncio.run(run()) == {'p':'Summary 1\n\nOther relevant history:\nNot documented.'}
    assert len(calls) == 2
