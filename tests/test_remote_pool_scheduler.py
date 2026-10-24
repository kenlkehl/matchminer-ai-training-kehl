"""Production generation must apply scheduler feedback to the live request pool."""
import asyncio
import json
import sys
from types import SimpleNamespace

import httpx

import adaptive_controller
import generate_synthetic_data as generation
from remote_vllm_pool import DynamicServerRegistry, build_registry_from_args


def metrics(*, running=128, cache=.4, tokens=0, preemptions=0):
    return '\n'.join(f'vllm:{name}{{engine="0"}} {value}' for name, value in {
        'num_requests_running': running, 'num_requests_waiting': 0,
        'kv_cache_usage_perc': cache, 'generation_tokens_total': tokens,
        'num_preemptions_total': preemptions}.items())


def test_live_scheduler_tunes_endpoints_independently_and_preserves_error_backoff(tmp_path, monkeypatch):
    async def scenario():
        urls = ['http://first/v1', 'http://second/v1']
        path = tmp_path / 'metrics.json'
        registry = DynamicServerRegistry(max_concurrent_per_server=768, request_timeout=60,
            static_urls=urls, scheduler_adaptive_concurrency=True, scheduler_metrics_file=str(path))
        async def close():
            pass
        monkeypatch.setattr(registry, '_make_client', lambda url: SimpleNamespace(close=close))
        await registry._reconcile_once()
        first, second = await registry.active_entries()
        assert first.semaphore.limit == second.semaphore.limit == 128
        clock = [0]
        monkeypatch.setattr(adaptive_controller.time, 'time', lambda: clock[0])
        def respond(request):
            assert request.url.path == '/metrics'
            return httpx.Response(200, text=metrics(cache=.95 if request.url.host == 'second' else .4,
                                                   tokens=clock[0] * 1000))
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            for clock[0] in range(0, 181, 15):
                await registry._sample_scheduler_once(client)
                # Ordinary URL reconciliation must not overwrite measured limits.
                await registry._reconcile_once()
            assert first.semaphore.limit == 192
            assert second.semaphore.limit == 96
            await first.semaphore.acquire()
            await first.semaphore.release(success=False)
            clock[0] = 195
            await registry._sample_scheduler_once(client)
            assert first.semaphore.limit == 96 and first.semaphore.max_limit == 192
            snapshot = json.loads(path.read_text())
            assert snapshot['servers'][0]['limit'] == 96
            assert snapshot['servers'][1]['max_limit'] == 96
            assert all(s['configured_ceiling'] == 768 for s in snapshot['servers'])
        # Missing metrics must not stop dispatch or change its last safe ceiling.
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(404))) as client:
            snapshot = await registry._sample_scheduler_once(client)
            assert all('metrics_error' in item for item in snapshot['servers'])
            assert first.semaphore.limit == 96 and first.semaphore.max_limit == 192
            assert not registry._scheduler_controllers[urls[0]].samples
        await registry.stop()
    asyncio.run(scenario())


def test_static_registry_starts_and_stops_automatic_scheduler_task(monkeypatch):
    async def scenario():
        registry = DynamicServerRegistry(max_concurrent_per_server=768, request_timeout=60,
            static_urls=['http://first/v1'], scheduler_adaptive_concurrency=True)
        started = asyncio.Event()
        async def poll():
            started.set()
            await asyncio.Event().wait()
        async def close():
            pass
        monkeypatch.setattr(registry, '_scheduler_loop', poll)
        monkeypatch.setattr(registry, '_make_client', lambda url: SimpleNamespace(close=close))
        await registry.start()
        await asyncio.wait_for(started.wait(), 1)
        task = registry._scheduler_task
        await registry.start()
        assert registry._scheduler_task is task
        await registry.stop()
        assert task.cancelled()
    asyncio.run(scenario())


def test_live_ceiling_caps_scheduler_and_does_not_resurrect_retired_endpoints(tmp_path, monkeypatch):
    async def scenario():
        url = 'http://first/v1'
        control, servers = tmp_path / 'limits.json', tmp_path / 'servers.json'
        control.write_text(json.dumps({'max_concurrent_per_server': 768}))
        servers.write_text(json.dumps({'servers': [{'url': url}]}))
        registry = DynamicServerRegistry(max_concurrent_per_server=768, request_timeout=60,
            servers_file=str(servers), concurrency_file=str(control), scheduler_adaptive_concurrency=True)
        async def close():
            pass
        monkeypatch.setattr(registry, '_make_client', lambda url: SimpleNamespace(close=close))
        await registry._reconcile_once()
        entry = (await registry.active_entries())[0]
        control.write_text(json.dumps({'max_concurrent_per_server': 8}))
        await registry._reconcile_once()
        assert entry.semaphore.limit == 8
        clock = [0]
        monkeypatch.setattr(adaptive_controller.time, 'time', lambda: clock[0])
        async with httpx.AsyncClient(transport=httpx.MockTransport(
                lambda request: httpx.Response(200, text=metrics(running=8, cache=.95)))) as client:
            for clock[0] in (0, 15, 30):
                await registry._sample_scheduler_once(client)
            assert entry.semaphore.limit <= 8
            servers.write_text(json.dumps({'servers': []}))
            await registry._reconcile_once()
            snapshot = await registry._sample_scheduler_once(client)
            assert entry.retired and snapshot['servers'] == []
        await registry.stop()
    asyncio.run(scenario())


def test_production_generator_enables_scheduler_by_default(tmp_path, monkeypatch):
    argv = ['generate_synthetic_data.py', '--input-csv', 'sample.csv', '--output-dir', str(tmp_path),
            '--model', 'gemma4-31b', '--server_urls', 'http://first/v1',
            '--max_concurrent_per_server', '768']
    monkeypatch.setattr(sys, 'argv', argv)
    args = generation.parse_args()
    registry = build_registry_from_args(args)
    assert registry._scheduler_enabled
    assert registry._scheduler_metrics_file == str(tmp_path / 'concurrency_status.json')
    assert registry._scheduler_initial == 128 and registry._max_concurrent == 768
    monkeypatch.setattr(sys, 'argv', [*argv, '--no-scheduler-adaptive-concurrency'])
    assert not build_registry_from_args(generation.parse_args())._scheduler_enabled
