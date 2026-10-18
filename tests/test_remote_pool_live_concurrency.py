import asyncio
import json
from types import SimpleNamespace

from remote_vllm_pool import AdaptiveSemaphore, DynamicServerRegistry


def test_resize_drains_requests_and_preserves_error_backoff():
    async def scenario():
        semaphore = AdaptiveSemaphore(2)
        await semaphore.acquire()
        await semaphore.acquire()
        await semaphore.set_max_limit(1)
        waiter = asyncio.create_task(semaphore.acquire())
        await asyncio.sleep(0)
        assert not waiter.done()
        await semaphore.release(success=None)
        await asyncio.sleep(0)
        assert not waiter.done()
        await semaphore.release(success=None)
        await asyncio.wait_for(waiter, 1)
        await semaphore.set_max_limit(8)
        assert semaphore.limit == 8
        await semaphore.release(success=False)
        assert semaphore.limit == 4
        await semaphore.set_max_limit(16)
        assert semaphore.limit == 4 and semaphore.max_limit == 16
    asyncio.run(scenario())


def test_independent_limits_follow_dynamic_endpoints_and_reject_partial_updates(tmp_path, monkeypatch):
    async def scenario():
        urls = ['http://first/v1', 'http://second/v1']
        control = tmp_path / 'concurrency.json'
        servers = tmp_path / 'servers.json'
        control.write_text(json.dumps({'max_concurrent_per_server': 4,
                                      'per_server': {urls[1]: 12}}))
        servers.write_text(json.dumps({'servers': [{'url': urls[0]}]}))
        registry = DynamicServerRegistry(max_concurrent_per_server=16,
            request_timeout=60, servers_file=str(servers), concurrency_file=str(control))
        async def close():
            pass
        monkeypatch.setattr(registry, '_make_client', lambda url: SimpleNamespace(close=close))
        await registry.start()
        try:
            first = (await registry.active_entries())[0]
            assert registry.request_capacity() == 4
            servers.write_text(json.dumps({'servers': [{'url': url} for url in urls]}))
            await registry._reconcile_once()
            entries = {entry.url: entry for entry in await registry.active_entries()}
            second = entries[urls[1]]
            assert entries[urls[0]] is first
            assert second.semaphore.limit == 12 and registry.request_capacity() == 16
            await second.semaphore.acquire()
            await second.semaphore.release(success=False)
            control.write_text(json.dumps({'max_concurrent_per_server': 8,
                                          'per_server': {urls[1]: 16}}))
            await registry._reconcile_once()
            assert first.semaphore.limit == 8
            assert second.semaphore.limit == 6 and second.semaphore.max_limit == 16
            assert registry.request_capacity() == 24
            for invalid in [None, [], {'': 4}, {urls[1]: True}, {urls[1]: 0},
                            {urls[1]: 17}, {urls[1]: '8'}]:
                control.write_text(json.dumps({'max_concurrent_per_server': 2,
                                              'per_server': invalid}))
                await registry._reconcile_once()
                assert registry.request_capacity() == 24
                assert first.semaphore.max_limit == 8
            control.write_text(json.dumps({'max_concurrent_per_server': 4}))
            await registry._reconcile_once()
            assert first.semaphore.limit == second.semaphore.limit == 4
            assert registry.request_capacity() == 8
            servers.write_text(json.dumps({'servers': [{'url': urls[1]}]}))
            await registry._reconcile_once()
            assert first.retired and registry.request_capacity() == 4
        finally:
            await registry.stop()
    asyncio.run(scenario())


def test_registry_reloads_live_concurrency_and_keeps_clients(tmp_path, monkeypatch):
    async def scenario():
        path = tmp_path/'concurrency.json'
        path.write_text(json.dumps({'max_concurrent_per_server': 4}))
        registry = DynamicServerRegistry(max_concurrent_per_server=16,
            request_timeout=60, static_urls=['http://test/v1'], concurrency_file=str(path))
        async def close():
            pass
        client = SimpleNamespace(close=close)
        monkeypatch.setattr(registry, '_make_client', lambda url: client)
        await registry.start()
        try:
            assert registry._reconcile_task is not None
            entry = (await registry.active_entries())[0]
            assert entry.client is client and registry.request_capacity() == 4
            path.write_text(json.dumps({'max_concurrent_per_server': 12}))
            await registry._reconcile_once()
            assert (await registry.active_entries())[0] is entry
            assert entry.semaphore.limit == 12 and registry.request_capacity() == 12
            for invalid in [0, 17, True, '8']:
                path.write_text(json.dumps({'max_concurrent_per_server': invalid}))
                await registry._reconcile_once()
                assert entry.semaphore.limit == 12
            path.write_text('{')
            await registry._reconcile_once()
            assert registry.request_capacity() == 12
            path.unlink()
            await registry._reconcile_once()
            assert registry.request_capacity() == 12
        finally:
            await registry.stop()
    asyncio.run(scenario())
