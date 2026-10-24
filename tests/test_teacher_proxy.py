import asyncio
import json

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from scripts.teacher_proxy import make_app


def test_proxy_long_timeout_and_disconnect_release_slot(tmp_path):
    async def scenario():
        entered, released = asyncio.Event(), asyncio.Event()
        upstream = web.Application()
        async def models(request):
            return web.json_response({'data': [{'id': 'actual', 'root': 'approved'}]})
        async def complete(request):
            body = await request.json()
            assert body['model'] == 'actual'
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                released.set()
        upstream.router.add_get('/v1/models', models)
        upstream.router.add_post('/v1/chat/completions', complete)
        async with TestServer(upstream, handler_cancellation=True) as server:
            config = tmp_path / 'proxy.json'
            config.write_text(json.dumps({'upstream': str(server.make_url('')).rstrip('/'),
                'expected_root': 'approved', 'served_model': 'alias', 'upstream_model': 'actual', 'concurrency': 1}))
            app = make_app(config)
            async with TestClient(TestServer(app, handler_cancellation=True)) as client:
                status = await (await client.get('/status')).json()
                assert status['request_timeout'] == 7260
                assert app['client'].timeout.total == 7260
                request = asyncio.create_task(client.post('/v1/chat/completions', json={'model': 'alias'}))
                try:
                    await asyncio.wait_for(entered.wait(), 3)
                    assert (await (await client.get('/status')).json())['active'] == 1
                finally:
                    request.cancel()
                    await asyncio.gather(request, return_exceptions=True)
                await asyncio.wait_for(released.wait(), 3)
                assert (await (await client.get('/status')).json())['active'] == 0
    asyncio.run(scenario())


def test_proxy_forwards_scheduler_metrics_and_preserves_completion_settings(tmp_path):
    async def scenario():
        upstream = web.Application()
        received = []
        metric_text = 'vllm:kv_cache_usage_perc{engine="0"} 0.42\n'
        async def models(request):
            return web.json_response({'data': [{'id': 'actual', 'root': 'approved'}]})
        async def metrics(request):
            return web.Response(text=metric_text)
        async def complete(request):
            received.append(await request.json())
            return web.json_response({'choices': [{'text': 'saved answer', 'finish_reason': 'stop'}]})
        upstream.router.add_get('/v1/models', models)
        upstream.router.add_get('/metrics', metrics)
        upstream.router.add_post('/v1/completions', complete)
        async with TestServer(upstream) as server:
            config = tmp_path / 'proxy.json'
            config.write_text(json.dumps({'upstream': str(server.make_url('')).rstrip('/'),
                'expected_root': 'approved', 'served_model': 'alias', 'upstream_model': 'actual', 'concurrency': 2}))
            async with TestClient(TestServer(make_app(config))) as client:
                response = await client.get('/metrics')
                assert response.status == 200 and await response.text() == metric_text
                body = dict(model='alias', prompt='Synthetic prompt', max_tokens=20000,
                            temperature=1.0, top_p=.95, top_k=64, min_p=0,
                            presence_penalty=0, repetition_penalty=1, skip_special_tokens=False)
                response = await client.post('/v1/completions', json=body)
                assert response.status == 200
                assert (await response.json())['choices'][0]['text'] == 'saved answer'
                assert received == [{**body, 'model': 'actual'}]
                status = await (await client.get('/status')).json()
                assert status['completed'] == 1 and status['active'] == 0
    asyncio.run(scenario())
