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
