import asyncio

import pytest

from remote_vllm_pool import (
    AdaptiveSemaphore,
    OutputValidationError,
    RequestSession,
    ServerEntry,
)


class Registry:
    def __init__(self):
        self.entry = ServerEntry("synthetic", None, AdaptiveSemaphore(2))
        self.stopped = False

    async def start(self):
        pass

    async def active_entries(self):
        return [self.entry]

    async def wait_for_change(self, timeout):
        await asyncio.sleep(0.01)

    async def stop(self):
        self.stopped = True


def test_session_shares_capacity_retries_json_neutrally_and_accepts_new_work():
    async def scenario():
        registry = Registry()
        running = peak = 0
        attempts = {}

        async def work(client, payload):
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            try:
                attempts[payload] = attempts.get(payload, 0) + 1
                await asyncio.sleep(0.01)
                if payload == "retry" and attempts[payload] == 1:
                    raise OutputValidationError("synthetic malformed JSON")
                return payload
            finally:
                running -= 1

        async with RequestSession(registry, work, max_attempts=2) as session:
            assert await asyncio.gather(
                *(session.submit(x) for x in ["a", "b", "retry"])
            ) == ["a", "b", "retry"]
            assert await session.submit("new child") == "new child"
            assert session.completed == 4
        assert peak == 2 and running == 0
        assert registry.entry.semaphore.limit == 2
        assert registry.entry.output_validation_err_count == 1
        assert registry.stopped

    asyncio.run(scenario())


def test_session_cancellation_releases_requests_and_clients():
    async def scenario():
        registry = Registry()
        running = 0
        started = asyncio.Event()

        async def work(client, payload):
            nonlocal running
            running += 1
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                running -= 1

        async def run():
            async with RequestSession(registry, work) as session:
                await session.submit("blocked")

        task = asyncio.create_task(run())
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)
        assert running == 0 and registry.stopped
        assert registry.entry.semaphore.in_flight == 0

    asyncio.run(scenario())


def test_cancellation_during_normal_session_shutdown_is_not_swallowed():
    async def scenario():
        waiting, exiting = asyncio.Event(), asyncio.Event()

        class SlowRegistry(Registry):
            async def wait_for_change(self, timeout):
                waiting.set()
                await asyncio.Event().wait()

        registry = SlowRegistry()

        async def work(client, payload):
            return payload

        async def run():
            async with RequestSession(registry, work):
                await waiting.wait()
                exiting.set()

        task = asyncio.create_task(run())
        await exiting.wait()
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)
        assert registry.stopped

    asyncio.run(scenario())
