import asyncio

import pytest

from colbert_experiments import label_concurrency as concurrency


def test_worker_target_supports_cloud_pool_and_explicit_small_runs():
    assert concurrency.worker_target(0, 64 * 96 + 320 + 5 * 32) == 8280
    assert concurrency.worker_target(0, 100000) == 16384
    assert concurrency.worker_target(0, 0) == 1
    assert concurrency.worker_target(3, 100000, maximum=2) == 3
    with pytest.raises(ValueError):
        concurrency.worker_target(-1, 4)
    with pytest.raises(ValueError):
        concurrency.worker_target(0, 4, maximum=0)


def test_dynamic_workers_grow_and_drain_without_losing_or_cancelling_pairs(monkeypatch):
    monkeypatch.setattr(concurrency, 'reserve_descriptors', lambda target: None)

    async def scenario():
        slots, targets, started, completed, cancelled = [1], [], [], [], []
        release = asyncio.Queue()

        async def process(item):
            started.append(item)
            try:
                await release.get()
                completed.append(item)
            except asyncio.CancelledError:
                cancelled.append(item)
                raise

        async def wait_for(predicate):
            async def poll():
                while not predicate():
                    await asyncio.sleep(.001)
            await asyncio.wait_for(poll(), 2)

        run = asyncio.create_task(concurrency.run_workers(range(12), process, lambda: slots[0],
                                  maximum=10, on_resize=targets.append, poll_seconds=.001))
        await wait_for(lambda: len(started) == 2)
        slots[0] = 4
        await wait_for(lambda: len(started) == 5)
        slots[0] = 0
        await wait_for(lambda: targets[-1] == 1)
        assert len(started) == 5 and not cancelled  # In-flight chains are retained.
        for _ in range(5):
            release.put_nowait(None)
        await wait_for(lambda: len(started) == 6)
        await asyncio.sleep(.01)
        assert len(started) == 6  # New work now obeys the smaller pool.
        for _ in range(12):
            release.put_nowait(None)
        await asyncio.wait_for(run, 2)
        assert sorted(completed) == list(range(12)) and len(set(started)) == 12
        assert not cancelled and targets == [2, 5, 1]

    asyncio.run(scenario())


def test_explicit_worker_limit_does_not_consult_pool_capacity(monkeypatch):
    monkeypatch.setattr(concurrency, 'reserve_descriptors', lambda target: None)

    async def scenario():
        active = peak = 0
        completed = []

        async def process(item):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(.001)
            completed.append(item)
            active -= 1

        await concurrency.run_workers(range(10), process, lambda: pytest.fail('Fixed override ignored'),
                                      workers=2, maximum=1, poll_seconds=.001)
        assert peak == 2 and sorted(completed) == list(range(10))

    asyncio.run(scenario())


@pytest.mark.parametrize('error', [RuntimeError('failed question'), asyncio.CancelledError()])
def test_worker_failure_or_cancellation_propagates_and_closes_siblings(monkeypatch, error):
    monkeypatch.setattr(concurrency, 'reserve_descriptors', lambda target: None)

    async def scenario():
        active = set()

        async def process(item):
            active.add(item)
            try:
                await asyncio.sleep(0)
                if item == 0:
                    raise error
                await asyncio.Future()
            finally:
                active.remove(item)

        with pytest.raises(type(error)):
            await concurrency.run_workers(range(10), process, lambda: 2, workers=2)
        assert not active

    asyncio.run(scenario())


def test_descriptor_reservation_stays_within_existing_hard_limit(monkeypatch):
    changes = []
    monkeypatch.setattr(concurrency.resource, 'getrlimit', lambda kind: (1024, 65536))
    monkeypatch.setattr(concurrency.resource, 'setrlimit', lambda kind, limits: changes.append(limits))
    concurrency.reserve_descriptors(8192)
    assert changes == [(18432, 65536)]
    monkeypatch.setattr(concurrency.resource, 'getrlimit', lambda kind: (1024, 4096))
    with pytest.raises(RuntimeError, match='open-file limit'):
        concurrency.reserve_descriptors(8192)
