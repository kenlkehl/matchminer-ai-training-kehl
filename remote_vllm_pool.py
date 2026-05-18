"""
Async dispatcher for vLLM inference against a dynamic, fault-tolerant
pool of OpenAI-compatible servers (vLLM in OpenAI server mode).

Designed for the GCP-orchestrated training pipeline where:
- the set of healthy servers changes over time (spot instances come and go);
- per-server concurrency must back off when servers get overloaded;
- every work item must eventually succeed ("retry until done"); a hard
  ceiling of max_attempts protects against true poison-pill items.

The orchestrator (gcp_vllm_orchestrator.py) writes the canonical list of
healthy server URLs to a JSON file; DynamicServerRegistry re-reads that
file periodically. A fixed snapshot of URLs (--server_urls) is also
supported for ad-hoc runs.

Callers provide work items, a work coroutine, and a shard_writer; this
module owns the queue, semaphores, retries, and dispatch.

Servers JSON file schema (written atomically by the orchestrator):
{
  "updated_at": "2026-05-16T12:00:00Z",
  "ready": true,
  "servers": [
    {"url": "http://10.0.0.5:8000/v1", "instance": "worker-a", "gpus": [0],
     "kind": "remote"},
    ...
  ]
}
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable, List, Optional, Tuple

from openai import AsyncOpenAI


# -------------------------
# CLI helpers
# -------------------------

def add_remote_cli_args(parser: argparse.ArgumentParser) -> None:
    """Attach the standard remote-mode CLI flags to a script's argparse."""
    g = parser.add_argument_group("Remote vLLM pool (GCP orchestrator)")
    g.add_argument(
        "--server_urls", type=str, default=None,
        help="Comma-separated URLs of existing vLLM servers "
             "(e.g. 'http://10.0.0.5:8000/v1,http://10.0.0.6:8000/v1'). "
             "Static snapshot; for a dynamic set use --server_urls_file.",
    )
    g.add_argument(
        "--server_urls_file", type=str, default=None,
        help="Path to a JSON file maintained by gcp_vllm_orchestrator.py "
             "containing the current list of healthy server URLs. The file "
             "is re-read every --server_urls_refresh seconds.",
    )
    g.add_argument(
        "--server_urls_refresh", type=float, default=15.0,
        help="Seconds between re-reads of --server_urls_file.",
    )
    g.add_argument(
        "--max_concurrent_per_server", type=int, default=50,
        help="Per-server concurrency ceiling (adaptive; backs off on errors).",
    )
    g.add_argument(
        "--concurrency_success_threshold", type=int, default=3,
        help="Successful requests needed before increasing a server's adaptive "
             "concurrency limit. Below half of max_concurrent_per_server the "
             "limit doubles per streak (slow start); above it grows by "
             "--concurrency_increase_step.",
    )
    g.add_argument(
        "--concurrency_increase_step", type=int, default=2,
        help="Adaptive concurrency slots to add after each clean success "
             "streak once the limit is past slow-start (>= max_limit/2).",
    )
    g.add_argument(
        "--concurrency_backoff_factor", type=float, default=0.5,
        help="Multiplier applied to a server's adaptive concurrency limit after "
             "a request error.",
    )
    g.add_argument(
        "--results_per_shard", type=int, default=200,
        help="Number of completed items per shard written by the pool.",
    )
    g.add_argument(
        "--request_timeout", type=float, default=600.0,
        help="Per-request timeout in seconds (passed to work_fn).",
    )
    g.add_argument(
        "--max_attempts", type=int, default=200,
        help="Per-item max retries before recording an ERROR placeholder. "
             "Set very high (default 200) for 'retry until done' behavior; "
             "the ceiling exists only as a poison-pill guard.",
    )


def parse_static_urls(server_urls: Optional[str]) -> List[str]:
    if not server_urls:
        return []
    return [u.strip() for u in server_urls.split(",") if u.strip()]


# -------------------------
# Adaptive concurrency
# -------------------------

class AdaptiveSemaphore:
    """
    A condition-variable semaphore whose capacity can shrink (on failure)
    and grow back (on sustained success). Slow-start + AIMD:
      - failure: multiplicative decrease by `backoff_factor` (min 1).
      - success: after `success_threshold` consecutive successes since the
        last decrease, grow the limit. While `limit < max_limit/2` the limit
        doubles (slow start); once at or above that threshold it grows
        additively by `increase_step` up to max_limit.
    """

    def __init__(
        self,
        max_limit: int,
        *,
        success_threshold: int = 3,
        increase_step: int = 2,
        backoff_factor: float = 0.5,
    ):
        self.max_limit = max(1, int(max_limit))
        self.limit = self.max_limit
        self.in_flight = 0
        self._cond = asyncio.Condition()
        self._success_streak = 0
        self._success_threshold = max(1, int(success_threshold))
        self._increase_step = max(1, int(increase_step))
        self._backoff_factor = min(0.99, max(0.01, float(backoff_factor)))
        self._slow_start_ceiling = max(2, self.max_limit // 2)

    async def acquire(self):
        async with self._cond:
            while self.in_flight >= self.limit:
                await self._cond.wait()
            self.in_flight += 1

    async def release(self, *, success: bool):
        async with self._cond:
            self.in_flight = max(0, self.in_flight - 1)
            if success:
                self._success_streak += 1
                if (self._success_streak >= self._success_threshold
                        and self.limit < self.max_limit):
                    if self.limit < self._slow_start_ceiling:
                        new_limit = min(self.max_limit, self.limit * 2)
                    else:
                        new_limit = min(self.max_limit,
                                        self.limit + self._increase_step)
                    self.limit = new_limit
                    self._success_streak = 0
            else:
                old = self.limit
                self.limit = max(1, int(self.limit * self._backoff_factor))
                self._success_streak = 0
                if self.limit != old:
                    # capacity shrank; some waiters may need to re-check
                    pass
            self._cond.notify_all()

    def stats(self) -> dict:
        return {
            "limit": self.limit,
            "max_limit": self.max_limit,
            "in_flight": self.in_flight,
            "streak": self._success_streak,
        }


# -------------------------
# Server registry
# -------------------------

@dataclass
class ServerEntry:
    url: str
    client: AsyncOpenAI
    semaphore: AdaptiveSemaphore
    instance: str = ""
    retired: bool = False
    # rolling stats for status logs
    ok_count: int = 0
    err_count: int = 0
    first_seen_at: float = field(default_factory=time.monotonic)


class DynamicServerRegistry:
    """
    Tracks the current set of vLLM servers. Two modes:
      - static: built from a list of URLs (server_urls); never changes.
      - dynamic: watches a JSON file written by gcp_vllm_orchestrator.py
        and reconciles every `refresh_seconds`.

    Adds new servers when they appear; marks vanished servers `retired`
    (in-flight calls drain, no new dispatches sent). The reconciler is
    started by `start()` and stopped by `stop()`.
    """

    def __init__(
        self,
        *,
        max_concurrent_per_server: int,
        request_timeout: float,
        concurrency_success_threshold: int = 3,
        concurrency_increase_step: int = 2,
        concurrency_backoff_factor: float = 0.5,
        servers_file: Optional[str] = None,
        static_urls: Optional[List[str]] = None,
        refresh_seconds: float = 15.0,
    ):
        if not servers_file and not static_urls:
            raise ValueError("Either servers_file or static_urls must be set.")
        self._servers_file = servers_file
        self._static_urls = list(static_urls) if static_urls else []
        self._refresh_seconds = max(1.0, float(refresh_seconds))
        self._max_concurrent = max_concurrent_per_server
        self._concurrency_success_threshold = concurrency_success_threshold
        self._concurrency_increase_step = concurrency_increase_step
        self._concurrency_backoff_factor = concurrency_backoff_factor
        self._request_timeout = request_timeout

        self._entries: dict[str, ServerEntry] = {}
        self._lock = asyncio.Lock()
        self._change_event = asyncio.Event()
        self._stop_event = asyncio.Event()
        self._reconcile_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        # Idempotent: safe to call multiple times.
        if self._reconcile_task is not None:
            return
        # Always reconcile once synchronously so callers see initial set.
        await self._reconcile_once()
        if self._servers_file:
            self._reconcile_task = asyncio.create_task(self._reconcile_loop())

    async def stop(self) -> None:
        self._stop_event.set()
        if self._reconcile_task:
            self._reconcile_task.cancel()
            try:
                await self._reconcile_task
            except (asyncio.CancelledError, Exception):
                pass
        # Close all clients
        async with self._lock:
            for entry in self._entries.values():
                try:
                    await entry.client.close()
                except Exception:
                    pass

    async def wait_for_change(self, timeout: float) -> bool:
        """Wait until the active server set changes, or timeout."""
        self._change_event.clear()
        try:
            await asyncio.wait_for(self._change_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def active_entries(self) -> List[ServerEntry]:
        async with self._lock:
            return [e for e in self._entries.values() if not e.retired]

    async def _read_servers_file(self) -> Optional[List[dict]]:
        try:
            with open(self._servers_file, "r") as fh:
                data = json.load(fh)
            return list(data.get("servers", []))
        except FileNotFoundError:
            return None
        except (json.JSONDecodeError, OSError):
            # Mid-write; try again next tick
            return None

    def _make_client(self, url: str) -> AsyncOpenAI:
        return AsyncOpenAI(
            base_url=url,
            api_key="not-needed",
            timeout=self._request_timeout + 60,
            max_retries=0,  # we own retry policy
        )

    async def _reconcile_once(self) -> None:
        if self._servers_file:
            servers = await self._read_servers_file()
            if servers is None:
                # File doesn't exist yet; treat as empty set, retry next tick.
                desired = {}
            else:
                desired = {
                    s["url"]: s
                    for s in servers
                    if isinstance(s, dict) and s.get("url")
                }
        else:
            desired = {url: {"url": url, "instance": ""} for url in self._static_urls}

        changed = False
        async with self._lock:
            # Add new
            for url, info in desired.items():
                if url in self._entries:
                    if self._entries[url].retired:
                        # Server came back; un-retire and reset stats
                        self._entries[url].retired = False
                        self._entries[url].first_seen_at = time.monotonic()
                        changed = True
                    continue
                client = self._make_client(url)
                self._entries[url] = ServerEntry(
                    url=url,
                    client=client,
                    semaphore=AdaptiveSemaphore(
                        self._max_concurrent,
                        success_threshold=self._concurrency_success_threshold,
                        increase_step=self._concurrency_increase_step,
                        backoff_factor=self._concurrency_backoff_factor,
                    ),
                    instance=info.get("instance", ""),
                )
                changed = True

            # Retire vanished
            for url, entry in self._entries.items():
                if url not in desired and not entry.retired:
                    entry.retired = True
                    changed = True

        if changed:
            self._change_event.set()

    async def _reconcile_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=self._refresh_seconds,
                    )
                    return
                except asyncio.TimeoutError:
                    pass
                try:
                    await self._reconcile_once()
                except Exception as e:
                    print(f"[remote_vllm_pool] reconcile error: {e}")
        except asyncio.CancelledError:
            raise


# -------------------------
# Result buffering / sharding
# -------------------------

WorkFn = Callable[[AsyncOpenAI, Any], Awaitable[Any]]
ShardWriter = Callable[[List[Tuple[Any, Any]], int], None]


class ResultBuffer:
    """Accumulates completed results and flushes them via shard_writer."""

    def __init__(
        self,
        *,
        results_per_shard: int,
        shard_writer: ShardWriter,
        starting_shard_idx: int = 0,
        requeue_fn: Optional[Callable[[List[Tuple[Any, Any]]], Awaitable[None]]] = None,
        write_retry_attempts: int = 5,
        write_retry_initial_sleep: float = 1.0,
    ):
        self._buf: List[Tuple[Any, Any]] = []
        self._lock = asyncio.Lock()
        self._results_per_shard = max(1, int(results_per_shard))
        self._writer = shard_writer
        self._next_idx = int(starting_shard_idx)
        self.total_written = 0
        # Unique item IDs that have been successfully added (i.e. work_fn
        # returned a result or a poison-pill placeholder was recorded). Used
        # by run_pool to identify items that never finished so it can write
        # placeholders for them rather than wait forever.
        self.completed_ids: set = set()
        # Called with the list of (item_id, result) tuples whose shard write
        # permanently failed; the callback should put them back on the work
        # queue. Without this, failed-shard items would be silently lost
        # (they're already in completed_ids by the time _call_writer runs).
        self._requeue_fn = requeue_fn
        self._write_retry_attempts = max(1, int(write_retry_attempts))
        self._write_retry_initial_sleep = max(0.1, float(write_retry_initial_sleep))

    @property
    def buf_size(self) -> int:
        """Items added but not yet flushed to disk."""
        return len(self._buf)

    async def add(self, item_id: Any, result: Any) -> None:
        flush_payload: Optional[List[Tuple[Any, Any]]] = None
        flush_idx = 0
        async with self._lock:
            self._buf.append((item_id, result))
            self.completed_ids.add(item_id)
            if len(self._buf) >= self._results_per_shard:
                flush_payload = self._buf
                self._buf = []
                flush_idx = self._next_idx
                self._next_idx += 1
        if flush_payload is not None:
            await self._call_writer(flush_payload, flush_idx)

    async def flush(self) -> None:
        async with self._lock:
            if not self._buf:
                return
            flush_payload = self._buf
            self._buf = []
            flush_idx = self._next_idx
            self._next_idx += 1
        await self._call_writer(flush_payload, flush_idx)

    async def _call_writer(self, payload: List[Tuple[Any, Any]], idx: int) -> None:
        # Run sync writer in a thread so disk I/O doesn't block the loop.
        # Retry transient failures (e.g. EMFILE under FD pressure) with
        # exponential backoff. If still failing, hand the payload back to
        # the requeue callback so the items can be re-inferenced; otherwise
        # those rows would be silently lost (they're already in
        # completed_ids by the time we get here).
        last_err: Optional[BaseException] = None
        for attempt in range(self._write_retry_attempts):
            try:
                await asyncio.to_thread(self._writer, payload, idx)
                self.total_written += len(payload)
                return
            except Exception as e:
                last_err = e
                if attempt + 1 < self._write_retry_attempts:
                    sleep_s = self._write_retry_initial_sleep * (2 ** attempt)
                    print(
                        f"[remote_vllm_pool] shard writer failed for "
                        f"shard_idx={idx} (attempt {attempt + 1}/"
                        f"{self._write_retry_attempts}): {e!r}; "
                        f"retrying in {sleep_s:.1f}s"
                    )
                    await asyncio.sleep(sleep_s)
                    continue

        ids = [iid for iid, _ in payload]
        async with self._lock:
            for iid in ids:
                self.completed_ids.discard(iid)
        if self._requeue_fn is not None:
            print(
                f"[remote_vllm_pool] CRITICAL: shard writer failed for "
                f"shard_idx={idx} after {self._write_retry_attempts} attempts "
                f"({last_err!r}); requeuing {len(payload)} item(s) for retry."
            )
            try:
                await self._requeue_fn(payload)
                return
            except Exception as e:
                print(
                    f"[remote_vllm_pool] requeue callback failed for "
                    f"shard_idx={idx}: {e!r}; the {len(payload)} item(s) "
                    f"are now lost."
                )
                raise
        else:
            print(
                f"[remote_vllm_pool] CRITICAL: shard writer failed for "
                f"shard_idx={idx} after {self._write_retry_attempts} attempts "
                f"and no requeue_fn was set; dropping {len(payload)} item(s): "
                f"{last_err!r}"
            )
            raise last_err  # type: ignore[misc]


# -------------------------
# Dispatch core
# -------------------------

async def _dispatch_one(
    entry: ServerEntry,
    item_id: Any,
    payload: Any,
    attempt: int,
    *,
    work_fn: WorkFn,
    work_queue: asyncio.Queue,
    results: ResultBuffer,
    max_attempts: int,
    error_placeholder: Callable[[Any, Exception], Any],
) -> None:
    success = False
    try:
        result = await work_fn(entry.client, payload)
        await results.add(item_id, result)
        entry.ok_count += 1
        success = True
    except Exception as e:
        entry.err_count += 1
        if attempt + 1 >= max_attempts:
            # Poison pill — record an error placeholder so the run can complete.
            print(f"[remote_vllm_pool] item {item_id!r} exhausted "
                  f"{max_attempts} attempts ({e!r}); writing error placeholder.")
            await results.add(item_id, error_placeholder(payload, e))
            success = False  # but don't requeue
        else:
            # Requeue with a tiny stagger so the queue doesn't hot-spin
            # when every server is failing.
            await asyncio.sleep(min(0.1 * (2 ** min(attempt, 6)), 5.0))
            await work_queue.put((item_id, payload, attempt + 1))
    finally:
        await entry.semaphore.release(success=success)


async def _server_loop(
    entry: ServerEntry,
    *,
    work_queue: asyncio.Queue,
    work_fn: WorkFn,
    results: ResultBuffer,
    max_attempts: int,
    error_placeholder: Callable[[Any, Exception], Any],
    stop_event: asyncio.Event,
) -> None:
    in_flight: set[asyncio.Task] = set()
    try:
        while not stop_event.is_set() and not entry.retired:
            try:
                await entry.semaphore.acquire()
            except asyncio.CancelledError:
                break
            # Have a slot; pull an item with a short timeout so we can re-check
            # entry.retired / stop_event regularly.
            try:
                item = await asyncio.wait_for(work_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                # No work right now — give back the slot and loop.
                await entry.semaphore.release(success=True)
                continue
            except asyncio.CancelledError:
                await entry.semaphore.release(success=True)
                break

            item_id, payload, attempt = item
            task = asyncio.create_task(
                _dispatch_one(
                    entry, item_id, payload, attempt,
                    work_fn=work_fn,
                    work_queue=work_queue,
                    results=results,
                    max_attempts=max_attempts,
                    error_placeholder=error_placeholder,
                )
            )
            in_flight.add(task)
            task.add_done_callback(in_flight.discard)
    finally:
        if in_flight:
            await asyncio.gather(*in_flight, return_exceptions=True)


async def _registry_supervisor(
    registry: DynamicServerRegistry,
    *,
    work_queue: asyncio.Queue,
    work_fn: WorkFn,
    results: ResultBuffer,
    max_attempts: int,
    error_placeholder: Callable[[Any, Exception], Any],
    stop_event: asyncio.Event,
) -> None:
    """Spawns and joins one _server_loop per active server, reacting to
    registry changes."""
    running: dict[str, Tuple[ServerEntry, asyncio.Task]] = {}
    try:
        while not stop_event.is_set():
            active = await registry.active_entries()
            active_urls = {e.url for e in active}

            # Start new loops
            for entry in active:
                if entry.url not in running:
                    task = asyncio.create_task(
                        _server_loop(
                            entry,
                            work_queue=work_queue,
                            work_fn=work_fn,
                            results=results,
                            max_attempts=max_attempts,
                            error_placeholder=error_placeholder,
                            stop_event=stop_event,
                        )
                    )
                    running[entry.url] = (entry, task)
                    print(f"[remote_vllm_pool] + server {entry.url}"
                          f" ({entry.instance})")

            # Reap finished or retired loops
            for url in list(running.keys()):
                e, t = running[url]
                if t.done():
                    running.pop(url, None)
                    print(f"[remote_vllm_pool] - server {url} (loop ended)")
                elif e.retired and url not in active_urls:
                    # Already marked retired; the loop will exit on its own.
                    pass

            # Wait for either a registry change or stop
            try:
                await asyncio.wait_for(
                    registry.wait_for_change(timeout=2.0),
                    timeout=2.5,
                )
            except asyncio.TimeoutError:
                pass
            if stop_event.is_set():
                break
    finally:
        # Stop signal already set or supervisor exiting; wait for loops.
        if running:
            await asyncio.gather(*[t for _, t in running.values()],
                                  return_exceptions=True)


async def _status_logger(
    registry: DynamicServerRegistry,
    work_queue: asyncio.Queue,
    results: ResultBuffer,
    *,
    total_items: int,
    interval: float,
    stop_event: asyncio.Event,
) -> None:
    start = time.monotonic()
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
            return
        except asyncio.TimeoutError:
            pass
        try:
            entries = await registry.active_entries()
            pending = work_queue.qsize()
            done = results.total_written
            buf = results.buf_size
            in_flight_sum = sum(e.semaphore.in_flight for e in entries)
            completed = len(results.completed_ids)
            elapsed = time.monotonic() - start
            rate = done / elapsed if elapsed > 0 else 0.0
            per_server = ", ".join(
                f"{e.instance or e.url.split('//')[-1].split('/')[0]}"
                f"[{e.semaphore.in_flight}/{e.semaphore.limit}"
                f" ok={e.ok_count} err={e.err_count}]"
                for e in entries
            ) or "(no active servers)"
            # "missing" = items not in any tracked state. If non-zero with
            # an empty queue, the pool can never make progress without help.
            missing = max(0, total_items - done - pending - in_flight_sum - buf)
            print(
                f"[pool] done={done}/{total_items} pending={pending} "
                f"in_flight={in_flight_sum} buf={buf} missing={missing} "
                f"completed={completed} rate={rate:.1f}/s "
                f"servers: {per_server}"
            )
        except Exception as e:
            print(f"[pool] status logger error: {e}")


# -------------------------
# Public entry point
# -------------------------

def _default_error_placeholder(payload: Any, err: Exception) -> Any:
    return f"ERROR: {err!r}"


async def run_pool(
    work_items: Iterable[Tuple[Any, Any]],
    work_fn: WorkFn,
    registry: DynamicServerRegistry,
    *,
    shard_writer: ShardWriter,
    results_per_shard: int = 200,
    starting_shard_idx: int = 0,
    max_attempts: int = 200,
    status_interval: float = 30.0,
    error_placeholder: Callable[[Any, Exception], Any] = _default_error_placeholder,
    min_servers_warn_secs: float = 300.0,
    stale_giveup_secs: float = 1800.0,
) -> int:
    """
    Drive `work_fn` over `work_items` using a fault-tolerant, dynamic pool
    of vLLM servers from `registry`. Persists results in batches of
    `results_per_shard` via `shard_writer(payload, shard_idx)`.

    work_items: iterable of (item_id, payload). item_id is any hashable;
    payload is opaque to this module and passed straight to work_fn.

    work_fn(client, payload) -> result (any). Raise on transient failure.

    Returns the total number of items written (== len(work_items) modulo
    poison-pill placeholders).
    """
    await registry.start()

    work_queue: asyncio.Queue = asyncio.Queue()
    payload_by_id: dict = {}
    total = 0
    for item_id, payload in work_items:
        work_queue.put_nowait((item_id, payload, 0))
        payload_by_id[item_id] = payload
        total += 1
    if total == 0:
        print("[pool] no work items to process.")
        return 0
    all_ids = set(payload_by_id.keys())

    async def _requeue_failed_shard(payload: List[Tuple[Any, Any]]) -> None:
        # Put items back on the work_queue with attempt=0 so a permanent
        # shard-write failure (e.g. exhausted FD-pressure retries) doesn't
        # silently lose them. Items have already been removed from
        # completed_ids by _call_writer.
        for item_id, _result in payload:
            payload_for_item = payload_by_id.get(item_id)
            if payload_for_item is None:
                # Should not happen: items in a shard payload were always
                # enqueued in this run.
                print(f"[pool] cannot requeue unknown item_id {item_id!r}")
                continue
            await work_queue.put((item_id, payload_for_item, 0))

    results = ResultBuffer(
        results_per_shard=results_per_shard,
        shard_writer=shard_writer,
        starting_shard_idx=starting_shard_idx,
        requeue_fn=_requeue_failed_shard,
    )

    stop_event = asyncio.Event()
    supervisor_task = asyncio.create_task(
        _registry_supervisor(
            registry,
            work_queue=work_queue,
            work_fn=work_fn,
            results=results,
            max_attempts=max_attempts,
            error_placeholder=error_placeholder,
            stop_event=stop_event,
        )
    )
    status_task = asyncio.create_task(
        _status_logger(
            registry, work_queue, results,
            total_items=total,
            interval=status_interval,
            stop_event=stop_event,
        )
    )

    # Wait until everything has been written. We check periodically.
    # `last_progress` measures the last time results.total_written advanced.
    # If it stays put for `stale_giveup_secs` AND the queue is empty, we
    # force-flush any buffered shard and (if needed) write error placeholders
    # for items the pool never recorded — otherwise the loop would hang
    # forever on a tiny number of lost items.
    # We exit when every item has been processed (success or poison-pill
    # placeholder) — i.e. len(completed_ids) >= total. This works regardless
    # of whether shards have been flushed yet, which matters for callers
    # like 6_summarize_patients.py that use a giant results_per_shard and
    # rely on the final flush() in `finally` to write the whole round.
    last_progress = time.monotonic()
    last_warned_at = last_progress
    last_completed = 0
    gave_up = False
    try:
        while len(results.completed_ids) < total and not gave_up:
            await asyncio.sleep(2.0)
            current_completed = len(results.completed_ids)
            if current_completed != last_completed:
                last_completed = current_completed
                last_progress = time.monotonic()
                last_warned_at = last_progress
                continue
            now = time.monotonic()
            stale = now - last_progress
            if stale > 0 and now - last_warned_at >= min_servers_warn_secs:
                active = await registry.active_entries()
                pending = work_queue.qsize()
                in_flight_sum = sum(e.semaphore.in_flight for e in active)
                completed = len(results.completed_ids)
                print(
                    f"[pool] no progress for {stale:.0f}s; "
                    f"{len(active)} active server(s), "
                    f"{pending} items pending in queue, "
                    f"{in_flight_sum} slots in flight, "
                    f"{completed}/{total} items completed."
                )
                last_warned_at = now
            if stale >= stale_giveup_secs and work_queue.qsize() == 0:
                # Flush whatever is in the buffer — these are real completions
                # that just haven't reached the per-shard threshold.
                print(f"[pool] giving up after {stale:.0f}s without progress; "
                      f"flushing buffered shard ({results.buf_size} items).")
                try:
                    await results.flush()
                except Exception as e:
                    print(f"[pool] final flush failed: {e!r}")
                # Any item that never reached results.add is truly lost
                # (e.g. swallowed shard-writer exception earlier in the run).
                # Write error placeholders so the pipeline can advance.
                missing = all_ids - results.completed_ids
                if missing:
                    sample = list(missing)[:5]
                    print(
                        f"[pool] writing error placeholders for "
                        f"{len(missing)} items the pool never recorded "
                        f"(sample ids: {sample!r})."
                    )
                    for item_id in missing:
                        payload = payload_by_id.get(item_id)
                        ph = error_placeholder(
                            payload,
                            RuntimeError(
                                f"pool gave up after {stale:.0f}s without progress"
                            ),
                        )
                        try:
                            await results.add(item_id, ph)
                        except Exception as e:
                            print(f"[pool] failed to record placeholder for "
                                  f"{item_id!r}: {e!r}")
                    try:
                        await results.flush()
                    except Exception as e:
                        print(f"[pool] flush after placeholders failed: {e!r}")
                gave_up = True
    finally:
        stop_event.set()
        await results.flush()
        try:
            await asyncio.wait_for(supervisor_task, timeout=30.0)
        except asyncio.TimeoutError:
            supervisor_task.cancel()
        try:
            await asyncio.wait_for(status_task, timeout=5.0)
        except asyncio.TimeoutError:
            status_task.cancel()

    print(f"[pool] complete: wrote {results.total_written}/{total} items.")
    return results.total_written


# -------------------------
# Standard completion work_fn
# -------------------------

@dataclass
class CompletionSampling:
    """Sampling parameters captured by make_completion_work_fn closures."""
    model: str
    temperature: float = 0.0
    top_k: int = 1
    top_p: float = 1.0
    presence_penalty: float = 0.0
    min_p: float = 0.0
    repetition_penalty: float = 1.1
    request_timeout: float = 600.0


def make_completion_work_fn(
    sampling: CompletionSampling,
    parser_name: str,
    tokenizer,
) -> WorkFn:
    """Returns a work_fn that POSTs a fully-rendered prompt to the OpenAI
    /v1/completions endpoint and parses the reasoning split via
    vllm_reasoning_utils.parse_reasoning_output.

    Payload schema (dict):
      - 'prompt': str          required
      - 'max_tokens': int      required
      - 'temperature': float   optional override
      - 'top_p': float         optional override

    Returns (reasoning, answer) tuples to the shard_writer.
    """
    from vllm_reasoning_utils import parse_reasoning_output  # local import keeps module import-light

    async def work_fn(client: AsyncOpenAI, payload: dict):
        prompt = payload["prompt"]
        max_tokens = int(payload["max_tokens"])
        temperature = float(payload.get("temperature", sampling.temperature))
        top_p = float(payload.get("top_p", sampling.top_p))
        extra = {
            "top_k": sampling.top_k,
            "repetition_penalty": sampling.repetition_penalty,
            "skip_special_tokens": False,
        }
        if sampling.min_p > 0.0:
            extra["min_p"] = sampling.min_p

        response = await asyncio.wait_for(
            client.completions.create(
                model=sampling.model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                presence_penalty=sampling.presence_penalty,
                extra_body=extra,
            ),
            timeout=sampling.request_timeout,
        )
        raw_text = response.choices[0].text
        reasoning, answer = parse_reasoning_output(raw_text, parser_name, tokenizer)
        return (reasoning, answer)

    return work_fn


# -------------------------
# Convenience constructor
# -------------------------

def build_registry_from_args(args) -> DynamicServerRegistry:
    """Build a DynamicServerRegistry from the standard remote CLI flags."""
    if not getattr(args, "server_urls", None) and not getattr(args, "server_urls_file", None):
        raise ValueError(
            "build_registry_from_args called without --server_urls or "
            "--server_urls_file; the script should not have entered remote mode."
        )
    if getattr(args, "server_urls", None) and getattr(args, "server_urls_file", None):
        raise ValueError("Specify only one of --server_urls / --server_urls_file.")
    static_urls = parse_static_urls(getattr(args, "server_urls", None))
    return DynamicServerRegistry(
        max_concurrent_per_server=int(getattr(args, "max_concurrent_per_server", 25)),
        concurrency_success_threshold=int(getattr(args, "concurrency_success_threshold", 3)),
        concurrency_increase_step=int(getattr(args, "concurrency_increase_step", 2)),
        concurrency_backoff_factor=float(getattr(args, "concurrency_backoff_factor", 0.5)),
        request_timeout=float(getattr(args, "request_timeout", 600.0)),
        servers_file=getattr(args, "server_urls_file", None) or None,
        static_urls=static_urls or None,
        refresh_seconds=float(getattr(args, "server_urls_refresh", 15.0)),
    )
