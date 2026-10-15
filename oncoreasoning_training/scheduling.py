"""Bounded continuous scheduling with serial-summary dependencies."""
from collections import defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait


def run_tasks(tasks, worker, *, concurrency, progress=None):
    """Refill a slot as soon as any request finishes; retain parent ordering.

    Requests are topologically ordered (summary chunks are prepared in rounds).
    A small bounded lookahead lets unrelated patients proceed when a child's
    parent is still running. No completed-ID set grows with the whole corpus.
    """
    pending, active_ids, waiting = {}, set(), defaultdict(list)
    ready, source = deque(), iter(tasks)
    exhausted, completed = False, 0
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        while pending or ready or waiting or not exhausted:
            while len(pending) < concurrency:
                if ready:
                    task = ready.popleft()
                elif not exhausted and len(pending) + sum(map(len, waiting.values())) < 2 * concurrency:
                    try:
                        task = next(source)
                    except StopIteration:
                        exhausted = True
                        continue
                    if task.get('parent_id') in active_ids:
                        waiting[task['parent_id']].append(task)
                        active_ids.add(task['id'])
                        continue
                else:
                    break
                future = executor.submit(worker, task)
                pending[future] = task['id']
                active_ids.add(task['id'])
            if not pending:
                if waiting:
                    raise ValueError('Unresolved summary dependency')
                continue
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                future.result()
                identifier = pending.pop(future)
                active_ids.remove(identifier)
                ready.extend(waiting.pop(identifier, []))
                completed += 1
                if progress:
                    progress(completed)
    return completed
