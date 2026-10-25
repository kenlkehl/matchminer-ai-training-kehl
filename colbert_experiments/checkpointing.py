"""Bounded, atomic checkpoints and a coordinated DDP pause request."""
from datetime import datetime
import json
from pathlib import Path
import signal
import time

import torch
import torch.distributed as dist

from .data import write_json


def deadline_timestamp(value):
    """Require an explicit offset so a GPU deadline cannot depend on host TZ."""
    if value is None:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError('stop-at must include a timezone offset')
    return parsed.timestamp()


class PauseRequest:
    def __init__(self, *, stop_at=None, pause_file=None):
        self.deadline = deadline_timestamp(stop_at)
        self.path = Path(pause_file) if pause_file else None
        self.signaled = False

    def __enter__(self):
        self.previous = signal.signal(signal.SIGUSR1, self._signal)
        return self

    def _signal(self, signum, frame):
        self.signaled = True

    def __exit__(self, *exc):
        signal.signal(signal.SIGUSR1, self.previous)

    def requested(self, device, world):
        requested = (self.signaled or (self.path is not None and self.path.exists())
                     or (self.deadline is not None and time.time() >= self.deadline))
        # Every rank calls at the same microbatch boundary, including no_sync
        # microbatches. A signal delivered to just one rank pauses all ranks.
        flag = torch.tensor(int(requested), device=device)
        if world > 1:
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())


def save_checkpoint(output, state, *, limit=2):
    """Keep the committed latest checkpoint until its successor is committed.

    Prune older generations before writing, so even the in-progress payload
    respects the retention bound. A failed save leaves latest.json resumable.
    Legacy checkpoint.pt pointers remain supported when resuming older runs.
    """
    if limit < 2:
        raise ValueError('Checkpoint limit must be at least two for atomic rotation')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    latest = output / 'latest.json'
    previous = json.loads(latest.read_text())['path'] if latest.exists() else None
    path = output / f'checkpoint-{state["step"]:08d}.pt'
    checkpoints = sorted(output.glob('checkpoint-*.pt'), reverse=True)
    legacy = output / 'checkpoint.pt'
    if legacy.exists():
        checkpoints.append(legacy)
    # Re-saving the current step replaces it atomically, still leaving the
    # previous committed file intact until os.replace succeeds.
    keep = {previous} if previous else set()
    for candidate in checkpoints:
        if len(keep) < limit - 1:
            keep.add(candidate.name)
        if candidate.name not in keep:
            candidate.unlink()
    temporary = output / 'checkpoint-writing.tmp'
    try:
        with temporary.open('wb') as stream:
            torch.save(state, stream)
            stream.flush()
            import os
            os.fsync(stream.fileno())
        temporary.replace(path)
        write_json(latest, dict(path=path.name, step=state['step']))
    finally:
        temporary.unlink(missing_ok=True)
