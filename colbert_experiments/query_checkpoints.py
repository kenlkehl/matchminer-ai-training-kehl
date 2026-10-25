"""Atomic query batches avoid one network-filesystem transaction per query.

Older per-query checkpoints remain readable. An interrupted batch is recomputed;
published batches are immutable and retain all candidate provenance and scores.
"""
import json
from pathlib import Path

from .data import digest, write_json


class QueryCheckpoints:
    def __init__(self, root):
        self.root = Path(root)
        self.legacy = {p.stem: p for p in (self.root / 'queries').glob('*/*.json')}
        self.keys = set(self.legacy)
        for path in sorted((self.root / 'query_batches').glob('*.json')):
            rows = self.read_batch(path)
            if self.keys.intersection(rows):
                raise ValueError('Duplicate query checkpoints')
            self.keys.update(rows)

    @staticmethod
    def read_batch(path):
        payload = json.loads(path.read_text())
        rows = payload['queries']
        if payload.get('version') != 1 or not rows or path.stem != digest(sorted(rows)):
            raise ValueError(f'Invalid query checkpoint batch: {path}')
        return rows

    def write_batch(self, rows):
        if not rows or self.keys.intersection(rows):
            raise ValueError('Query checkpoint batch must contain only new queries')
        write_json(self.root / 'query_batches' / f'{digest(sorted(rows))}.json',
                   dict(version=1, queries=rows))
        self.keys.update(rows)

    def ordered_hits(self, keys):
        # Called after the distributed barrier. Read each file once; restoring
        # original query order makes output independent of rank/batch scheduling.
        rows = {key: json.loads(path.read_text()) for key, path in self.legacy.items()}
        for path in sorted((self.root / 'query_batches').glob('*.json')):
            batch = self.read_batch(path)
            if rows.keys() & batch.keys():
                raise ValueError('Duplicate query checkpoints')
            rows.update(batch)
        return [hit for key in keys for hit in rows[key]]
