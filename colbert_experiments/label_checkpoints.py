"""Avoid repeated directory mutations on shared checkpoint filesystems."""
import json
from pathlib import Path

from .data import write_json


class ResponseStore:
    """Caller holds the label-directory lock; only this writer creates responses.

    Inspect legacy filenames once and create the 256 shards once. Resume reads
    only requested unfinished questions, without scanning completed shard files.
    A missing shard file costs one open instead of repeated stat/mkdir operations.
    """

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.legacy = {p.stem for p in self.root.iterdir() if p.suffix == '.json'}
        for value in range(256):
            (self.root / f'{value:02x}').mkdir(exist_ok=True)

    def path(self, key):
        return (self.root if key in self.legacy else self.root / key[:2]) / (key + '.json')

    def read(self, key):
        path = self.path(key)
        try:
            raw = path.read_text()
        except FileNotFoundError:
            return path, None
        return path, json.loads(raw)

    def write(self, path, row):
        write_json(path, row, create_parents=False)
