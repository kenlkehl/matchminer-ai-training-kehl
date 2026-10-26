import json
from pathlib import Path

import pytest

from colbert_experiments.data import write_json
from colbert_experiments.label_checkpoints import ResponseStore


def test_resume_preserves_legacy_and_sharded_traces_without_scanning_shards(tmp_path, monkeypatch):
    write_json(tmp_path / 'ab123.json', {'reasoning': 'legacy'})
    write_json(tmp_path / 'cd/cd123.json', {'reasoning': 'sharded'})
    iterdir = Path.iterdir
    def root_only(path):
        assert path == tmp_path
        return iterdir(path)
    monkeypatch.setattr(Path, 'iterdir', root_only)
    store = ResponseStore(tmp_path)
    assert store.read('ab123')[1] == {'reasoning': 'legacy'}
    assert store.read('cd123')[1] == {'reasoning': 'sharded'}
    path, row = store.read('ef123')
    assert row is None
    def no_mkdir(*args, **kwargs):
        raise AssertionError('Checkpoint writes must not mutate parent directories')
    monkeypatch.setattr(Path, 'mkdir', no_mkdir)
    store.write(path, {'reasoning': 'new complete reasoning'})
    assert store.read('ef123')[1] == {'reasoning': 'new complete reasoning'}
    store.write(store.path('ab123'), {'reasoning': 'upgraded legacy'})
    assert json.loads((tmp_path / 'ab123.json').read_text())['reasoning'] == 'upgraded legacy'


def test_corrupt_checkpoint_is_not_treated_as_missing(tmp_path):
    store = ResponseStore(tmp_path)
    store.path('aa123').write_text('{broken')
    with pytest.raises(json.JSONDecodeError):
        store.read('aa123')
