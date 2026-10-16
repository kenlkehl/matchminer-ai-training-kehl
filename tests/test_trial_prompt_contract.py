import importlib.util
from pathlib import Path

import pytest

from tests.test_trial_treatment_screening import endpoint_run, spaces


def test_complete_rendered_prompt_matches_inference_checkout(monkeypatch):
    root = Path(__file__).resolve().parents[2] / 'matchminer-ai-inference-kehl/src/matchminer_ai'
    if not root.is_dir():
        pytest.skip('Cross-repository contract requires the sibling inference-kehl checkout')
    spec = importlib.util.spec_from_file_location('inference_trial_prompt_reference', root/'trials/prompt_builder.py')
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    monkeypatch.setattr(reference, 'load_prompt_text', lambda name: (root/'prompts'/name).read_text())
    document = 'Synthetic trial: any solid tumor; stage IV disease; prior adjuvant therapy allowed.'
    expected = reference.get_filled_trial_prompt(document, 'trial.user.primer.txt', 'trial.user.question.txt')
    assert spaces.build_messages(document) == expected


@pytest.mark.parametrize('change', ['primer', 'question', 'untracked'])
def test_extraction_prompt_change_blocks_reuse_but_preserves_screening(endpoint_run, monkeypatch, change):
    state, path, _ = endpoint_run
    spaces.main()
    screen_manifest = path/'work/treatment_screening/screening_manifest.json'
    screen_identity = screen_manifest.read_bytes()
    screen_shards = {p.name: p.read_bytes() for p in (path/'work/treatment_screening').glob('*.parquet')}
    extraction = path/'work/space_extraction'
    previous_output = (extraction/'remote_output_shard_000000.parquet').read_bytes()
    if change == 'primer':
        monkeypatch.setattr(spaces, 'PROMPT_HEADER', spaces.PROMPT_HEADER + '\nChanged extraction instruction.\n')
    elif change == 'question':
        monkeypatch.setattr(spaces, 'PROMPT_SUFFIX', spaces.PROMPT_SUFFIX + '\nChanged closing instruction.\n')
    else:
        (extraction/'extraction_manifest.json').unlink()
    with pytest.raises(ValueError, match='fresh extraction checkpoints'):
        spaces.main()
    assert len(state.calls) == 4  # No new screening or extraction LLM requests.
    assert screen_manifest.read_bytes() == screen_identity
    assert {p.name: p.read_bytes() for p in (path/'work/treatment_screening').glob('*.parquet')} == screen_shards
    assert (extraction/'remote_output_shard_000000.parquet').read_bytes() == previous_output
