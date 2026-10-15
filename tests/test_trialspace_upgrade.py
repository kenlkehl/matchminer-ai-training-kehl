from copy import deepcopy
import json

import pytest

from oncoreasoning_training.contracts import digest
from oncoreasoning_training.pipeline_upgrade import compatible_identity
from scripts.register_trialspace_upgrade import validate_plan_change
import train_from_summaries as runner


def test_correction_preserves_all_other_stage_commands():
    args = runner.parser().parse_args([])
    new = [vars(stage) for stage in runner.stages(args)]
    old = deepcopy(new)
    fit = next(s for s in old if s['name'] == 'train-trialspace-2')['command']
    fit[fit.index('--base-model') + 1] = args.embedding_model
    fit += ['--input-parquet', str(args.run_dir / 'top_cohorts_tocheck_round1_labels/labels.parquet')]
    validate_plan_change(old, new, args)
    for name in ('mine-2', 'label-top_cohorts_tocheck_round2', 'catalog', 'train-oncoreasoning'):
        changed = deepcopy(new)
        next(s for s in changed if s['name'] == name)['command'].append('--unexpected-change')
        with pytest.raises(ValueError, match='only round 2'):
            validate_plan_change(old, changed, args)


def test_correction_receipt_preserves_original_identity_but_rejects_other_changes(tmp_path, monkeypatch):
    import scripts.register_trialspace_upgrade as upgrade
    previous = {'args': {}, 'inputs': ['recorded'], 'training_commit': 'old'}
    current = {**previous, 'training_commit': 'fixed'}
    (tmp_path / 'trialspace_code_upgrade.json').write_text(json.dumps({
        'kind': 'trialspace-sequential-refinement-v1',
        'original_identity_sha256': digest(previous), 'approved_identity_sha256': digest(current),
    }))
    calls = []
    monkeypatch.setattr(upgrade, 'validate_scope', lambda *args: calls.append(args))
    assert compatible_identity(previous, current, tmp_path, tmp_path) == previous
    assert len(calls) == 1
    for invalid in ({**current, 'inputs': ['changed']}, {**current, 'training_commit': 'other'}):
        with pytest.raises(ValueError, match='changed'):
            compatible_identity(previous, invalid, tmp_path, tmp_path)
