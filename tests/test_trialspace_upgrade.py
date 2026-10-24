from copy import deepcopy
import json

import pytest

from oncoreasoning_training.contracts import digest
from oncoreasoning_training.pipeline_upgrade import compatible_identity
from scripts.register_trialspace_upgrade import validate_plan_change
import train_from_summaries as runner


def test_correction_preserves_all_other_stage_commands():
    args = runner.parser().parse_args([])
    # This narrowly scoped receipt upgrades historical pooled-vector runs.
    # Keep that historical plan explicit now production uses the ColBERT CLI.
    old = [dict(name=name, command=['python', name]) for name in
           ('mine-2', 'label-top_cohorts_tocheck_round2', 'catalog', 'train-oncoreasoning')]
    old.append(dict(name='train-trialspace-2', command=[
        'python', 'finetune_embedder.py', '--base-model', 'google/embeddinggemma-2',
        '--input-parquet', str(args.run_dir / 'top_cohorts_tocheck_round1_labels/labels.parquet'),
        '--input-parquet', str(args.run_dir / 'top_patients_tocheck_round1_labels/labels.parquet')]))
    new = deepcopy(old)
    new[-1]['command'] = ['python', 'finetune_embedder.py', '--base-model',
                         str(args.models_dir / 'trialspace_round1')]
    for direction in ('cohorts', 'patients'):
        new[-1]['command'] += ['--input-parquet', str(args.run_dir / f'top_{direction}_tocheck_round2_labels/labels.parquet')]
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
