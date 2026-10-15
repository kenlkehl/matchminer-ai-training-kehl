from copy import deepcopy
import json

import pytest

import train_from_summaries as runner
from oncoreasoning_training.contracts import digest
from oncoreasoning_training.pipeline_upgrade import compatible_identity
from scripts.register_downstream_upgrade import KIND, baseline, validate_plan_change


def test_downstream_plan_preserves_all_mining_and_training():
    args = runner.parser().parse_args([])
    new = [vars(stage) for stage in runner.stages(args)]
    old = deepcopy(new)
    for stage in old:
        command = stage['command']
        if stage['name'] in ('catalog', 'prepare-exclusions', 'distill-oncoreasoning'):
            flag = '--max-concurrent-requests' if stage['name'] == 'catalog' else '--concurrency'
            command[command.index(flag) + 1] = '32'
        if stage['name'] == 'prepare-oncoreasoning':
            i = command.index('--pipeline-manifest')
            del command[i:i+2]
    validate_plan_change(old, new, args)
    for name in ('mine-1', 'mine-3', 'train-trialspace-2', 'train-oncoreasoning', 'catalog'):
        changed = deepcopy(new)
        next(s for s in changed if s['name'] == name)['command'].append('--unexpected')
        with pytest.raises(ValueError, match='preserve'):
            validate_plan_change(old, changed, args)


def test_upgrade_receipt_is_bound_to_exact_code_and_identity(tmp_path, monkeypatch):
    previous = {'args': {'teacher_concurrency': '32'}, 'inputs': ['unchanged'], 'training_commit': 'original'}
    base = {**previous, 'training_commit': 'round2-fix'}
    (tmp_path / 'trialspace_code_upgrade.json').write_text(json.dumps({
        'kind': 'trialspace-sequential-refinement-v1', 'to_commit': 'round2-fix',
        'original_identity_sha256': digest(previous), 'approved_identity_sha256': digest(base)}))
    assert baseline(previous, tmp_path) == base
    current = {**base, 'training_commit': 'downstream-fix', 'args': {'teacher_concurrency': '0'}}
    (tmp_path / 'downstream_code_upgrade.json').write_text(json.dumps({
        'kind': KIND, 'original_identity_sha256': digest(previous), 'approved_identity_sha256': digest(current)}))
    called = []
    monkeypatch.setattr('scripts.register_downstream_upgrade.validate_scope', lambda *args: called.append(args))
    assert compatible_identity(previous, current, tmp_path, tmp_path) == previous
    assert len(called) == 1
    for changed in ({**current, 'inputs': ['changed']}, {**current, 'training_commit': 'unapproved'}):
        with pytest.raises(ValueError, match='changed'):
            compatible_identity(previous, changed, tmp_path, tmp_path)
    with pytest.raises(ValueError, match='prior TrialSpace'):
        baseline({**previous, 'inputs': ['changed']}, tmp_path)
