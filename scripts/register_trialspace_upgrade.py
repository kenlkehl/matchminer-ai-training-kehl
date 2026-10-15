#!/usr/bin/env python3
"""Register the round 2 procedure correction while preserving completed work."""
import argparse
import ast
import datetime as dt
import fcntl
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from oncoreasoning_training.contracts import digest
from oncoreasoning_training.create_all_training_data import atomic_json
from oncoreasoning_training.pipeline_upgrade import git

ALLOWED_PATHS = {
    'README.md', 'train_from_summaries.py', 'finetune_embedder.py',
    'scripts/remote_trialspace.py', 'scripts/cloud_trialspace_job.py',
    'scripts/register_trialspace_upgrade.py', 'oncoreasoning_training/pipeline_upgrade.py',
    'tests/test_gemma_pipeline.py', 'tests/test_remote_trialspace.py',
    'tests/test_trialspace_upgrade.py',
}


def plan_at(repo, revision, args):
    """Evaluate only the stage builder from the explicitly selected Git revision."""
    import train_from_summaries as runner
    tree = ast.parse(git(repo, 'show', f'{revision}:train_from_summaries.py'))
    stage_function = next(node for node in tree.body
                          if isinstance(node, ast.FunctionDef) and node.name == 'stages')
    namespace = {'sys': sys, 'Stage': runner.Stage}
    exec(compile(ast.Module(body=[stage_function], type_ignores=[]), '<recorded-stage-plan>', 'exec'), namespace)
    return [vars(stage) for stage in namespace['stages'](args)]


def validate_plan_change(old, new, args):
    expected = []
    for stage in old:
        stage = {**stage, 'command': list(stage['command'])}
        if stage['name'] == 'train-trialspace-2':
            command = stage['command']
            command[command.index('--base-model') + 1] = str(args.models_dir / 'trialspace_round1')
            prefix = command[:command.index('--input-parquet')]
            for direction in ('cohorts', 'patients'):
                prefix += ['--input-parquet', str(args.run_dir / f'top_{direction}_tocheck_round2_labels/labels.parquet')]
            stage['command'] = prefix
        expected.append(stage)
    if expected != new:
        raise ValueError('Upgrade must change only round 2 starting weights and round 2 label inputs')


def validate_scope(repo, previous, current):
    import train_from_summaries as runner
    if {**previous, 'training_commit': current['training_commit']} != current:
        raise ValueError('TrialSpace correction cannot change inputs, rubric, or settings')
    changed = git(repo, 'diff', '--name-only', previous['training_commit'], current['training_commit']).splitlines()
    if not changed or set(changed) - ALLOWED_PATHS:
        raise ValueError('Changes outside the TrialSpace correction scope')
    git(repo, 'diff', '--exit-code', 'HEAD', '--')
    args = runner.parser().parse_args([])
    for name, value in previous['args'].items():
        default = getattr(args, name)
        if isinstance(default, Path):
            value = Path(value)
        elif isinstance(default, int):
            value = int(value)
        elif isinstance(default, list):
            value = ast.literal_eval(value)
        setattr(args, name, value)
    validate_plan_change(plan_at(repo, previous['training_commit'], args),
                         plan_at(repo, current['training_commit'], args), args)
    return changed


def register(repo, run_dir):
    run_dir = Path(run_dir).resolve()
    with (run_dir / 'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = json.loads((run_dir / 'pipeline_manifest.json').read_text())
        current = {**previous, 'training_commit': git(repo, 'rev-parse', 'HEAD')}
        changed = validate_scope(repo, previous, current)
        expected = {'mine-1', 'mine-2', 'train-trialspace-1',
                    'label-top_cohorts_tocheck_round1', 'label-top_patients_tocheck_round1',
                    'label-top_cohorts_tocheck_round2', 'label-top_patients_tocheck_round2'}
        stamps = list((run_dir / 'completed').glob('*.json'))
        if {p.stem for p in stamps} != expected:
            raise ValueError('Register only after round 2 labeling, before completing round 2 training')
        for stamp in stamps:
            marker = json.loads(stamp.read_text())
            if marker['fingerprint'] != digest(previous) or not all(Path(p).exists() for p in marker['outputs']):
                raise ValueError('Invalid completed stage: ' + stamp.stem)
        models = Path(previous['args']['models_dir'])
        for name in ('trialspace_round2', 'trialspace_round2_checkpoints'):
            if (models / name).exists():
                raise ValueError('Archive the incorrect round 2 outputs/checkpoints before registering')
        receipt = {'kind': 'trialspace-sequential-refinement-v1',
                   'original_identity_sha256': digest(previous),
                   'approved_identity_sha256': digest(current),
                   'from_commit': previous['training_commit'], 'to_commit': current['training_commit'],
                   'changed_paths': changed, 'preserved_stages': sorted(expected),
                   'registered_at': dt.datetime.now(dt.timezone.utc).isoformat()}
        atomic_json(run_dir / 'trialspace_code_upgrade.json', receipt)
        return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    receipt = register(ROOT, args.run_dir)
    print(f"Registered round 2 correction {receipt['from_commit']} -> {receipt['to_commit']}")
