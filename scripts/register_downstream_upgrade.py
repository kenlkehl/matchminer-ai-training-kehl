#!/usr/bin/env python3
"""Register downstream teacher fixes without changing completed TrialSpace work."""
import argparse
import ast
from copy import deepcopy
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
from scripts.register_trialspace_upgrade import plan_at

KIND = 'downstream-teacher-recovery-v1'
TRAINING_PATHS = {
    'README.md', 'oncoreasoning_training/README.md', 'train_from_summaries.py',
    'train_good_option_checker.py', 'oncoreasoning_training/contracts.py',
    'oncoreasoning_training/create_all_training_data.py', 'oncoreasoning_training/prepare_boilerplate.py',
    'oncoreasoning_training/teacher.py', 'oncoreasoning_training/scheduling.py',
    'oncoreasoning_training/pipeline_upgrade.py', 'scripts/teacher_proxy.py',
    'scripts/register_downstream_upgrade.py', 'tests/test_teacher_execution.py',
    'tests/test_teacher_proxy.py', 'tests/test_downstream_upgrade.py',
    'tests/test_gemma_pipeline.py', 'tests/test_train_good_option_checker.py',
}
INFERENCE_PATHS = {
    'src/matchminer_ai/llm/endpoint_pool.py', 'src/matchminer_ai/llm/remote_inference.py',
    'src/matchminer_ai/trials/drug_catalog.py', 'tests/test_endpoint_pool.py',
}


def arguments(identity):
    import train_from_summaries as runner
    args = runner.parser().parse_args([])
    for name, value in identity['args'].items():
        default = getattr(args, name)
        if isinstance(default, Path):
            value = Path(value)
        elif isinstance(default, int):
            value = int(value)
        elif isinstance(default, list):
            value = ast.literal_eval(value)
        setattr(args, name, value)
    return args


def validate_plan_change(old, new, args):
    expected = deepcopy(old)
    for stage in expected:
        command = stage['command']
        if stage['name'] in ('catalog', 'prepare-exclusions', 'distill-oncoreasoning'):
            flag = '--max-concurrent-requests' if stage['name'] == 'catalog' else '--concurrency'
            command[command.index(flag) + 1] = str(args.teacher_concurrency)
        if stage['name'] == 'prepare-oncoreasoning':
            index = command.index('--candidates')
            command[index:index] = ['--pipeline-manifest', str(args.run_dir / 'pipeline_manifest.json')]
    if expected != new:
        raise ValueError('Upgrade must preserve all mining, labeling, training and task settings')


def baseline(previous, run_dir):
    """Follow the already registered round 2 correction, not the original bug."""
    receipt = json.loads((Path(run_dir) / 'trialspace_code_upgrade.json').read_text())
    base = {**previous, 'training_commit': receipt['to_commit']}
    if (receipt.get('kind') != 'trialspace-sequential-refinement-v1'
            or receipt.get('original_identity_sha256') != digest(previous)
            or receipt.get('approved_identity_sha256') != digest(base)):
        raise ValueError('Missing valid prior TrialSpace correction')
    return base


def validate_scope(repo, previous, current, run_dir):
    base = baseline(previous, run_dir)
    expected = {**base, 'training_commit': current['training_commit'],
                'inference_commit': current['inference_commit'],
                'args': {**base['args'], 'teacher_concurrency': '0'}}
    if current != expected:
        raise ValueError('Downstream upgrade cannot change source inputs, rubric, models or other settings')
    changed = git(repo, 'diff', '--name-only', base['training_commit'], current['training_commit']).splitlines()
    if not changed or set(changed) - TRAINING_PATHS:
        raise ValueError('Training changes outside downstream upgrade scope')
    inference = Path(current['args']['inference_repo'])
    inference_changed = git(inference, 'diff', '--name-only', base['inference_commit'], current['inference_commit']).splitlines()
    if not inference_changed or set(inference_changed) - INFERENCE_PATHS:
        raise ValueError('Inference changes outside teacher transport scope')
    git(repo, 'diff', '--exit-code', 'HEAD', '--')
    # Protect the files used by this upgrade without touching unrelated user edits.
    git(inference, 'diff', '--exit-code', 'HEAD', '--', *sorted(INFERENCE_PATHS))
    validate_plan_change(plan_at(repo, base['training_commit'], arguments(base)),
                         plan_at(repo, current['training_commit'], arguments(current)), arguments(current))
    return changed, inference_changed


def register(repo, run_dir):
    import train_from_summaries as runner
    run_dir = Path(run_dir).resolve()
    with (run_dir / 'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = json.loads((run_dir / 'pipeline_manifest.json').read_text())
        current = {**previous, 'training_commit': git(repo, 'rev-parse', 'HEAD'),
                   'inference_commit': runner.INFERENCE_COMMIT,
                   'args': {**previous['args'], 'teacher_concurrency': '0'}}
        changed, inference_changed = validate_scope(repo, previous, current, run_dir)
        stamps = list((run_dir / 'completed').glob('*.json'))
        allowed = {stage['name'] for stage in plan_at(repo, current['training_commit'], arguments(current))
                   if stage['name'].startswith(('mine-', 'label-', 'train-trialspace-'))}
        for stamp in stamps:
            marker = json.loads(stamp.read_text())
            if (stamp.stem not in allowed or marker['fingerprint'] != digest(previous)
                    or not all(Path(p).exists() for p in marker['outputs'])):
                raise ValueError('Register before downstream stages have started')
        for name in ('good_option_catalog', 'catalog_checkpoints', 'boilerplate_components', 'oncoreasoning'):
            path = run_dir / name
            if path.exists() and any(path.iterdir()):
                raise ValueError('Downstream output already exists: ' + name)
        receipt = {'kind': KIND, 'original_identity_sha256': digest(previous),
                   'approved_identity_sha256': digest(current),
                   'from_commit': baseline(previous, run_dir)['training_commit'],
                   'to_commit': current['training_commit'], 'inference_commit': current['inference_commit'],
                   'changed_paths': changed, 'inference_changed_paths': inference_changed,
                   'preserved_stages': sorted(p.stem for p in stamps),
                   'registered_at': dt.datetime.now(dt.timezone.utc).isoformat()}
        atomic_json(run_dir / 'downstream_code_upgrade.json', receipt)
        return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    receipt = register(ROOT, args.run_dir)
    print(f"Registered downstream fixes {receipt['from_commit']} -> {receipt['to_commit']}")
