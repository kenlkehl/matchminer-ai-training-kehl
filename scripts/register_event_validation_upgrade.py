#!/usr/bin/env python3
"""Register the validation-only correction for a stopped event-generation run.

Prompts, seeds, sampling, input data, and the original patient-ID namespace stay
fixed. Only the generator source may change. Both writer locks must be free.
"""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from generate_synthetic_data import VALIDATION_REVISION, now, save_json, sha_file


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def register(run_dir, pipeline_dir=None):
    from run_pipeline import file_record

    run_dir = Path(run_dir).resolve()
    pipeline_dir = Path(pipeline_dir).resolve() if pipeline_dir else run_dir / 'pipeline'
    with ExitStack() as stack:
        for path in (pipeline_dir / 'pipeline.lock', run_dir / 'generation.lock'):
            lock = stack.enter_context(path.open('r'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (pipeline_dir / 'completed/generate-synthetic.json').exists() or any(
                (run_dir / name).exists() for name in ('note_input_manifest.json', 'all_synthetic_notes.parquet')):
            raise ValueError('Register during event generation, before generating notes')
        original = json.loads((run_dir / 'manifest.json').read_text())
        if original['version'] != 'synthetic-cohort-seeds-patient-events-notes-v5':
            raise ValueError('Unsupported generation contract')
        approved = json.loads(json.dumps(original))
        for name, previous_hash in original['source_sha256'].items():
            current_hash = sha_file(ROOT / name)
            if name != 'generate_synthetic_data.py' and current_hash != previous_hash:
                raise ValueError('Validation upgrade cannot change ' + name)
            approved['source_sha256'][name] = current_hash
        if approved == original:
            raise ValueError('No generator validation change to register')

        request_path = pipeline_dir / 'requests/generate-synthetic.json'
        receipt_path = run_dir / 'event_validation_upgrade.json'
        backup = run_dir / 'event_validation_upgrade_backup'
        prior_request = backup / 'pipeline_request.json'
        request = json.loads((prior_request if prior_request.exists() else request_path).read_text())
        identity = {key: request[key] for key in ('command', 'inputs', 'sources')}
        if digest(identity) != request['fingerprint']:
            raise ValueError('Invalid original pipeline request fingerprint')
        cache = {}
        current = {**identity,
                   'inputs': [file_record(row['path'], cache) for row in identity['inputs']],
                   'sources': [file_record(row['path'], cache) for row in identity['sources']]}
        if current['inputs'] != identity['inputs']:
            raise ValueError('Validation upgrade cannot change generation inputs')
        driver = str(ROOT / 'generate_synthetic_data.py')
        if sum(row['path'] == driver for row in identity['sources']) != 1:
            raise ValueError('Missing or duplicated generator source identity')
        for old, new in zip(identity['sources'], current['sources']):
            if old['path'] == driver:
                if old['sha256'] != original['source_sha256']['generate_synthetic_data.py']:
                    raise ValueError('Generator source differs between manifest and pipeline request')
            elif old != new:
                raise ValueError('Validation upgrade cannot change pipeline source ' + old['path'])
        updated_request = dict(fingerprint=digest(current), **current)
        existing = json.loads(request_path.read_text())
        if existing not in (request, updated_request):
            raise ValueError('Pipeline request changed outside this upgrade')
        receipt = dict(kind=VALIDATION_REVISION, registered_at=now(),
                       original_manifest_sha256=digest(original), approved_manifest_sha256=digest(approved),
                       original_request_sha256=request['fingerprint'],
                       approved_request_sha256=updated_request['fingerprint'],
                       original_generator_sha256=original['source_sha256']['generate_synthetic_data.py'],
                       approved_generator_sha256=approved['source_sha256']['generate_synthetic_data.py'],
                       preserved_patient_identity=digest(original),
                       changed_sources=['generate_synthetic_data.py'])
        if receipt_path.exists():
            saved = json.loads(receipt_path.read_text())
            if {**saved, 'registered_at': receipt['registered_at']} != receipt:
                raise ValueError('A different validation upgrade is already registered')
            receipt = saved
        backup.mkdir(exist_ok=True)
        if not prior_request.exists():
            save_json(prior_request, request)
            save_json(backup / 'manifest.json', original)
            save_json(backup / 'failures.json', [json.loads(p.read_text())
                      for p in sorted((run_dir / 'generation_failures').glob('*.json'))])
            save_json(backup / 'status.json', json.loads((run_dir / 'status.json').read_text()))
        # The receipt is durable before either coordinator can resume. Keep the
        # original generation manifest byte-for-byte; it determines patient IDs.
        save_json(receipt_path, receipt)
        save_json(request_path, updated_request)
        return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--pipeline-dir', type=Path)
    args = parser.parse_args()
    print(json.dumps(register(args.run_dir, args.pipeline_dir), indent=2))
