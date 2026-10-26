"""Publish complete training scores independently of the large reasoning export."""
import hashlib
import json
import os
from pathlib import Path


FORMAT = 'complete-training-scores-v1'


def fingerprint(path):
    stat = Path(path).stat()
    return dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def content_fingerprint(path):
    """Portable byte identity; timestamps only detect writes during this read."""
    before = fingerprint(path)
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024):
            checksum.update(block)
    if fingerprint(path) != before:
        raise ValueError('File changed during content verification: ' + str(path))
    return dict(**before, sha256=checksum.hexdigest())


def input_fingerprints(corpus, candidates):
    return dict(patients=content_fingerprint(Path(corpus) / 'patients.jsonl'),
                cohorts=content_fingerprint(Path(corpus) / 'cohorts.jsonl'),
                candidates=content_fingerprint(candidates))


def publish_scores(output, signature, scores, pairs, *, inputs):
    """Called with validated journal/aggregate records while holding labeling.lock."""
    from .data import write_json
    from .regression import score_balance
    from .score_journal import pending_candidates
    if pending_candidates(pairs, scores):
        raise ValueError('Cannot publish training scores: candidate labels are incomplete')
    output = Path(output)
    target = output / 'scores.jsonl'
    temporary = target.with_suffix('.jsonl.tmp')
    checksum = hashlib.sha256()
    questions = 0
    with temporary.open('wb') as stream:
        for key in sorted(scores):
            row = scores[key]
            raw = (json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n').encode()
            stream.write(raw)
            checksum.update(raw)
            questions += len(row['question_keys'])
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(target)
    write_json(output / 'score_balance.json', score_balance(list(scores.values())))
    receipt = dict(format=FORMAT, complete=True, pairs=len(scores), questions=questions,
                   teacher_manifest=signature, inputs=inputs, scores_file='scores.jsonl',
                   scores_sha256=checksum.hexdigest(), scores_stat=fingerprint(target),
                   reasoning_source='responses', candidate_coverage_verified=True)
    write_json(output / 'scores_ready.json', receipt)
    return receipt


def ready_scores(output, *, corpus, candidates, expected_pairs=None):
    """Verify published content across mounts without replaying reasoning traces."""
    output = Path(output)
    path = output / 'scores_ready.json'
    if not path.exists():
        return None
    receipt = json.loads(path.read_text())
    signature = json.loads((output / 'labeling_manifest.json').read_text())
    if (receipt.get('format') != FORMAT or receipt.get('complete') is not True
            or receipt.get('candidate_coverage_verified') is not True
            or receipt.get('scores_file') != 'scores.jsonl'
            or receipt.get('teacher_manifest') != signature):
        raise ValueError('Training-score publication has an invalid or changed teacher identity')
    if expected_pairs is not None and receipt.get('pairs') != expected_pairs:
        raise ValueError('Published training-score count differs from the expected complete candidate set')
    paths = dict(patients=Path(corpus) / 'patients.jsonl', cohorts=Path(corpus) / 'cohorts.jsonl',
                 candidates=Path(candidates), scores=output / 'scores.jsonl')
    recorded = {**receipt.get('inputs', {}),
                'scores': {**receipt.get('scores_stat', {}), 'sha256': receipt.get('scores_sha256')}}
    for name, filename in paths.items():
        expected = recorded.get(name, {})
        if not expected.get('sha256'):
            raise ValueError('Training-score receipt needs portable content hashes: ' + name)
        actual = content_fingerprint(filename)
        if any(actual.get(key) != expected.get(key) for key in ('bytes', 'sha256')):
            raise ValueError(f'Training data content differs from the validated publication: {filename}; '
                             f'expected bytes={expected.get("bytes")}, sha256={expected.get("sha256")}; '
                             f'observed bytes={actual["bytes"]}, sha256={actual["sha256"]}')
    return receipt
