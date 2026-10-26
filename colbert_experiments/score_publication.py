"""Publish complete training scores independently of the large reasoning export."""
import hashlib
import json
import os
from pathlib import Path


FORMAT = 'complete-training-scores-v1'


def fingerprint(path):
    stat = Path(path).stat()
    return dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def input_fingerprints(corpus, candidates):
    return dict(patients=fingerprint(Path(corpus) / 'patients.jsonl'),
                cohorts=fingerprint(Path(corpus) / 'cohorts.jsonl'),
                candidates=fingerprint(candidates))


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
    """Check the publication and immutable-file identities without replaying traces."""
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
    if receipt.get('inputs') != input_fingerprints(corpus, candidates):
        raise ValueError('Corpus or candidate files changed after training-score validation')
    if receipt.get('scores_stat') != fingerprint(output / 'scores.jsonl'):
        raise ValueError('Training scores changed after atomic publication')
    return receipt
