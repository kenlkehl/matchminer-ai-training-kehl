"""Import completed labels from a candidate subset with identical clinical inputs.

An explicit OUTPUT/reuse_sources.json lists {"sources": [{"labels": PATH,
"candidates": PATH}]}. Source labelers may run concurrently with retrieval;
their exclusive locks are awaited before importing durable answers and traces.
"""
from concurrent.futures import ThreadPoolExecutor
import errno
import fcntl
import hashlib
from itertools import islice
import json
import os
from pathlib import Path
import shutil

from .data import digest, read_jsonl, write_json
from .score_journal import ScoreJournal, pending_candidates


def _sha(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def _link_trace(job):
    source, destination, key = job
    path = source / key[:2] / (key + '.json')
    if not path.exists():
        path = source / (key + '.json')
    if not path.is_file():
        raise FileNotFoundError('Imported score is missing its reasoning trace: ' + key)
    target = destination / key[:2] / (key + '.json')
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not os.path.samefile(path, target) and _sha(path) != _sha(target):
            raise ValueError('Conflicting reasoning trace while importing labels: ' + key)
        return
    try:
        os.link(path, target)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        temporary = target.with_suffix('.import.tmp')
        shutil.copy2(path, temporary)
        temporary.replace(target)


def import_completed_sources(output, signature, patients, cohorts, pairs):
    """Caller owns the destination labeling lock; preserve source provenance."""
    output = Path(output)
    plan = json.loads((output / 'reuse_sources.json').read_text())
    sources = plan['sources']
    if not sources:
        raise ValueError('Label reuse requires explicit source directories')
    target_contract = {k: v for k, v in signature.items() if k != 'pairs'}
    journal = None
    existing = {}
    receipts = []
    for entry in sources:
        source = Path(entry['labels']).resolve()
        if source == output.resolve():
            raise ValueError('Cannot import labels from their own destination')
        if not (source / 'labeling.lock').exists():
            raise FileNotFoundError('Source labeling has not started: ' + str(source))
        with (source / 'labeling.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            original = json.loads((source / 'labeling_manifest.json').read_text())
            if {k: v for k, v in original.items() if k != 'pairs'} != target_contract:
                raise ValueError('Label reuse requires identical corpus, prompts, tokenizer and teacher settings')
            candidates = read_jsonl(entry['candidates'])
            if digest(candidates) != original['pairs']:
                raise ValueError('Source candidates differ from their labeling manifest')
            scores = ScoreJournal(source, original).load(patients, cohorts)
            pending_candidates(candidates, scores)
            pending_candidates(pairs, scores)  # Every imported pair must be in final retrieval.
            if journal is None:
                journal = ScoreJournal(output, signature, allow_import=True)
                existing = journal.load(patients, cohorts)
            fresh = [row for key, row in scores.items() if key not in existing]
            jobs = ((source / 'responses', output / 'responses', key)
                    for row in fresh for key in row['question_keys'].values())
            with ThreadPoolExecutor(max_workers=16) as readers:
                while batch := list(islice(jobs, 256)):
                    list(readers.map(_link_trace, batch))
            # Reasoning is accessible under the destination before scores commit.
            if fresh:
                journal.append(fresh)
                existing.update((digest([row['patient_id'], row['cohort_id']]), row) for row in fresh)
            receipts.append(dict(source=str(source), candidates=str(Path(entry['candidates']).resolve()),
                source_manifest=original, completed_source_pairs=len(scores), imported_pairs=len(fresh),
                source_scores_sha256=_sha(source / 'score_journal/scores.jsonl') if scores else None))
    write_json(output / 'reuse_receipt.json', dict(target_manifest=signature, sources=receipts,
                                                 total_completed_pairs=len(existing)))
    return sum(item['imported_pairs'] for item in receipts)
