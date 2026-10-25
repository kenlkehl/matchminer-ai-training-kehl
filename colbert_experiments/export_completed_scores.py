"""Pack live question checkpoints once on the producing host for sequential NFS reads.

python colbert_experiments/export_completed_scores.py --source-run RUN
Writes only a new completed_scores archive; existing responses and reasoning stay intact.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
from itertools import islice
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SIBLING = ROOT.parent / 'matchminer-ai-inference-kehl/src'
if SIBLING.exists():
    sys.path.insert(0, str(SIBLING))

from colbert_experiments.contracts import TASKS, contract_hash, judge_messages, parse_judgment, prompt_sources
from colbert_experiments.data import digest, load_corpus, write_json
from colbert_experiments.regression import GATE, SPECIFICITY, score_contract

TASK_ORDER = tuple(TASKS)
DEFAULT_RUN = ROOT.parent / 'data/no_phi/synthetic_cohorts_5000_gemma4_31b_20261024/post_notes_colbert_20261025'
_CONTEXT = None
_READERS = None


def score_from_answers(patient_id, cohort_id, split, values, question_contract, aggregate_contract):
    answers = dict(zip(TASK_ORDER, values))
    if answers.get(GATE) is None:
        return None
    asked = [GATE] + (list(SPECIFICITY) if answers[GATE] == 'yes' else [])
    if any(answers[t] is None for t in asked):
        return None
    answers = {task: answers[task] if task in asked else None for task in TASK_ORDER}
    score = sum(answer == 'yes' for answer in answers.values())
    return dict(patient_id=patient_id, cohort_id=cohort_id, split=split,
        score=score, normalized_score=score/5, answers=answers, asked_tasks=asked,
        skipped_tasks=[task for task in TASK_ORDER if task not in asked],
        question_keys={task: digest([patient_id, cohort_id, task]) for task in asked},
        question_contract=question_contract, contract=aggregate_contract)


def validate_score(row, patients, cohorts, question_contract, aggregate_contract):
    """Validate the small aggregate record, without reopening its source traces."""
    p, c = patients[row['patient_id']], cohorts[row['cohort_id']]
    if p['split'] != c['split'] or row['split'] != p['split']:
        raise ValueError('Compact score crosses effective corpus splits')
    answers = row['answers']
    if set(answers) != set(TASK_ORDER) or answers[GATE] not in ('yes', 'no'):
        raise ValueError('Invalid compact-score answers')
    if any(value not in ('yes', 'no', None) for value in answers.values()):
        raise ValueError('Invalid compact-score answer value')
    expected = score_from_answers(p['id'], c['id'], row['split'], [answers[t] for t in TASK_ORDER],
                                  question_contract, aggregate_contract)
    if expected is None or type(row.get('score')) is not int or any(row.get(k) != v for k, v in expected.items()):
        raise ValueError('Compact score is incomplete or violates the gated score contract')


def initialize_reader(corpus, question_contract, cutoff_ns, reader_threads):
    global _CONTEXT, _READERS
    patients, cohorts = load_corpus(corpus)
    _CONTEXT = ({p['id']: p for p in patients}, {c['id']: c for c in cohorts},
                question_contract, cutoff_ns, prompt_sources())
    _READERS = ThreadPoolExecutor(max_workers=reader_threads)


def read_one(path):
    patients, cohorts, question_contract, cutoff_ns, sources = _CONTEXT
    with open(path, 'rb') as stream:
        if os.fstat(stream.fileno()).st_mtime_ns > cutoff_ns:
            return None, 0
        raw = stream.read()
    row = json.loads(raw)
    patient, cohort, task = patients[row['patient_id']], cohorts[row['cohort_id']], row['task']
    key = digest([patient['id'], cohort['id'], task])
    if (task not in TASKS or row.get('key') != key or Path(path).stem != key
            or row.get('contract') != question_contract or row.get('finish_reason') != 'stop'
            or row['split'] != patient['split'] or row['split'] != cohort['split']
            or row.get('prompt_sha256') != digest(judge_messages(task, patient['text'], cohort['text'], sources=sources))):
        raise ValueError('Response identity, contract or full-text prompt mismatch: ' + path)
    parse_judgment(json.dumps({k: row[k] for k in ('answer', 'evidence_status', 'reason')}))
    return (patient['id'], cohort['id'], task, row['answer']), len(raw)


def read_batch(paths):
    rows, byte_count = [], 0
    for row, size in _READERS.map(read_one, paths):
        byte_count += size
        if row is not None:
            rows.append(row)
    return rows, byte_count


def response_paths(root):
    with os.scandir(root) as entries:
        for entry in entries:
            if entry.name.endswith('.json'):
                yield entry.path
            elif len(entry.name) == 2 and entry.is_dir():
                with os.scandir(entry.path) as children:
                    for child in children:
                        if child.name.endswith('.json'):
                            yield child.path


def batches(paths, size=128):
    iterator = iter(paths)
    while batch := list(islice(iterator, size)):
        yield batch


def export(source_run, output, *, workers=4, reader_threads=8):
    output.mkdir(parents=True, exist_ok=True)
    with (output/'export.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _export(source_run, output, workers=workers, reader_threads=reader_threads)


def _export(source_run, output, *, workers, reader_threads):
    cutoff_ns = time.time_ns()
    patients, cohorts = load_corpus(source_run/'corpus_all')
    pmap, cmap = {p['id']: p for p in patients}, {c['id']: c for c in cohorts}
    teacher = json.loads((source_run/'labels/labeling_manifest.json').read_text())
    question_contract = contract_hash()
    aggregate_contract = score_contract(question_contract)
    if any(teacher.get(k) != v for k, v in dict(contract=question_contract, aggregation_contract=aggregate_contract,
            patients=digest(patients), cohorts=digest(cohorts)).items()):
        raise ValueError('Source teacher/corpus contract mismatch')
    pairs, counts = {}, Counter()
    started = progress = time.monotonic()
    task_indices = {task: i for i, task in enumerate(TASK_ORDER)}
    jobs = batches(response_paths(source_run/'labels/responses'))
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
            initializer=initialize_reader, initargs=(source_run/'corpus_all', question_contract, cutoff_ns, reader_threads)) as pool:
        # Bound submissions so a million filenames never become a million futures.
        while window := list(islice(jobs, workers * 4)):
            for rows, byte_count in pool.map(read_batch, window):
                counts['response_bytes'] += byte_count
                counts['responses'] += len(rows)
                for patient_id, cohort_id, task, answer in rows:
                    key = (pmap[patient_id]['id'], cmap[cohort_id]['id'])
                    answers = pairs.setdefault(key, [None] * len(TASK_ORDER))
                    index = task_indices[task]
                    if answers[index] is not None and answers[index] != answer:
                        raise ValueError('Conflicting duplicate teacher question')
                    answers[index] = answer
            if time.monotonic() - progress >= 10:
                elapsed = time.monotonic() - started
                report = dict(state='packing', responses=counts['responses'], pairs_seen=len(pairs),
                    MiB=round(counts['response_bytes']/2**20, 1), elapsed_seconds=round(elapsed, 1),
                    responses_per_second=round(counts['responses']/elapsed), cutoff_ns=cutoff_ns)
                print(json.dumps(report), flush=True)
                write_json(output/'status.json', report)
                progress = time.monotonic()
    filename = f'scores-{cutoff_ns}.jsonl'
    destination = output/filename
    temporary = destination.with_suffix('.jsonl.tmp')
    hasher = hashlib.sha256()
    split_counts, score_counts = Counter(), Counter()
    with temporary.open('wb') as stream:
        for (patient_id, cohort_id), answers in pairs.items():
            row = score_from_answers(patient_id, cohort_id, pmap[patient_id]['split'], answers,
                                     question_contract, aggregate_contract)
            if row is None:
                counts['incomplete_pairs'] += 1
                continue
            raw = (json.dumps(row, separators=(',', ':')) + '\n').encode()
            stream.write(raw)
            hasher.update(raw)
            counts['completed_pairs'] += 1
            split_counts[pmap[patient_id].get('source_split', pmap[patient_id]['split'])] += 1
            score_counts[str(row['score'])] += 1
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(destination)
    manifest = dict(format='completed-gated-scores-v1', scores_file=filename, scores_sha256=hasher.hexdigest(),
        teacher_manifest=teacher, cutoff_utc=datetime.fromtimestamp(cutoff_ns/1e9, timezone.utc).isoformat(),
        created_at=datetime.now(timezone.utc).isoformat(), counts=dict(counts),
        by_patient_source_split=dict(split_counts), score_distribution=dict(score_counts),
        source_responses=str((source_run/'labels/responses').resolve()),
        validation='Question keys, stopped completions, full-text prompt hashes, corpus and gated-score contracts checked; reasoning retained in source responses')
    write_json(output/'manifest.json', manifest)
    write_json(output/'status.json', dict(state='complete', **manifest))
    print(json.dumps(dict(state='complete', archive=str(output), **dict(counts),
                         by_patient_source_split=dict(split_counts))), flush=True)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', type=Path, default=DEFAULT_RUN)
    p.add_argument('--output', type=Path, help='Default SOURCE_RUN/labels/completed_scores')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--reader-threads', type=int, default=8, help='Concurrent NFS reads per CPU process')
    args = p.parse_args()
    if min(args.workers, args.reader_threads) < 1:
        raise ValueError('workers and reader-threads must be positive')
    export(args.source_run.resolve(), (args.output or args.source_run/'labels/completed_scores').resolve(),
           workers=args.workers, reader_threads=args.reader_threads)


if __name__ == '__main__':
    main()
