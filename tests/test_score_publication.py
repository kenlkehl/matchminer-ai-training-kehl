import json
import os

import pytest

from colbert_experiments.contracts import contract_hash
from colbert_experiments.data import digest, load_corpus, read_jsonl, write_json, write_jsonl
from colbert_experiments.export_completed_scores import score_from_answers
from colbert_experiments.finalize_labels import finalize
from colbert_experiments.fixtures import create_fixture
from colbert_experiments.regression import score_contract
from colbert_experiments.score_journal import ScoreJournal
from colbert_experiments.score_publication import ready_scores


def source(tmp_path, *, incomplete=False):
    corpus, output = tmp_path / 'corpus_all', tmp_path / 'labels'
    create_fixture(corpus)
    pairs = read_jsonl(corpus / 'train_pairs.jsonl')[:2]
    candidates = tmp_path / 'mining_all/candidates.jsonl'
    write_jsonl(candidates, pairs)
    patients, cohorts = load_corpus(corpus)
    signature = dict(contract=contract_hash(), aggregation_contract=score_contract(),
                     patients=digest(patients), cohorts=digest(cohorts), pairs=digest(pairs))
    write_json(output / 'labeling_manifest.json', signature)
    write_json(output / 'status.json', dict(state='exporting_labels', total_pairs=2, scored_pairs=2, failures={}))
    write_json(tmp_path / 'status.json', dict(phase='failed', error='teacher supervisor failed after inference'))
    journal = ScoreJournal(output, signature)
    scores = [score_from_answers(pair['patient_id'], pair['cohort_id'], pair['split'], ['no'] + [None] * 4,
              signature['contract'], signature['aggregation_contract']) for pair in pairs]
    journal.append(scores[:1] if incomplete else scores)
    return corpus, candidates, output


def test_finalization_publishes_exact_complete_scores_without_reading_traces(tmp_path):
    corpus, candidates, output = source(tmp_path)
    receipt = finalize(corpus, candidates, output, scores_only=True)
    assert receipt['pairs'] == receipt['questions'] == 2
    assert receipt['candidate_coverage_verified'] is True
    assert len(read_jsonl(output / 'scores.jsonl')) == 2
    assert not (output / 'labels.jsonl').exists()
    assert not (output / 'responses').exists()
    assert ready_scores(output, corpus=corpus, candidates=candidates, expected_pairs=2) == receipt


def test_missing_candidate_labels_cannot_publish_a_complete_receipt(tmp_path):
    corpus, candidates, output = source(tmp_path, incomplete=True)
    with pytest.raises(ValueError, match='Journal is incomplete'):
        finalize(corpus, candidates, output, scores_only=True)
    assert not (output / 'scores_ready.json').exists()


def test_finalization_rejects_changed_clinical_inputs(tmp_path):
    corpus, candidates, output = source(tmp_path)
    patients = read_jsonl(corpus / 'patients.jsonl')
    patients[0]['text'] += ' changed history'
    write_jsonl(corpus / 'patients.jsonl', patients)
    with pytest.raises(ValueError, match='original teacher manifest'):
        finalize(corpus, candidates, output, scores_only=True)
    assert not (output / 'scores_ready.json').exists()


@pytest.mark.parametrize('changed', ['scores', 'patients', 'teacher'])
def test_publication_rejects_files_changed_after_validation(tmp_path, changed):
    corpus, candidates, output = source(tmp_path)
    finalize(corpus, candidates, output, scores_only=True)
    path = {'scores': output / 'scores.jsonl', 'patients': corpus / 'patients.jsonl',
            'teacher': output / 'labeling_manifest.json'}[changed]
    if changed == 'teacher':
        signature = json.loads(path.read_text())
        write_json(path, {**signature, 'contract': 'other'})
    else:
        with path.open('a') as stream:
            stream.write('\n')
    with pytest.raises(ValueError):
        ready_scores(output, corpus=corpus, candidates=candidates, expected_pairs=2)


def test_gb300_can_use_validated_scores_after_export_workflow_failure(tmp_path, monkeypatch):
    from colbert_experiments import train_full_gb300 as launcher
    corpus, candidates, output = source(tmp_path)
    finalize(corpus, candidates, output, scores_only=True)
    monkeypatch.setattr(launcher, 'EXPECTED_PAIRS', 2)
    ready, status = launcher.ready(tmp_path)
    assert ready and status['scored_pairs'] == 2
    assert not (output / 'labels.jsonl').exists()


def test_other_mount_timestamp_precision_does_not_reject_identical_content(tmp_path, monkeypatch):
    from colbert_experiments import score_publication as publication
    corpus, candidates, output = source(tmp_path)
    receipt = finalize(corpus, candidates, output, scores_only=True)
    local_stat = publication.fingerprint
    def remote_stat(path):
        value = local_stat(path)
        return {**value, 'mtime_ns': value['mtime_ns'] // 1_000_000_000 * 1_000_000_000}
    monkeypatch.setattr(publication, 'fingerprint', remote_stat)
    assert ready_scores(output, corpus=corpus, candidates=candidates, expected_pairs=2) == receipt


def test_same_size_and_timestamp_do_not_hide_changed_bytes(tmp_path):
    corpus, candidates, output = source(tmp_path)
    finalize(corpus, candidates, output, scores_only=True)
    path = output / 'scores.jsonl'
    stat = path.stat()
    raw = bytearray(path.read_bytes())
    raw[2] = ord('X')
    path.write_bytes(raw)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ValueError, match='content differs.*scores.jsonl'):
        ready_scores(output, corpus=corpus, candidates=candidates, expected_pairs=2)
