import asyncio
import json

import pytest

from colbert_experiments.contracts import contract_hash
from colbert_experiments.data import digest, write_json
from colbert_experiments.export_completed_scores import score_from_answers
from colbert_experiments.regression import score_contract
from colbert_experiments.score_journal import ScoreJournal, import_archive


def fixture():
    signature = dict(contract=contract_hash(), aggregation_contract=score_contract(),
                     patients='full-patient-hash', cohorts='full-cohort-hash', pairs='candidate-hash')
    patients = {'p': dict(id='p', split='train')}
    cohorts = {'c': dict(id='c', split='train')}
    score = score_from_answers('p', 'c', 'train', ['yes'] * 5,
                               signature['contract'], signature['aggregation_contract'])
    return signature, patients, cohorts, score


def test_compact_resume_never_needs_response_files_and_rejects_changed_identity(tmp_path):
    signature, patients, cohorts, score = fixture()
    async def run():
        async with ScoreJournal(tmp_path, signature) as journal:
            await journal.commit(score)
            assert journal.load(patients, cohorts) == {digest(['p', 'c']): score}
    asyncio.run(run())
    assert ScoreJournal(tmp_path, signature).load(patients, cohorts)
    with pytest.raises(ValueError, match='identity changed'):
        ScoreJournal(tmp_path, {**signature, 'patients': 'changed-clinical-text'})


def test_recover_torn_tail_without_accepting_corrupt_complete_rows(tmp_path):
    signature, patients, cohorts, score = fixture()
    journal = ScoreJournal(tmp_path, signature)
    journal.append([score])
    with journal.path.open('ab') as stream:
        stream.write(b'{"patient_id":')
    assert len(journal.load(patients, cohorts)) == 1
    assert list(journal.root.glob('torn-tail-*.bin'))
    assert journal.path.read_bytes().endswith(b'\n')
    with journal.path.open('ab') as stream:
        stream.write(b'broken complete row\n')
    with pytest.raises(json.JSONDecodeError):
        journal.load(patients, cohorts)


def test_commit_propagates_disk_failure_without_marking_pair_complete(tmp_path, monkeypatch):
    signature, _, _, score = fixture()
    async def run():
        journal = ScoreJournal(tmp_path, signature)
        def fail(rows):
            raise OSError('disk full')
        monkeypatch.setattr(journal, 'append', fail)
        with pytest.raises(OSError, match='disk full'):
            async with journal:
                await journal.commit(score)
    asyncio.run(run())


def test_legacy_import_requires_matching_validated_archive(tmp_path):
    import hashlib
    signature, patients, cohorts, score = fixture()
    output, archive = tmp_path/'labels', tmp_path/'archive'
    (output/'responses'/'ab').mkdir(parents=True)
    write_json(output/'labeling_manifest.json', signature)
    with pytest.raises(ValueError, match='one-time compact score import'):
        ScoreJournal(output, signature)
    archive.mkdir()
    raw = (json.dumps(score) + '\n').encode()
    (archive/'scores.jsonl').write_bytes(raw)
    write_json(archive/'manifest.json', dict(format='completed-gated-scores-v1',
        teacher_manifest=signature, scores_file='scores.jsonl', scores_sha256=hashlib.sha256(raw).hexdigest(),
        counts=dict(completed_pairs=1)))
    assert import_archive(archive, output) == 1
    assert len(ScoreJournal(output, signature).load(patients, cohorts)) == 1
    with pytest.raises(ValueError, match='already exists'):
        import_archive(archive, output)
