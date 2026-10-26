import json

import pytest

from colbert_experiments.data import digest, load_corpus, read_jsonl, write_json, write_jsonl
from colbert_experiments.export_completed_scores import score_from_answers
from colbert_experiments.fixtures import create_fixture
from colbert_experiments.label_reuse import import_completed_sources
from colbert_experiments.score_journal import ScoreJournal


def setup(tmp_path):
    corpus = tmp_path / 'corpus'
    create_fixture(corpus)
    patients, cohorts = load_corpus(corpus)
    pmap, cmap = {r['id']: r for r in patients}, {r['id']: r for r in cohorts}
    pairs = read_jsonl(corpus / 'train_pairs.jsonl')[:2]
    source, target = tmp_path / 'source', tmp_path / 'target'
    signature = dict(contract='questions', aggregation_contract='scores', patients=digest(patients),
                     cohorts=digest(cohorts), pairs=digest(pairs), sampling={'model': 'gemma'})
    original = {**signature, 'pairs': digest(pairs[:1])}
    write_json(source / 'labeling_manifest.json', original)
    (source / 'labeling.lock').touch()
    write_jsonl(source / 'candidates.jsonl', pairs[:1])
    pair = pairs[0]
    row = score_from_answers(pair['patient_id'], pair['cohort_id'], pair['split'],
                             ['no', None, None, None, None], 'questions', 'scores')
    ScoreJournal(source, original).append([row])
    key = next(iter(row['question_keys'].values()))
    trace = dict(key=key, reasoning='complete saved reasoning', answer='no')
    write_json(source / 'responses' / key[:2] / (key + '.json'), trace)
    write_json(target / 'reuse_sources.json', dict(sources=[
        dict(labels=str(source), candidates=str(source / 'candidates.jsonl'))]))
    return source, target, signature, pmap, cmap, pairs, key, trace


def test_reuses_only_candidate_subset_with_exact_contract_and_preserves_reasoning(tmp_path):
    source, target, signature, patients, cohorts, pairs, key, trace = setup(tmp_path)
    assert import_completed_sources(target, signature, patients, cohorts, pairs) == 1
    assert len(ScoreJournal(target, signature).load(patients, cohorts)) == 1
    assert json.loads((target / 'responses' / key[:2] / (key + '.json')).read_text()) == trace
    assert import_completed_sources(target, signature, patients, cohorts, pairs) == 0
    assert len((target / 'score_journal/scores.jsonl').read_text().splitlines()) == 1


@pytest.mark.parametrize('field', ['patients', 'cohorts', 'contract', 'sampling'])
def test_rejects_changed_clinical_text_prompt_or_teacher(tmp_path, field):
    source, target, signature, patients, cohorts, pairs, _, _ = setup(tmp_path)
    with pytest.raises(ValueError, match='identical corpus'):
        import_completed_sources(target, {**signature, field: 'changed'}, patients, cohorts, pairs)
    assert not (target / 'score_journal').exists()


def test_rejects_seed_pairs_missing_from_final_retrieval(tmp_path):
    source, target, signature, patients, cohorts, pairs, _, _ = setup(tmp_path)
    with pytest.raises(ValueError, match='absent from the candidate set'):
        import_completed_sources(target, {**signature, 'pairs': digest(pairs[1:])}, patients, cohorts, pairs[1:])


def test_missing_reasoning_never_imports_a_completed_score(tmp_path):
    source, target, signature, patients, cohorts, pairs, key, _ = setup(tmp_path)
    (source / 'responses' / key[:2] / (key + '.json')).unlink()
    with pytest.raises(FileNotFoundError, match='reasoning trace'):
        import_completed_sources(target, signature, patients, cohorts, pairs)
    assert not (target / 'score_journal/scores.jsonl').exists()
