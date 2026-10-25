from colbert_experiments.summary_repair import promoted_pairs, repair_label_reasons
import pytest


def test_promoted_other_patient_requires_label_even_if_already_another_source():
    promoted = promoted_pairs({'c': ['faulty', 'steady']}, {'c': ['steady', 'other']})
    assert promoted == {('other', 'c')}
    reasons = repair_label_reasons(dict(patient_id='other', cohort_id='c'),
        old_pair_keys={('other', 'c')}, repaired_patient_ids={'faulty'}, promoted=promoted)
    assert reasons == ['promoted_into_cohort_top_k']


def test_changed_text_is_relabelled_and_unchanged_pairs_reuse_labels():
    kwargs = dict(old_pair_keys={('faulty', 'c'), ('steady', 'c')},
                  repaired_patient_ids={'faulty'}, promoted=set())
    assert repair_label_reasons(dict(patient_id='faulty', cohort_id='c'), **kwargs) == ['repaired_patient_text']
    assert repair_label_reasons(dict(patient_id='steady', cohort_id='c'), **kwargs) == []
    assert repair_label_reasons(dict(patient_id='new', cohort_id='c'), **kwargs) == ['new_candidate_pair']


def test_partial_cohort_retrieval_is_rejected():
    with pytest.raises(ValueError, match='all original cohort'):
        promoted_pairs({'a': [], 'b': []}, {'a': []})
