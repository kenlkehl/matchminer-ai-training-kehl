"""Plan label reconciliation after source-summary and retrieval repairs.

Callers must rerun all cohort queries first. This module compares full candidate
sets, so newly promoted *other* patients cannot be missed by an affected-ID filter.
Publication and response archival require the production label writer to stop.
"""


def pair_key(row):
    return row['patient_id'], row['cohort_id']


def directional_rankings(rows, direction):
    groups = {}
    query_field = 'cohort_id' if direction == 'cohort_to_patient' else 'patient_id'
    candidate_field = 'patient_id' if direction == 'cohort_to_patient' else 'cohort_id'
    for row in rows:
        if row['direction'] == direction and row['source'] == 'colbert':
            groups.setdefault(row[query_field], []).append((row['rank'], row[candidate_field]))
    return {key: [candidate for _, candidate in sorted(values)] for key, values in groups.items()}


def promoted_pairs(old_rankings, new_rankings):
    if old_rankings.keys() != new_rankings.keys():
        raise ValueError('Repair must retrieve all original cohort queries')
    return {(patient, cohort) for cohort, patients in new_rankings.items()
            for patient in set(patients) - set(old_rankings[cohort])}


def repair_label_reasons(pair, *, old_pair_keys, repaired_patient_ids, promoted):
    """Return reasons for checking/labeling a current pair during the repair.

    Newly promoted pairs already in another retrieval source can reuse a valid
    existing label, but still need checking: that label may not have finished yet.
    Every current pair involving a repaired patient requires a fresh judgment.
    """
    key = pair_key(pair)
    reasons = []
    if pair['patient_id'] in repaired_patient_ids:
        reasons.append('repaired_patient_text')
    if key not in old_pair_keys:
        reasons.append('new_candidate_pair')
    if key in promoted:
        reasons.append('promoted_into_cohort_top_k')
    return reasons
