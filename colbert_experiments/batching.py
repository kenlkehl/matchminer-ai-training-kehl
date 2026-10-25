"""Score-diverse regression update batches, encoded one query group at a time."""
from collections import defaultdict
import random


POLICY = 'direction_score_stratified_v1'


class RegressionBatchSampler:
    """Balance available direction/score anchors without inventing any labels.

    Each optimizer update visits every available stratum when its query-group
    budget permits. Candidates stay grouped by query. Rare scores are deliberately
    oversampled; this is a training distribution, not an eligibility prevalence.
    """

    def __init__(self, groups):
        self.groups = groups
        self.buckets = defaultdict(list)
        for index, group in enumerate(groups):
            if not group['labels'] or len(group['labels']) != len(group['documents']):
                raise ValueError('Regression groups need aligned, nonempty labels')
            for score in sorted(set(group['labels'])):
                if type(score) is not int or not 0 <= score <= 5:
                    raise ValueError('Regression batch scores must be integers in 0..5')
                self.buckets[group['direction'], score].append(index)
        self.strata = sorted(self.buckets)
        if not self.strata:
            raise ValueError('No regression training strata')

    def batch(self, *, query_groups, documents_per_query, seed, step):
        """Return a reproducible global update; DDP ranks take interleaved slices."""
        if query_groups < 1 or documents_per_query < 1:
            raise ValueError('Regression batch sizes must be positive')
        rng = random.Random(f'{seed}:{step}:{POLICY}')
        schedule = []
        while len(schedule) < query_groups:
            cycle = list(self.strata)
            rng.shuffle(cycle)
            schedule.extend(cycle)
        result, used_groups = [], set()
        for stratum in schedule[:query_groups]:
            choices = self.buckets[stratum]
            chosen = rng.choice(choices)
            if chosen in used_groups:
                remaining = [i for i in choices if i not in used_groups]
                if remaining:
                    chosen = rng.choice(remaining)
            used_groups.add(chosen)
            group = self.groups[chosen]
            by_score = defaultdict(list)
            for i, score in enumerate(group['labels']):
                by_score[score].append(i)
            scores = sorted(by_score)
            remainder = list(scores)
            rng.shuffle(remainder)
            # Keep the selected anchor and the within-query range even when an
            # operator requests fewer documents than available score levels.
            priority = list(dict.fromkeys([stratum[1], min(scores), max(scores), *remainder]))
            count = min(documents_per_query, len(group['documents']))
            indices = [rng.choice(by_score[score]) for score in priority[:count]]
            if len(indices) < count:
                selected = set(indices)
                indices += rng.sample([i for i in range(len(group['documents'])) if i not in selected],
                                      count - len(indices))
            rng.shuffle(indices)
            result.append((group, indices))
        return result

    def report(self):
        return dict(policy=POLICY, strata={f'{direction}/{score}': len(indices)
                    for (direction, score), indices in self.buckets.items()},
                    weighting='equal available direction/score anchors; score-diverse documents within query')
