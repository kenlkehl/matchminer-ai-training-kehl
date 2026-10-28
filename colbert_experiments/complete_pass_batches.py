"""Deterministic grouped epochs that visit every directional labeled pair."""
from collections import Counter, defaultdict
import heapq
import math
import random

from .regression import MAX_SCORE


POLICY = 'complete_query_pair_epochs_v1'


class CompletePassBatchSampler:
    def __init__(self, groups, *, documents_per_query, query_groups_per_update, seed):
        if not groups or min(documents_per_query, query_groups_per_update) < 1:
            raise ValueError('Complete epochs require nonempty groups and positive batch sizes')
        self.groups = groups
        self.documents_per_query = documents_per_query
        self.query_groups_per_update = query_groups_per_update
        self.seed = seed
        self.microbatches = 0
        self.pairs = 0
        for group in groups:
            if not group['labels'] or len(group['labels']) != len(group['documents']):
                raise ValueError('Every query needs aligned, nonempty labels and documents')
            if any(type(score) is not int or not 0 <= score <= MAX_SCORE for score in group['labels']):
                raise ValueError(f'Scores must be integers in 0..{MAX_SCORE}')
            self.microbatches += math.ceil(len(group['documents']) / documents_per_query)
            self.pairs += len(group['documents'])
        self.updates_per_epoch = math.ceil(self.microbatches / query_groups_per_update)
        self.cached_epoch, self.plan = None, None

    def _epoch(self, epoch):
        rng = random.Random(f'{self.seed}:{epoch}:{POLICY}')
        buckets = defaultdict(list)
        for group_index, group in enumerate(self.groups):
            # Mix available scores within a query while consuming every document
            # exactly once. Constant-score queries remain part of the epoch.
            by_score = defaultdict(list)
            for index, score in enumerate(group['labels']):
                by_score[score].append(index)
            for indices in by_score.values():
                rng.shuffle(indices)
            ordered = []
            while by_score:
                levels = list(by_score)
                rng.shuffle(levels)
                for level in levels:
                    ordered.append(by_score[level].pop())
                    if not by_score[level]:
                        del by_score[level]
            for start in range(0, len(ordered), self.documents_per_query):
                indices = ordered[start:start + self.documents_per_query]
                # Place score-diverse chunks throughout the epoch, including
                # its tail, instead of exhausting the rare-score chunks first.
                stratum = group['direction'], max(group['labels'][i] for i in indices)
                buckets[stratum].append((group_index, indices))
        heap = []
        for stratum, chunks in sorted(buckets.items()):
            rng.shuffle(chunks)
            heapq.heappush(heap, ((.5 / len(chunks)), stratum, 0))
        plan = []
        while heap:
            _, stratum, index = heapq.heappop(heap)
            chunks = buckets[stratum]
            plan.append(chunks[index])
            if index + 1 < len(chunks):
                heapq.heappush(heap, ((index + 1.5) / len(chunks), stratum, index + 1))
        return plan

    def batch(self, step):
        epoch, update = divmod(step, self.updates_per_epoch)
        if epoch != self.cached_epoch:
            self.plan = self._epoch(epoch)
            self.cached_epoch = epoch
        start = update * self.query_groups_per_update
        return [(self.groups[group_index], indices)
                for group_index, indices in self.plan[start:start + self.query_groups_per_update]]

    def report(self):
        return dict(policy=POLICY, directional_pairs_per_epoch=self.pairs,
            query_chunks_per_epoch=self.microbatches, updates_per_epoch=self.updates_per_epoch,
            score_counts=dict(sorted(Counter(str(score) for g in self.groups for score in g['labels']).items())),
            coverage='Every directional pair exactly once per complete epoch; no dropped tail or replacement sampling',
            ordering='Shuffle within queries and direction/score strata; distribute strata throughout the epoch')
