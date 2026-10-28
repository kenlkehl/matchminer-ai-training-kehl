"""Production synthetic data form one retrieval and fitting pool.

Historical train/val/test assignments are provenance, never holdouts. Real-data
evaluation and explicitly experimental modules are separate from this policy.
"""

POLICY = 'all_synthetic_training'
SOURCE_SPLITS = ('train', 'val', 'test')


def require_all_sources(splits):
    normalized = {'val' if value == 'validation' else value for value in splits}
    if normalized != set(SOURCE_SPLITS):
        raise ValueError('Production uses all synthetic data; source-split holdouts/subsets are disabled')


def pooled_records(rows):
    return [dict(row, source_split=row.get('source_split', row.get('split', 'train')), split='train')
            for row in rows]


def pooled_frame(frame):
    frame = frame.copy()
    if 'source_split' not in frame:
        frame['source_split'] = frame['split'] if 'split' in frame else 'train'
    frame['split'] = 'train'
    return frame


def require_full_pool(patients, cohorts):
    if any(row.get('split') != 'train' for row in [*patients, *cohorts]):
        raise ValueError('Production retrieval/labeling/fitting requires one full synthetic pool. '
                         'Reprepare with --merge-source-splits and rerun both retrieval directions; '
                         'historical assignments belong only in source_split.')


def require_full_corpus(path):
    from colbert_experiments.data import load_corpus
    patients, cohorts = load_corpus(path)
    require_full_pool(patients, cohorts)
    return patients, cohorts
