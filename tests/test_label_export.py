import json

import pytest

from colbert_experiments.data import write_json
from colbert_experiments.label_export import export_labels


def test_parallel_export_preserves_order_and_full_traces_in_both_layouts(tmp_path):
    root = tmp_path / 'responses'
    rows = [dict(key=key, reasoning='Wait\nα', messages=[dict(content='full clinical text')],
                 raw_answer='```json\n{"answer":"yes"}\n```', arbitrary=dict(nested=[1, None]))
            for key in ('cc123', 'aa123', 'bb123')]
    for i, row in enumerate(rows):
        path = root / (row['key'] + '.json') if i == 1 else root / row['key'][:2] / (row['key'] + '.json')
        write_json(path, row)
    progress = []
    target = tmp_path / 'labels.jsonl'
    export_labels(root, target, (row['key'] for row in rows), total=3,
                  workers=2, reader_threads=2, batch_size=1, progress=progress.append)
    assert [json.loads(line) for line in target.read_text().splitlines()] == rows
    assert progress[-1]['exported_questions'] == progress[-1]['total_questions'] == 3
    assert not target.with_suffix('.jsonl.tmp').exists()


@pytest.mark.parametrize('problem', ['missing', 'mismatch', 'count'])
def test_export_error_never_replaces_previous_complete_archive(tmp_path, problem):
    target = tmp_path / 'labels.jsonl'
    target.write_text('previous complete export\n')
    root = tmp_path / 'responses'
    if problem != 'missing':
        write_json(root / 'aa' / 'aa123.json', dict(key='wrong' if problem == 'mismatch' else 'aa123'))
    with pytest.raises((FileNotFoundError, ValueError)):
        export_labels(root, target, ['aa123'], total=2 if problem == 'count' else 1,
                      workers=1, reader_threads=1)
    assert target.read_text() == 'previous complete export\n'
