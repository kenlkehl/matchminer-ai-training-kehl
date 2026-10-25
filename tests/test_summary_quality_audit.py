import hashlib
import json

import pandas as pd
import pytest

from scripts.check_patient_summaries import audit


@pytest.mark.parametrize('suffix', ['jsonl', 'csv', 'parquet'])
def test_audit_supports_saved_formats_and_preserves_inputs(tmp_path, suffix):
    source = tmp_path / f'patients.{suffix}'
    rows = [dict(patient_id='p1', patient_summary='Cancer type: Lung\nOther relevant history:\nWatch and wait.', source_split='train'),
            dict(patient_id='p2', patient_summary='I should rewrite this.\nFinal Polish:\nCancer type: Gastric', source_split='val')]
    if suffix == 'jsonl':
        source.write_text('\n'.join(map(json.dumps, rows)))
    elif suffix == 'csv':
        pd.DataFrame(rows).to_csv(source, index=False)
    else:
        pd.DataFrame(rows).to_parquet(source, index=False)
    before = source.read_bytes()
    result = audit(source, tmp_path/'audit')
    assert result['counts']['total'] == 2
    assert result['counts']['flagged'] == 1
    assert result['by_source_split']['val']['flagged'] == 1
    assert result['source_sha256'] == hashlib.sha256(before).hexdigest()
    assert source.read_bytes() == before
    flagged = json.loads((tmp_path/'audit/flagged.jsonl').read_text())
    assert flagged['patient_id'] == 'p2'
    assert 'summary_text' not in flagged
    assert 'Gastric' not in json.dumps(flagged)


def test_corpus_columns_and_explicit_text_opt_in(tmp_path):
    source = tmp_path/'patients.jsonl'
    source.write_text(json.dumps(dict(id='p', text='Wait, I should recheck the prompt.', source_split='val')))
    audit(source, tmp_path/'audit', include_text=True)
    flagged = json.loads((tmp_path/'audit/flagged.jsonl').read_text())
    assert flagged['summary_text'].startswith('Wait,')
    with pytest.raises(FileExistsError):
        audit(source, tmp_path/'audit')
