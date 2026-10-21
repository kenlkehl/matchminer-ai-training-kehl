"""Re-extract a subset without changing its saved screening/split identity."""
import json
import sys

import pandas as pd
import pytest

from tests.test_trial_selection import load_script
from trial_treatment_screening import screening_record


spaces = load_script('preserve_trial_splits', '0b_create_trial_spaces.py')


@pytest.mark.parametrize('problem', [None, 'missing', 'invalid', 'empty', 'conflicting'])
def test_subset_reuses_screening_with_original_splits(tmp_path, monkeypatch, problem):
    rows = pd.DataFrame([
        dict(nct_id='NCT00000002', trial_text='Synthetic trial B', split='test'),
        dict(nct_id='NCT00000001', trial_text='Synthetic trial A', split='val'),
    ]).assign(study_type='INTERVENTIONAL', phases='["PHASE2"]', study_first_post_date='2021-01-01')
    audit = rows.copy()
    record = screening_record('', json.dumps({
        'is_cancer_treatment_trial': True, 'rationale': 'Treatment trial',
    }), 'stop')
    for name, value in record.items():
        audit[name] = value
    audit['treatment_screen_model'] = 'original-screen-model'
    screen_path = tmp_path / 'screen.csv'
    audit.to_csv(screen_path, index=False)
    if problem == 'missing':
        rows = rows.drop(columns='split')
    elif problem in {'invalid', 'empty'}:
        rows.loc[0, 'split'] = 'validation' if problem == 'invalid' else None
    elif problem == 'conflicting':
        rows = pd.concat([rows, rows.iloc[[0]].assign(split='train')], ignore_index=True)
    source = tmp_path / 'input.csv'
    rows.to_csv(source, index=False)
    dispatched = []

    def remote(args, trials, base_cols, reasoning_parser):
        dispatched.append(trials.copy())
        assert trials.set_index('nct_id')['split'].to_dict() == {
            'NCT00000001': 'val', 'NCT00000002': 'test',
        }
        return []

    monkeypatch.setattr(spaces, 'remote_main', remote)
    import trial_space_export
    monkeypatch.setattr(trial_space_export, 'write_trial_exports',
                        lambda *args: {'trials': 2, 'spaces': 2})
    monkeypatch.setattr(sys, 'argv', [
        '0b', '--input', str(source), '--screening-input', str(screen_path),
        '--work-dir', str(tmp_path / 'work'), '--preserve-input-splits',
        '--server_urls', 'http://unused/v1',
    ])
    if problem:
        with pytest.raises(ValueError, match='--preserve-input-splits'):
            spaces.main()
        assert not dispatched
    else:
        spaces.main()
        assert len(dispatched) == 1
        manifest = json.loads((tmp_path / 'work' / 'screening_input_manifest.json').read_text())
        assert manifest['screening_models'] == ['original-screen-model']
        assert manifest['rows'] == 2
