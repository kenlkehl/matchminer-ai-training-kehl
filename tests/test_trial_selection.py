import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parents[1]

def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

parse = load_script('ctgov_parse', '0a_parse_ctgov_json.py')
spaces = load_script('trial_spaces_filter', '0b_create_trial_spaces.py')

REGISTRY_METADATA = {
    'overall_status': 'COMPLETED',
    'status_verified_date': '2025-11',
    'study_first_post_date': '2020-01-01',
    'study_first_post_date_type': 'ACTUAL',
    'study_start_date': '2020-10',
    'study_start_date_type': 'ACTUAL',
    'primary_completion_date': '2023-10-30',
    'primary_completion_date_type': 'ACTUAL',
    'study_completion_date': '2024-10-31',
    'study_completion_date_type': 'ACTUAL',
    'last_update_post_date': '2025-11-25',
    'last_update_post_date_type': 'ACTUAL',
}


def test_parse_retains_structured_design_fields(tmp_path):
    path = tmp_path / 'export.json'
    path.write_text(json.dumps([
        {'protocolSection': {'identificationModule': {'nctId': 'NCT00000001'},
                             'designModule': {'studyType': 'INTERVENTIONAL', 'phases': ['PHASE1', 'PHASE2']}}},
        {'protocolSection': {'identificationModule': {'nctId': 'NCT00000002'}}},
    ]))
    rows = parse.parse_clinical_trials_json(path)
    assert rows.iloc[0].study_type == 'INTERVENTIONAL'
    assert json.loads(rows.iloc[0].phases) == ['PHASE1', 'PHASE2']
    assert rows.iloc[1].study_type == '' and json.loads(rows.iloc[1].phases) == []
    csv = tmp_path / 'trials.csv'
    rows.to_csv(csv, index=False)
    assert spaces.select_trials(pd.read_csv(csv), 'interventional-phase1-3').nct_id.tolist() == ['NCT00000001']


def test_parse_preserves_registry_dates_without_inventing_precision_or_status(tmp_path):
    path = tmp_path / 'export.json'
    protocol = {
        'identificationModule': {'nctId': 'NCT00000001', 'briefTitle': 'Example trial'},
        'descriptionModule': {'briefSummary': 'Example summary'},
        'eligibilityModule': {'eligibilityCriteria': 'Example criteria'},
        'statusModule': {
            'overallStatus': 'ACTIVE_NOT_RECRUITING',
            'statusVerifiedDate': '2025-11',
            'studyFirstPostDateStruct': {'date': '2019-11-05', 'type': 'ACTUAL'},
            'startDateStruct': {'date': '2020-10', 'type': 'ACTUAL'},
            'primaryCompletionDateStruct': {'date': '2023-10-30', 'type': 'ACTUAL'},
            'completionDateStruct': {'date': '2028-06', 'type': 'ESTIMATED'},
            'lastUpdatePostDateStruct': {'date': '2025-11-25', 'type': 'ACTUAL'},
        },
    }
    path.write_text(json.dumps([
        {'protocolSection': protocol},
        {'protocolSection': {'identificationModule': {'nctId': 'NCT00000002'}}},
        {'protocolSection': {'statusModule': None}},
        {'protocolSection': {'statusModule': {
            'studyFirstPostDateStruct': None,
            'completionDateStruct': {'date': None, 'type': None},
        }}},
    ]))
    rows = parse.parse_clinical_trials_json(path)
    assert len(rows) == 4
    expected = dict(REGISTRY_METADATA, study_first_post_date='2019-11-05',
                    overall_status='ACTIVE_NOT_RECRUITING',
                    study_completion_date='2028-06', study_completion_date_type='ESTIMATED')
    assert rows.iloc[0][list(expected)].to_dict() == expected
    assert rows.iloc[0].trial_text == 'Example trial\nExample summary\nExample criteria'
    assert rows.iloc[1:][list(expected)].eq('').all().all()
    output = tmp_path / 'trials.csv'
    rows.to_csv(output, index=False)
    saved = pd.read_csv(output, keep_default_na=False)
    assert saved.iloc[0][list(expected)].to_dict() == expected
    assert saved.iloc[1:][list(expected)].eq('').all().all()


@pytest.mark.parametrize('study_type,phases,keep', [
    ('INTERVENTIONAL', ['PHASE1'], True),
    ('INTERVENTIONAL', ['PHASE2'], True),
    ('INTERVENTIONAL', ['PHASE3'], True),
    ('INTERVENTIONAL', ['PHASE1', 'PHASE2'], True),
    ('INTERVENTIONAL', ['PHASE2', 'PHASE3'], True),
    ('INTERVENTIONAL', ['EARLY_PHASE1'], True),
    ('OBSERVATIONAL', ['EARLY_PHASE1'], False),
    ('INTERVENTIONAL', ['PHASE4'], False),
    ('INTERVENTIONAL', ['PHASE3', 'PHASE4'], False),
    ('INTERVENTIONAL', ['NA'], False),
    ('INTERVENTIONAL', [], False),
    ('OBSERVATIONAL', ['PHASE2'], False),
    ('', ['PHASE2'], False),
])
def test_filter_uses_structured_metadata_only(study_type, phases, keep):
    rows = pd.DataFrame([dict(nct_id='NCT00000001', study_type=study_type,
                            phases=json.dumps(phases), trial_text='Phase 2 interventional cancer study')])
    assert len(spaces.select_trials(rows, 'interventional-phase1-3')) == int(keep)
    assert len(spaces.select_trials(rows, 'all')) == 1


def test_missing_and_malformed_metadata_are_not_guessed():
    with pytest.raises(ValueError, match='Rerun 0a'):
        spaces.select_trials(pd.DataFrame([{'trial_text': 'Phase 2 study'}]), 'interventional-phase1-3')
    for value in ['Phase 1/Phase 2', '"PHASE1"', '[1]']:
        with pytest.raises(ValueError, match='JSON array'):
            spaces.select_trials(pd.DataFrame([dict(study_type='INTERVENTIONAL', phases=value)]), 'interventional-phase1-3')
    rows = pd.DataFrame([dict(study_type='INTERVENTIONAL', phases=float('nan'))])
    assert spaces.select_trials(rows, 'interventional-phase1-3').empty


def test_start_year_uses_first_posted_date_inclusively_without_inventing_dates():
    dates = ['2019-12-31', '2020-01-01', '2020-12-31', '2021-01-01',
             '2022', '2023-05', '', None, 'invalid', '2020-02-30']
    rows = pd.DataFrame({
        'study_first_post_date': dates,
        'study_start_date': ['2025-01-01'] * len(dates),
        'last_update_post_date': ['2026-10-15'] * len(dates),
        'trial_text': ['First posted in 2026'] * len(dates),
    })
    original = rows.copy()
    assert spaces.select_trials_since_year(rows, 2020).index.tolist() == [1, 2, 3, 4, 5]
    assert spaces.select_trials_since_year(rows, 2021).index.tolist() == [3, 4, 5]
    assert spaces.select_trials_since_year(rows, 2024).empty
    pd.testing.assert_frame_equal(rows, original)
    with pytest.raises(ValueError, match='study_first_post_date.*Rerun 0a'):
        spaces.select_trials_since_year(rows.drop(columns='study_first_post_date'), 2020)


def test_resume_rejects_different_selection_and_untracked_old_shards(tmp_path):
    rows = pd.DataFrame([dict(nct_id='NCT00000001', trial_text='Example')])
    spaces.check_trial_selection_manifest(rows, tmp_path, 'all', 42)
    spaces.check_trial_selection_manifest(rows, tmp_path, 'all', 42)
    with pytest.raises(ValueError, match='new --work-dir'):
        spaces.check_trial_selection_manifest(rows, tmp_path, 'interventional-phase1-3', 42)
    with pytest.raises(ValueError, match='new --work-dir'):
        spaces.check_trial_selection_manifest(rows, tmp_path, 'all', 42, start_year=2021)
    changed = rows.assign(trial_text='Changed example')
    with pytest.raises(ValueError, match='new --work-dir'):
        spaces.check_trial_selection_manifest(changed, tmp_path, 'all', 42)
    (tmp_path/'trial_selection.json').unlink()
    (tmp_path/'remote_output_shard_000000.parquet').touch()
    with pytest.raises(ValueError, match='no trial-selection manifest'):
        spaces.check_trial_selection_manifest(rows, tmp_path, 'all', 42)


@pytest.mark.parametrize("state_label", ["Cancer burden allowed", "Disease state allowed"])
@pytest.mark.parametrize('with_registry_metadata', [False, True])
@pytest.mark.parametrize('start_year', [None, 2019])
@pytest.mark.parametrize('mode', ['local', 'remote'])
def test_main_filters_before_dispatch_and_keeps_metadata(tmp_path, monkeypatch, state_label, with_registry_metadata, start_year, mode):
    import sys
    metadata = REGISTRY_METADATA if with_registry_metadata else {'study_first_post_date': '2020-01-01'}
    rows = pd.DataFrame([
        dict(nct_id='NCT00000001', study_type='INTERVENTIONAL', phases='["PHASE1", "PHASE2"]', trial_text='Example', **metadata),
        dict(nct_id='NCT00000002', study_type='INTERVENTIONAL', phases='["PHASE4"]', trial_text='Example', **metadata),
        dict(nct_id='NCT00000003', study_type='OBSERVATIONAL', phases='["PHASE2"]', trial_text='Example', **metadata),
        dict(nct_id='NCT00000004', study_type='INTERVENTIONAL', phases='["PHASE1", "PHASE2"]', trial_text='Older',
             **dict(metadata, study_first_post_date='2019-12-31')),
        dict(nct_id='NCT00000005', study_type='INTERVENTIONAL', phases='["PHASE2"]', trial_text='Missing date'),
    ])
    expected_ids = ['NCT00000001', 'NCT00000004'] if start_year == 2019 else ['NCT00000001']
    source = tmp_path / 'input.csv'
    rows.to_csv(source, index=False)
    output_trials, output_spaces = tmp_path/'trials.csv', tmp_path/'spaces.csv'
    def remote(args, trials, base_cols, reasoning_parser, *, screening=False):
        assert trials.nct_id.tolist() == expected_ids
        assert trials.iloc[0][list(metadata)].to_dict() == metadata
        out = trials.copy()
        if screening:
            from trial_treatment_screening import screening_record
            record = screening_record('', json.dumps({'cancer_treatment_trial': 'yes', 'multiple_cohorts': 'no'}), 'stop')
            for name, value in record.items():
                out[name] = value
            path = Path(args.work_dir)/'remote_output_shard_000000.parquet'
            out.to_parquet(path, index=False)
            return [str(path)]
        final = ('1. Age range allowed: 18+. Sex allowed: Any. Cancer type allowed: lung cancer. '
                 f'Histology allowed: NA. {state_label}: metastatic. '
                 'Prior treatment required: NA. Prior treatment excluded: NA. '
                 'Biomarkers required: NA. Biomarkers excluded: NA.')
        for name in ['space_text', 'space_output_no_reasoning', 'space_reasoning_and_output']:
            out[name] = final
        out['trial_boilerplate_text'] = 'Active infection'
        path = Path(args.work_dir)/'remote_output_shard_000000.parquet'
        out.to_parquet(path, index=False)
        return [str(path)]
    monkeypatch.setattr(spaces, f'{mode}_main', remote)
    argv = ['0b_create_trial_spaces.py', '--input', str(source),
        '--work-dir', str(tmp_path/'work'), '--output-trials', str(output_trials),
        '--output-spaces', str(output_spaces)]
    argv.extend(['--server_urls', 'http://unused/v1'] if mode == 'remote' else ['--gpus', '0'])
    if start_year is not None:
        argv.extend(['--start-year', str(start_year)])
    monkeypatch.setattr(sys, 'argv', argv)
    spaces.main()
    for path in [output_trials, output_spaces, tmp_path/'trial_treatment_screening.csv']:
        result = pd.read_csv(path)
        assert result.nct_id.tolist() == expected_ids
        assert json.loads(result.phases.iloc[0]) == ['PHASE1', 'PHASE2']
        assert result.study_type.iloc[0] == 'INTERVENTIONAL'
        assert result.iloc[0][list(metadata)].to_dict() == metadata
        if not with_registry_metadata:
            assert set(REGISTRY_METADATA).intersection(result.columns) == {'study_first_post_date'}
    # Exercise the actual main-path export for local/remote and metadata variants.
    lineitems = pd.read_csv(output_spaces)
    assert not {'space_reasoning_and_output', 'space_output_no_reasoning', 'space_text'} & set(lineitems)
    assert lineitems.reasoning_trials_path.eq(str(output_trials.resolve())).all()
    assert pd.read_csv(output_trials).space_reasoning_and_output.notna().all()
    for folder in [tmp_path/'work', tmp_path/'work'/'space_extraction']:
        assert json.loads((folder/'trial_selection.json').read_text())['start_year'] == (start_year or 2020)


@pytest.mark.parametrize('trial_filter', ['all', 'interventional-phase1-3'])
def test_start_year_excluding_every_trial_stops_before_inference(tmp_path, monkeypatch, trial_filter):
    import sys
    source = tmp_path/'input.csv'
    pd.DataFrame([dict(nct_id='NCT00000001', study_type='INTERVENTIONAL', phases='["PHASE2"]',
                       study_first_post_date='2019-12-31')]).to_csv(source, index=False)
    monkeypatch.setattr(spaces, 'screen_trials', lambda *a: pytest.fail('Screening must not start'))
    monkeypatch.setattr(sys, 'argv', ['0b', '--input', str(source), '--work-dir', str(tmp_path/'work'),
                                    '--trial-filter', trial_filter])
    with pytest.raises(ValueError, match='No trials match.*no inference was started'):
        spaces.main()


@pytest.mark.parametrize('year', ['0', '10000', 'not-a-year'])
def test_cli_rejects_invalid_start_year_before_reading_input(monkeypatch, year):
    import sys
    monkeypatch.setattr(sys, 'argv', ['0b', '--input', 'unused.csv', '--start-year', year])
    with pytest.raises(SystemExit) as exc:
        spaces.main()
    assert exc.value.code == 2
