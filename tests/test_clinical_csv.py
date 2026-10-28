"""Clinical CSV text must survive ordinary production and standalone readers."""
import asyncio
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from colbert_experiments.data import load_corpus, read_table
from generate_synthetic_data import Generator, load_stage
from make_top_matches import load_retrieval_tables
from prepare_synthetic_training import load_spaces
from tests.test_synthetic_generation import Tokenizer, args_for
from trial_cohort_format import REPRESENTATION, read_clinical_csv
from trialspace_training import prepare_corpus


def inputs(tmp_path, literal):
    rows = [dict(nct_id='NCT00000001', trial_id='NCT00000001',
                 cohort_trial_id=f'NCT00000001-cohort-{i}', cohort_number=i,
                 cohort_count=2, cancer_treatment_trial='yes', multiple_cohorts='yes',
                 cohort_name=literal if i == 0 else 'Second cohort',
                 cohort_specific_eligibility_criteria=literal if i == 0 else 'Previously treated.',
                 common_eligibility_criteria='Age >= 18.' if i == 0 else literal,
                 cohort_eligibility_criteria='Stale combined text', eligibility_criteria='Original.',
                 split='train', trial_representation=REPRESENTATION) for i in range(2)]
    rows.append(dict(nct_id='NCT00000002', trial_id='NCT00000002',
                     cohort_trial_id='NCT00000002-cohort-0', cohort_number=0, cohort_count=1,
                     cancer_treatment_trial='yes', multiple_cohorts='no',
                     cohort_eligibility_criteria='Stale single text', eligibility_criteria=literal,
                     split='train', trial_representation=REPRESENTATION))
    cohorts, patients = tmp_path / 'input.csv', tmp_path / 'patients.csv'
    for i, row in enumerate(rows):
        row['cohort_index'] = i
    pd.DataFrame(rows).to_csv(cohorts, index=False)
    pd.DataFrame([dict(pseudo_mrn='synthetic-patient', patient_summary='Cancer history.',
                      patient_boilerplate_text=literal, nct_id='NCT00000001', split='train')]).to_csv(patients, index=False)
    expected = [f'Cohort-specific eligibility criteria:\n{literal}\n\nCommon eligibility criteria:\nAge >= 18.',
                f'Cohort-specific eligibility criteria:\nPreviously treated.\n\nCommon eligibility criteria:\n{literal}',
                literal]
    return cohorts, patients, expected


@pytest.mark.parametrize('literal', ['None', 'NA', 'N/A', 'null', 'NaN'])
def test_csv_text_survives_sampling_generation_and_both_retrieval_paths(tmp_path, monkeypatch, literal):
    cohorts, patients, expected = inputs(tmp_path, literal)
    raw = read_table(cohorts)
    assert raw.cohort_specific_eligibility_criteria.iloc[0] == literal
    assert raw.common_eligibility_criteria.iloc[1] == literal
    assert pd.isna(raw.cohort_name.iloc[2])
    assert raw.cohort_count.tolist() == [2, 2, 1]
    dfci = tmp_path / 'dfci.csv'
    pd.DataFrame(columns=['nct_id', 'open_date', 'close_date', 'status']).to_csv(dfci, index=False)
    sampled_path = tmp_path / 'sample.csv'
    sampled = load_stage('0c_sample_trial_spaces.py').sample_trial_spaces(
        cohorts, sampled_path, dfci, n_trials=2, chunksize=1)
    expected_by_id = dict(zip(raw.cohort_trial_id, expected))
    expected = [expected_by_id[cid] for cid in sampled.cohort_trial_id]
    assert sampled.cohort_eligibility_criteria.tolist() == expected
    named_id = 'NCT00000001-cohort-0'
    assert read_clinical_csv(sampled_path).set_index('cohort_trial_id').loc[named_id, 'cohort_name'] == literal
    prepared = load_spaces(sampled_path)
    assert prepared.this_space.tolist() == expected
    ps, cs = load_retrieval_tables(SimpleNamespace(
        patients_parquet=patients, trials_file=sampled_path, splits=['train', 'val', 'test']))
    assert ps.patient_boilerplate_text.tolist() == [literal]
    assert cs.this_space.tolist() == expected

    # The ordinary production entrypoint uses the same reader without a launcher.
    output = tmp_path / 'corpus'
    result = prepare_corpus(patients, sampled_path, output)
    ps, cs = load_corpus(output)
    assert result['cohorts'] == 3 and result['source_trial_pairs'] == 2
    by_id = {c['id']: c for c in cs}
    assert by_id[named_id]['cohort_name'] == literal
    assert by_id['NCT00000002-cohort-0']['cohort_name'] is None
    assert [c['text'] for c in cs] == expected
    assert ps[0]['text'].endswith('Other relevant history:\n' + literal)

    # Exercise Generator.run's real file loading, stopping before any LLM work.
    generator = Generator(args_for(tmp_path), Tokenizer())
    class InputChecked(Exception):
        pass
    async def check_input(frame):
        assert frame.this_space.tolist() == expected
        assert frame.set_index('cohort_trial_id').loc[named_id, 'cohort_name'] == literal
        raise InputChecked
    monkeypatch.setattr(generator, 'generate_patient_events', check_input)
    with pytest.raises(InputChecked):
        asyncio.run(generator.run())


def test_direct_production_prepare_cli_preserves_literal_none(tmp_path):
    cohorts, patients, expected = inputs(tmp_path, 'None')
    output = tmp_path / 'corpus'
    process = subprocess.run([sys.executable, 'trialspace_training.py', 'prepare',
        '--patients', str(patients), '--cohorts', str(cohorts), '--output', str(output)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    assert [c['text'] for c in load_corpus(output)[1]] == expected


def test_empty_required_criteria_still_fail_validation(tmp_path):
    cohorts, patients, _ = inputs(tmp_path, 'None')
    frame = read_clinical_csv(cohorts)
    frame.loc[0, 'cohort_specific_eligibility_criteria'] = ''
    frame.to_csv(cohorts, index=False)
    with pytest.raises(ValueError, match='Missing cohort-specific criteria'):
        prepare_corpus(patients, cohorts, tmp_path / 'corpus')

    frame.loc[0, 'cohort_specific_eligibility_criteria'] = 'None'
    frame.loc[0, 'common_eligibility_criteria'] = ''
    frame.to_csv(cohorts, index=False)
    assert load_spaces(cohorts).this_space.iloc[0] == 'None'
