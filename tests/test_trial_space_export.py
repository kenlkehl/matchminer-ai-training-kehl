import pandas as pd
import pytest

from trial_space_export import TRIAL_ONLY_COLUMNS, write_trial_exports


SPACE = ('Age range allowed: 18+. Sex allowed: Any. Cancer primary site allowed: Lung. '
         'Histology allowed: Adenocarcinoma. Disease state allowed: Metastatic. '
         'Prior treatment requirements: No prior systemic therapy. Biomarker requirements: EGFR mutation.')


def test_many_spaces_keep_full_trial_audit_once_and_link_to_it(tmp_path):
    raw = 'original reasoning and branch contexts ' * 50000
    spaces = '\n'.join(f'{index}. {SPACE}' for index in range(1, 101))
    row = dict(nct_id='NCT1', study_first_post_date='2020-01-02', title='Trial',
               trial_text='Original source', trial_boilerplate_text='Active infection',
               space_reasoning_and_output=raw, space_output_no_reasoning=spaces + '\nBoilerplate exclusions: infection',
               space_text=spaces)
    shard = tmp_path / 'shard.parquet'
    pd.DataFrame([row]).to_parquet(shard, index=False)
    trials, lineitems = tmp_path / 'trials.csv', tmp_path / 'spaces.csv'
    result = write_trial_exports([shard], trials, lineitems, row.keys())
    assert result == {'trials': 1, 'spaces': 100}
    full, individual = pd.read_csv(trials), pd.read_csv(lineitems)
    assert full.space_reasoning_and_output.iloc[0] == raw
    assert full.space_text.iloc[0] == spaces
    assert not set(individual) & TRIAL_ONLY_COLUMNS
    assert individual.this_space.eq(SPACE).all()
    assert individual.space_number.tolist() == list(range(1, 101))
    assert individual.reasoning_trials_path.eq(str(trials.resolve())).all()
    assert individual.nct_id.eq('NCT1').all()
    assert individual.study_first_post_date.eq('2020-01-02').all()
    assert lineitems.stat().st_size < len(raw) // 10


def test_streamed_multiple_shards_and_empty_export_have_the_same_columns(tmp_path):
    paths = []
    for index in range(2):
        path = tmp_path / f'shard{index}.parquet'
        pd.DataFrame([dict(nct_id=f'NCT{index}', space_text=f'1. {SPACE}',
                          space_reasoning_and_output=f'audit {index}',
                          space_output_no_reasoning=f'1. {SPACE}', trial_boilerplate_text='infection')]
                     ).to_parquet(path, index=False)
        paths.append(path)
    trials, spaces = tmp_path / 'trials.csv', tmp_path / 'spaces.csv'
    write_trial_exports(paths, trials, spaces, ['nct_id'])
    assert pd.read_csv(trials).space_reasoning_and_output.tolist() == ['audit 0', 'audit 1']
    expected = pd.read_csv(spaces)
    assert expected.nct_id.tolist() == ['NCT0', 'NCT1']
    write_trial_exports([], trials, spaces, ['nct_id'])
    assert pd.read_csv(spaces).empty
    assert pd.read_csv(spaces).columns.tolist() == expected.columns.tolist()


def test_duplicate_checkpoint_trials_do_not_replace_existing_exports(tmp_path):
    source = tmp_path / 'shard.parquet'
    pd.DataFrame([dict(nct_id='NCT1', space_text=f'1. {SPACE}')]).to_parquet(source, index=False)
    trials, spaces = tmp_path / 'trials.csv', tmp_path / 'spaces.csv'
    trials.write_text('original trials')
    spaces.write_text('original spaces')
    with pytest.raises(ValueError, match='Duplicate trial'):
        write_trial_exports([source, source], trials, spaces, ['nct_id'])
    assert trials.read_text() == 'original trials'
    assert spaces.read_text() == 'original spaces'
