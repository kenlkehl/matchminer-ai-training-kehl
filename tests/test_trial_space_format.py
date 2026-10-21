"""Check the downstream sampling filter without reading patient-linked exclusions."""
import ast
from pathlib import Path
import pandas as pd
from trial_space_format import has_required_space_fields

SPACE = ('Age range allowed: 18+. Sex allowed: Any. Cancer primary site allowed: lung. '
         'Histology allowed: adenocarcinoma. Disease state allowed: metastatic. '
         'Prior treatment requirements: prior cisplatin AND no prior pembrolizumab. '
         'Biomarker requirements: KRAS G12D AND no EGFR mutation.')


def _sampling_function(name, namespace):
    path = Path(__file__).parents[1] / '0c_sample_trial_spaces.py'
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def test_sampling_keeps_combined_requirements_and_rejects_missing_field():
    clean_trial_spaces = _sampling_function('clean_trial_spaces', {'has_required_space_fields': has_required_space_fields})
    frame = pd.DataFrame({'this_space': [SPACE, SPACE.replace('Biomarker requirements:', 'Missing:')]})
    assert clean_trial_spaces(frame).tolist() == [True, False]
    assert frame.iloc[0].this_space == SPACE


def test_dfci_trials_held_out_unless_dates_fall_outside_window():
    open_during = _sampling_function('dfci_nct_ids_open_during', {'pd': pd})
    rows = [
        ('NCT00000001', 'IRB STUDY CLOSURE', '2018-01-05', '2024-12-31'),
        ('NCT00000002', 'IRB STUDY CLOSURE', '2018-01', '2024-12'),
        ('NCT00000003', 'CLOSED TO ACCRUAL', '2019-05', '2025-01'),
        ('NCT00000004', 'NEW', '2026-12', '2029-06-30'),
        ('NCT00000005', 'NEW', '2027-03', '2030-01'),
        ('NCT00000006', 'OPEN TO ACCRUAL', '2027-03', '2030-01'),
        ('NCT00000007', 'OPEN TO ACCRUAL', '2016-02', '2023-08'),
        ('NCT00000008', 'IRB STUDY CLOSURE', '2010-04', None),
        ('NCT00000009', 'IRB STUDY CLOSURE', None, None),
        ('NCT00000010', 'IRB STUDY CLOSURE', None, '2020-06'),
        ('NCT00000011', 'IRB STUDY CLOSURE', '2015-01', '2020-01'),
        ('NCT00000011', 'CLOSED TO ACCRUAL', '2022-01', '2026-01'),
        (None, 'OPEN TO ACCRUAL', '2025-01', '2028-01'),
    ]
    dfci_trials = pd.DataFrame(rows, columns=['nct_id', 'status', 'open_date', 'close_date'])
    assert open_during(dfci_trials, 2025, 2026) == {
        'NCT00000003', 'NCT00000004', 'NCT00000006', 'NCT00000007',
        'NCT00000008', 'NCT00000009', 'NCT00000011',
    }
