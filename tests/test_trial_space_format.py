"""Check the downstream sampling filter without reading patient-linked exclusions."""
import ast
from pathlib import Path
import pandas as pd
from trial_space_format import has_required_space_fields

SPACE = ('Age range allowed: 18+. Sex allowed: Any. Cancer primary site allowed: lung. '
         'Histology allowed: adenocarcinoma. Disease state allowed: metastatic. '
         'Prior treatment requirements: prior cisplatin AND no prior pembrolizumab. '
         'Biomarker requirements: KRAS G12D AND no EGFR mutation.')


def test_sampling_keeps_combined_requirements_and_rejects_missing_field():
    path = Path(__file__).parents[1] / '0c_sample_trial_spaces.py'
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'clean_trial_spaces')
    namespace = {'has_required_space_fields': has_required_space_fields}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    frame = pd.DataFrame({'this_space': [SPACE, SPACE.replace('Biomarker requirements:', 'Missing:')]})
    assert namespace['clean_trial_spaces'](frame).tolist() == [True, False]
    assert frame.iloc[0].this_space == SPACE
