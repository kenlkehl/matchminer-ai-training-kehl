import importlib.util
import contextlib
import io
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_UTILS_PATHS = [
    "real_eval_code/eval_phi_enrollments/eval_utils.py",
    "real_eval_code/eval_phi_soc/eval_utils.py",
]


def load_module(relative_path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(
        module_name,
        REPO_ROOT / relative_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EvalConfidenceIntervalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modules = [
            load_module(path, f"eval_utils_ci_{idx}")
            for idx, path in enumerate(EVAL_UTILS_PATHS)
        ]

    def test_binary_metric_cis_are_deterministic(self):
        actual = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        predicted = np.array([0.05, 0.20, 0.80, 0.70, 0.35, 0.90, 0.10, 0.65])

        for module in self.modules:
            with self.subTest(module=module.__file__):
                auc_ci = module.bootstrap_metric_ci(
                    [actual, predicted],
                    module.binary_auroc_score,
                    n_bootstrap=200,
                    random_state=123,
                )
                auc_ci_again = module.bootstrap_metric_ci(
                    [actual, predicted],
                    module.binary_auroc_score,
                    n_bootstrap=200,
                    random_state=123,
                )
                auprc_ci = module.bootstrap_metric_ci(
                    [actual, predicted],
                    module.average_precision_metric,
                    n_bootstrap=200,
                    random_state=123,
                )

                self.assertEqual(auc_ci, auc_ci_again)
                self.assertIsNotNone(auc_ci)
                self.assertIsNotNone(auprc_ci)
                self.assertGreater(auc_ci["n_bootstrap"], 0)
                self.assertLessEqual(0.0, auc_ci["lower"])
                self.assertLessEqual(auc_ci["lower"], auc_ci["upper"])
                self.assertLessEqual(auc_ci["upper"], 1.0)
                self.assertIn("95% CI [", module.format_metric_with_ci(0.9, auc_ci))

    def test_invalid_bootstrap_samples_are_skipped(self):
        actual = np.zeros(8)
        predicted = np.linspace(0.1, 0.8, 8)

        for module in self.modules:
            with self.subTest(module=module.__file__):
                auc_ci = module.bootstrap_metric_ci(
                    [actual, predicted],
                    module.binary_auroc_score,
                    n_bootstrap=50,
                    random_state=123,
                )
                self.assertIsNone(auc_ci)
                self.assertEqual(
                    module.format_metric_with_ci(0.5, auc_ci),
                    "0.5000 (95% CI N/A)",
                )

    def test_grouped_map_at_k_ci(self):
        frame = pd.DataFrame({
            "patient": ["p1", "p1", "p2", "p2", "p3", "p3"],
            "label": [1, 0, 0, 1, 1, 1],
        })

        for module in self.modules:
            with self.subTest(module=module.__file__):
                map_k, stats = module.calculate_map_at_k(
                    frame,
                    group_col="patient",
                    label_col="label",
                    k=2,
                    n_bootstrap=200,
                    random_state=123,
                )

                self.assertAlmostEqual(map_k, (1.0 + 0.5 + 1.0) / 3)
                self.assertIsNotNone(stats["map_at_k_ci"])
                self.assertIsNotNone(stats["positive_rate_ci"])
                self.assertLessEqual(0.0, stats["map_at_k_ci"]["lower"])
                self.assertLessEqual(stats["map_at_k_ci"]["upper"], 1.0)

    def test_categorical_metric_cis_are_returned(self):
        actual = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
        probs = np.array([
            [0.80, 0.10, 0.10],
            [0.20, 0.70, 0.10],
            [0.15, 0.15, 0.70],
            [0.55, 0.35, 0.10],
            [0.10, 0.75, 0.15],
            [0.20, 0.20, 0.60],
            [0.65, 0.25, 0.10],
            [0.15, 0.65, 0.20],
            [0.10, 0.30, 0.60],
            [0.45, 0.40, 0.15],
            [0.20, 0.55, 0.25],
            [0.10, 0.35, 0.55],
        ])

        for module in self.modules:
            with self.subTest(module=module.__file__):
                with contextlib.redirect_stdout(io.StringIO()):
                    metrics = module.eval_model_categorical(
                        probs,
                        actual,
                        ["none", "partial", "good"],
                        pdf_path=None,
                        n_bootstrap=100,
                        random_state=123,
                    )

                self.assertIsNotNone(metrics)
                self.assertIn("accuracy_ci", metrics)
                self.assertIn("macro_f1_ci", metrics)
                self.assertIn("binary_auroc_ci", metrics)
                self.assertIsNotNone(metrics["accuracy_ci"])
                self.assertIsNotNone(metrics["macro_f1_ci"])
                self.assertIsNotNone(metrics["binary_auroc_ci"])


if __name__ == "__main__":
    unittest.main()
