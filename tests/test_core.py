"""Focused regression tests for core data, feature, and metric helpers."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import load_cycle_csv, summarize_missingness, validate_columns
from src.evaluation import regression_metrics
from src.features import extract_degradation_trend, extract_impedance_features


class DegradationTrendTests(unittest.TestCase):
    def test_trailing_mean_uses_only_current_and_past_values(self):
        values = [1.0, 2.0, 3.0, 4.0, 100.0]
        actual = extract_degradation_trend(values, window=3)
        np.testing.assert_allclose(actual.to_numpy(), [1.0, 1.5, 2.0, 3.0, 35.6666666667])

    def test_future_change_does_not_change_earlier_features(self):
        first = extract_degradation_trend([1.0, 2.0, 3.0, 4.0], window=3)
        changed_future = extract_degradation_trend([1.0, 2.0, 3.0, 400.0], window=3)
        np.testing.assert_allclose(first.iloc[:3], changed_future.iloc[:3])

    def test_window_must_be_positive_integer(self):
        for invalid in (0, -1, 1.5, True):
            with self.subTest(window=invalid), self.assertRaises(ValueError):
                extract_degradation_trend([1, 2], window=invalid)


class ImpedanceFeatureTests(unittest.TestCase):
    def test_descriptive_statistics(self):
        result = extract_impedance_features(pd.DataFrame({"Re": [1, 3, 5]}), ["Re"])
        self.assertEqual(result.loc[0, "Re_mean"], 3.0)
        self.assertEqual(result.loc[0, "Re_min"], 1.0)
        self.assertEqual(result.loc[0, "Re_max"], 5.0)
        self.assertAlmostEqual(result.loc[0, "Re_std"], 2.0)

    def test_missing_column_is_reported(self):
        with self.assertRaisesRegex(ValueError, "not found"):
            extract_impedance_features(pd.DataFrame({"Re": [1]}), ["Rct"])

    def test_all_non_numeric_values_produce_nan_statistics(self):
        result = extract_impedance_features(pd.DataFrame({"Re": ["bad", None]}), ["Re"])
        self.assertTrue(result.loc[0, ["Re_mean", "Re_std", "Re_min", "Re_max"]].isna().all())


class DataLoaderTests(unittest.TestCase):
    def test_load_cycle_csv_and_reject_duplicate_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cycles.csv"
            path.write_text("Cycle,Capacity\n1,2.0\n", encoding="utf-8")
            frame = load_cycle_csv(path)
            self.assertEqual(frame.shape, (1, 2))
            duplicate_path = Path(directory) / "duplicate.csv"
            duplicate_path.write_text("Cycle,Cycle\n1,2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate column"):
                load_cycle_csv(duplicate_path)

    def test_missing_columns_and_missingness_summary(self):
        frame = pd.DataFrame({"Cycle": [1, 2], "Capacity": [2.0, np.nan]})
        validate_columns(frame, ["Cycle", "Capacity"])
        with self.assertRaisesRegex(ValueError, "Missing required"):
            validate_columns(frame, ["RUL"])
        summary = summarize_missingness(frame)
        self.assertEqual(summary.loc["Capacity", "missing_count"], 1)
        self.assertEqual(summary.loc["Capacity", "missing_percent"], 50.0)


class RegressionMetricTests(unittest.TestCase):
    def test_known_regression_metrics(self):
        result = regression_metrics([1, 2, 3], [1, 2, 4])
        self.assertAlmostEqual(result["mae"], 1 / 3)
        self.assertAlmostEqual(result["rmse"], np.sqrt(1 / 3))
        self.assertAlmostEqual(result["r2"], 0.5)

    def test_rejects_mismatched_or_non_finite_inputs(self):
        with self.assertRaisesRegex(ValueError, "same number"):
            regression_metrics([1, 2], [1])
        with self.assertRaisesRegex(ValueError, "finite"):
            regression_metrics([1, np.nan], [1, 2])

    def test_single_observation_has_undefined_r2(self):
        result = regression_metrics([2], [2.5])
        self.assertTrue(np.isnan(result["r2"]))


if __name__ == "__main__":
    unittest.main()
