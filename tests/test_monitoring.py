from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from service.model import ModelBundle, PCA_FEATURES
from service.monitor import analyze_frame


ROOT = Path(__file__).resolve().parents[1]


def monitoring_frame(rows: int = 200) -> pd.DataFrame:
    data: dict[str, list[float]] = {
        "Time": [float(index * 600) for index in range(rows)],
        "Amount": [25.0 + float(index % 10) for index in range(rows)],
    }
    for index, name in enumerate(PCA_FEATURES, start=1):
        data[name] = [((row % 11) - 5) * 0.05 + index * 0.001 for row in range(rows)]
    return pd.DataFrame(data)


class MonitoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = ModelBundle.load(ROOT)

    def test_report_is_versioned_and_keeps_unlabelled_boundary(self) -> None:
        report = analyze_frame(monitoring_frame(), self.bundle)
        self.assertEqual(report["model_version"], "1.0.0")
        self.assertEqual(report["rows"], 200)
        self.assertIn(report["overall_status"], {"pass", "warn", "fail"})
        self.assertEqual(len(report["feature_drift"]), 33)
        self.assertIn("Delayed confirmed labels", report["boundary"])

    def test_missing_feature_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            analyze_frame(monitoring_frame().drop(columns=["V14"]), self.bundle)


if __name__ == "__main__":
    unittest.main()
