from __future__ import annotations

import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.package_model import package_model
from scripts.promote_model import promote
from service.model import ModelBundle, PCA_FEATURES
from src.fraud_pipeline import engineer_features


ROOT = Path(__file__).resolve().parents[1]


def representative_transaction() -> dict[str, float]:
    transaction = {
        "Time": 91_234.0,
        "Amount": 123.45,
    }
    transaction.update({name: (index - 14.0) / 10.0 for index, name in enumerate(PCA_FEATURES, start=1)})
    return transaction


class InferenceParityTests(unittest.TestCase):
    def test_service_feature_engineering_matches_training_pipeline(self) -> None:
        bundle = ModelBundle.load(ROOT)
        transaction = representative_transaction()
        training_frame = engineer_features(pd.DataFrame([transaction]))
        training_features = training_frame[list(bundle.feature_names)].to_numpy(dtype=float)[0]
        standardized = (training_features - bundle.mean) / bundle.scale
        expected_logit = float(np.dot(standardized, bundle.weights) + bundle.intercept)
        expected_score = 1.0 / (1.0 + math.exp(-max(-35.0, min(35.0, expected_logit))))
        self.assertAlmostEqual(bundle.predict(transaction).score, expected_score, places=14)


class ReleaseWorkflowTests(unittest.TestCase):
    def test_release_is_immutable_and_promotion_uses_compare_and_swap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate_dir = root / "artifacts"
            candidate_dir.mkdir(parents=True)
            candidate = candidate_dir / "fraud_model.json"
            shutil.copy2(ROOT / "artifacts" / "fraud_model.json", candidate)

            package_model(project_root=root, candidate=candidate, model_version="2.0.0")
            pointer = promote(
                project_root=root,
                model_version="2.0.0",
                expected_current=None,
            )
            self.assertEqual(ModelBundle.load(root).model_version, "2.0.0")
            before = pointer.read_bytes()

            with self.assertRaises(FileExistsError):
                package_model(project_root=root, candidate=candidate, model_version="2.0.0")

            package_model(project_root=root, candidate=candidate, model_version="2.0.1")
            with self.assertRaisesRegex(RuntimeError, "Production pointer changed"):
                promote(
                    project_root=root,
                    model_version="2.0.1",
                    expected_current="1.9.9",
                )
            self.assertEqual(pointer.read_bytes(), before)
            current = json.loads(pointer.read_text(encoding="utf-8"))
            self.assertEqual(current["model_version"], "2.0.0")


if __name__ == "__main__":
    unittest.main()
