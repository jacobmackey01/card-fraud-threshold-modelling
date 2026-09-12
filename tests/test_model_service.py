from __future__ import annotations

import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from service.app import create_app
from service.model import BundleError, ModelBundle, PCA_FEATURES


ROOT = Path(__file__).resolve().parents[1]


def zero_transaction(transaction_id: str | None = "synthetic-zero") -> dict[str, float | str | None]:
    transaction: dict[str, float | str | None] = {
        "transaction_id": transaction_id,
        "Time": 0.0,
        "Amount": 0.0,
    }
    transaction.update({name: 0.0 for name in PCA_FEATURES})
    return transaction


class ModelBundleTests(unittest.TestCase):
    def test_current_release_passes_integrity_checks_and_scores_the_contract(self) -> None:
        bundle = ModelBundle.load(ROOT)
        self.assertEqual(bundle.model_version, "1.0.0")
        self.assertEqual(len(bundle.weights), 33)
        prediction = bundle.predict(zero_transaction())
        self.assertTrue(math.isfinite(prediction.score))
        self.assertGreaterEqual(prediction.score, 0.0)
        self.assertLessEqual(prediction.score, 1.0)

    def test_tampered_artifact_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(ROOT / "artifacts", root / "artifacts")
            artifact_path = root / "artifacts" / "releases" / "1.0.0" / "fraud_model.json"
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            artifact["threshold"] = 0.99
            artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
            with self.assertRaisesRegex(BundleError, "artifact hash"):
                ModelBundle.load(root)

    def test_tampered_manifest_is_rejected_by_production_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(ROOT / "artifacts", root / "artifacts")
            manifest_path = root / "artifacts" / "releases" / "1.0.0" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["boundary"] = "changed"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(BundleError, "manifest hash"):
                ModelBundle.load(root)


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(create_app(ROOT))

    def test_health_and_model_metadata_are_versioned(self) -> None:
        ready = self.client.get("/health/ready")
        self.assertEqual(ready.status_code, 200)
        self.assertEqual(ready.json()["model_version"], "1.0.0")
        metadata = self.client.get("/v1/model")
        self.assertEqual(metadata.status_code, 200)
        self.assertIn("not an autonomous fraud decision", metadata.json()["boundary"])

    def test_prediction_is_deterministic_and_exposes_decision_boundary(self) -> None:
        payload = zero_transaction()
        first = self.client.post("/v1/predictions", json=payload)
        second = self.client.post("/v1/predictions", json=payload)
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(first.json()["score"], second.json()["score"])
        self.assertEqual(first.json()["threshold"], 0.053668899607021495)
        self.assertEqual(first.json()["transaction_id"], "synthetic-zero")
        self.assertIn("review_recommended", first.json())

    def test_invalid_and_extra_inputs_are_rejected(self) -> None:
        negative_amount = zero_transaction()
        negative_amount["Amount"] = -1.0
        response = self.client.post("/v1/predictions", json=negative_amount)
        self.assertEqual(response.status_code, 422)
        extra_field = zero_transaction()
        extra_field["Class"] = 0.0
        response = self.client.post("/v1/predictions", json=extra_field)
        self.assertEqual(response.status_code, 422)

    def test_batch_and_metrics(self) -> None:
        response = self.client.post(
            "/v1/predictions/batch",
            json={"transactions": [zero_transaction("a"), zero_transaction("b")]},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()["predictions"]), 2)
        metrics = self.client.get("/metrics")
        self.assertEqual(metrics.status_code, 200)
        self.assertIn("fraud_predictions_total", metrics.text)


if __name__ == "__main__":
    unittest.main()
