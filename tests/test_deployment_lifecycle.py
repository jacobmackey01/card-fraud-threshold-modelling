from __future__ import annotations

import json
import logging
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from scripts.cloud_run_traffic import require_transition, serving_revision
from scripts.verify_deployment import validate_live_contract
from service.app import create_app


ROOT = Path(__file__).resolve().parents[1]
COMMIT = "a" * 40
IMAGE_DIGEST = "sha256:" + "b" * 64
MODEL_DIGEST = "9800efa3737531fdd0df78bcbb782dd0108cbfca593a76d33f0f534a1842645d"
MODEL_VERSION = "1.0.0"
THRESHOLD = 0.053668899607021495
FIXTURE = "synthetic-demo-v1"


class SyntheticDeploymentApiTests(unittest.TestCase):
    def setUp(self) -> None:
        environment = {
            "DEPLOYED_GIT_COMMIT": COMMIT,
            "DEPLOYED_IMAGE_DIGEST": IMAGE_DIGEST,
            "K_REVISION": "fraud-demo-rev-001",
        }
        with patch.dict("os.environ", environment, clear=False):
            self.client = TestClient(create_app(ROOT, synthetic_demo_only=True))

    def test_demo_endpoint_uses_only_the_checked_in_fixture(self) -> None:
        response = self.client.post("/v1/demo-prediction")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["fixture_id"], FIXTURE)
        self.assertEqual(payload["transaction_id"], FIXTURE)
        self.assertEqual(payload["model_version"], MODEL_VERSION)
        self.assertEqual(payload["threshold"], THRESHOLD)
        self.assertIn("review_recommended", payload)

        self.assertEqual(self.client.post("/v1/predictions", json={}).status_code, 404)
        self.assertEqual(self.client.post("/v1/demo-prediction", json={}).status_code, 400)

    def test_live_metadata_exposes_the_complete_provenance_contract(self) -> None:
        ready = self.client.get("/health/ready").json()
        model = self.client.get("/v1/model").json()
        for payload in (ready, model):
            deployment = payload["deployment"]
            self.assertTrue(deployment["rehearsal"])
            self.assertEqual(deployment["input_policy"], "checked_in_synthetic_fixture_only")
            self.assertEqual(deployment["git_commit"], COMMIT)
            self.assertEqual(deployment["container_digest"], IMAGE_DIGEST)
            self.assertEqual(deployment["model_release"], MODEL_VERSION)
            self.assertEqual(deployment["model_file_sha256"], MODEL_DIGEST)
            self.assertEqual(deployment["decision_threshold"], THRESHOLD)
            self.assertEqual(deployment["synthetic_fixture_version"], FIXTURE)

    def test_demo_log_contains_fixture_revision_and_outcome_but_no_features(self) -> None:
        with self.assertLogs("fraud_service", level=logging.INFO) as captured:
            response = self.client.post("/v1/demo-prediction")
        self.assertEqual(response.status_code, 200)
        event = json.loads(captured.output[-1].split(":", 2)[-1])
        self.assertEqual(event["fixture_id"], FIXTURE)
        self.assertEqual(event["cloud_run_revision"], "fraud-demo-rev-001")
        self.assertIn("latency_ms", event)
        self.assertIn("review_recommended", event)
        for feature_name in ("Time", "Amount", *(f"V{i}" for i in range(1, 29))):
            self.assertNotIn(feature_name, event)


class DeploymentVerificationTests(unittest.TestCase):
    def test_validator_accepts_matching_live_provenance(self) -> None:
        deployment = {
            "git_commit": COMMIT,
            "container_digest": IMAGE_DIGEST,
            "model_release": MODEL_VERSION,
            "model_file_sha256": MODEL_DIGEST,
            "decision_threshold": THRESHOLD,
            "synthetic_fixture_version": FIXTURE,
            "input_policy": "checked_in_synthetic_fixture_only",
            "rehearsal": True,
        }
        validate_live_contract(
            ready={"status": "ready", "artifact_sha256": MODEL_DIGEST, "deployment": deployment},
            model={"deployment": deployment},
            prediction={
                "fixture_id": FIXTURE,
                "transaction_id": FIXTURE,
                "model_version": MODEL_VERSION,
                "threshold": THRESHOLD,
                "review_recommended": False,
            },
            metrics='fraud_predictions_total{model_version="1.0.0"} 1\n',
            expected_commit=COMMIT,
            expected_container_digest=IMAGE_DIGEST,
            expected_model_release=MODEL_VERSION,
            expected_model_file_digest=MODEL_DIGEST,
            expected_threshold=THRESHOLD,
            expected_fixture=FIXTURE,
        )

    def test_validator_rejects_a_different_running_image(self) -> None:
        deployment = {
            "git_commit": COMMIT,
            "container_digest": "sha256:" + "c" * 64,
            "model_release": MODEL_VERSION,
            "model_file_sha256": MODEL_DIGEST,
            "decision_threshold": THRESHOLD,
            "synthetic_fixture_version": FIXTURE,
            "input_policy": "checked_in_synthetic_fixture_only",
            "rehearsal": True,
        }
        with self.assertRaisesRegex(RuntimeError, "container_digest"):
            validate_live_contract(
                ready={"status": "ready", "artifact_sha256": MODEL_DIGEST, "deployment": deployment},
                model={"deployment": deployment},
                prediction={},
                metrics="",
                expected_commit=COMMIT,
                expected_container_digest=IMAGE_DIGEST,
                expected_model_release=MODEL_VERSION,
                expected_model_file_digest=MODEL_DIGEST,
                expected_threshold=THRESHOLD,
                expected_fixture=FIXTURE,
            )


class GuardedTrafficTests(unittest.TestCase):
    def test_serving_revision_requires_one_full_traffic_target(self) -> None:
        service = {"status": {"traffic": [{"revisionName": "revision-a", "percent": 100}]}}
        self.assertEqual(serving_revision(service), "revision-a")
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            serving_revision(
                {
                    "status": {
                        "traffic": [
                            {"revisionName": "revision-a", "percent": 50},
                            {"revisionName": "revision-b", "percent": 50},
                        ]
                    }
                }
            )

    def test_transition_is_compare_and_swap_guarded(self) -> None:
        require_transition(current="revision-a", expected_current="revision-a", target="revision-b")
        with self.assertRaisesRegex(RuntimeError, "Traffic changed"):
            require_transition(current="revision-c", expected_current="revision-a", target="revision-b")
        with self.assertRaisesRegex(RuntimeError, "already receives"):
            require_transition(current="revision-a", expected_current="revision-a", target="revision-a")


if __name__ == "__main__":
    unittest.main()
