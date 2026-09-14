from __future__ import annotations

import argparse
import json
import re
import tempfile
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _json_request(base_url: str, path: str, *, method: str = "GET") -> dict[str, Any]:
    request = urllib.request.Request(f"{base_url.rstrip('/')}{path}", method=method)
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.load(response)
    except (OSError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{method} {path} failed: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{method} {path} did not return a JSON object")
    return payload


def _text_request(base_url: str, path: str) -> str:
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}{path}", timeout=20) as response:
            return response.read().decode("utf-8")
    except (OSError, urllib.error.HTTPError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"GET {path} failed: {exc}") from exc


def validate_live_contract(
    *,
    ready: dict[str, Any],
    model: dict[str, Any],
    prediction: dict[str, Any],
    metrics: str,
    expected_commit: str,
    expected_container_digest: str,
    expected_model_release: str,
    expected_model_file_digest: str,
    expected_threshold: float,
    expected_fixture: str,
) -> None:
    expected_deployment = {
        "git_commit": expected_commit,
        "container_digest": expected_container_digest,
        "model_release": expected_model_release,
        "model_file_sha256": expected_model_file_digest,
        "decision_threshold": expected_threshold,
        "synthetic_fixture_version": expected_fixture,
        "input_policy": "checked_in_synthetic_fixture_only",
        "rehearsal": True,
    }
    for endpoint_name, payload in (("/health/ready", ready), ("/v1/model", model)):
        deployment = payload.get("deployment")
        if not isinstance(deployment, dict):
            raise RuntimeError(f"{endpoint_name} is missing deployment provenance")
        for key, expected in expected_deployment.items():
            if deployment.get(key) != expected:
                raise RuntimeError(
                    f"{endpoint_name} reported {key}={deployment.get(key)!r}; expected {expected!r}"
                )
    if ready.get("status") != "ready":
        raise RuntimeError("/health/ready did not report ready")
    if ready.get("artifact_sha256") != expected_model_file_digest:
        raise RuntimeError("/health/ready model digest does not match the expected release")
    expected_prediction = {
        "fixture_id": expected_fixture,
        "transaction_id": expected_fixture,
        "model_version": expected_model_release,
        "threshold": expected_threshold,
    }
    for key, expected in expected_prediction.items():
        if prediction.get(key) != expected:
            raise RuntimeError(
                f"/v1/demo-prediction reported {key}={prediction.get(key)!r}; expected {expected!r}"
            )
    if not isinstance(prediction.get("review_recommended"), bool):
        raise RuntimeError("/v1/demo-prediction did not return a boolean outcome")
    metric_match = re.search(
        rf'^fraud_predictions_total\{{model_version="{re.escape(expected_model_release)}"\}} ([0-9]+)$',
        metrics,
        flags=re.MULTILINE,
    )
    if metric_match is None or int(metric_match.group(1)) < 1:
        raise RuntimeError("/metrics did not record the synthetic deployment check")


def verify_deployment(args: argparse.Namespace) -> dict[str, Any]:
    ready = _json_request(args.base_url, "/health/ready")
    model = _json_request(args.base_url, "/v1/model")
    prediction = _json_request(args.base_url, "/v1/demo-prediction", method="POST")
    metrics = _text_request(args.base_url, "/metrics")
    validate_live_contract(
        ready=ready,
        model=model,
        prediction=prediction,
        metrics=metrics,
        expected_commit=args.expected_commit,
        expected_container_digest=args.expected_container_digest,
        expected_model_release=args.expected_model_release,
        expected_model_file_digest=args.expected_model_file_digest,
        expected_threshold=args.expected_threshold,
        expected_fixture=args.expected_fixture,
    )
    return {
        "schema_version": 1,
        "verification_status": "passed",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "service_url": args.base_url,
        "cloud_run_revision": ready["deployment"].get("cloud_run_revision"),
        "git_commit": args.expected_commit,
        "container_digest": args.expected_container_digest,
        "model_release": args.expected_model_release,
        "model_file_sha256": args.expected_model_file_digest,
        "decision_threshold": args.expected_threshold,
        "synthetic_fixture_version": args.expected_fixture,
        "prediction": {
            "score": prediction.get("score"),
            "review_recommended": prediction.get("review_recommended"),
        },
        "verified_endpoints": [
            "/health/ready",
            "/v1/model",
            "/v1/demo-prediction",
            "/metrics",
        ],
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as temporary:
        json.dump(payload, temporary, indent=2)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify and record the live synthetic deployment contract.")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-container-digest", required=True)
    parser.add_argument("--expected-model-release", required=True)
    parser.add_argument("--expected-model-file-digest", required=True)
    parser.add_argument("--expected-threshold", type=float, required=True)
    parser.add_argument("--expected-fixture", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    evidence = verify_deployment(args)
    _write_json_atomic(args.output, evidence)
    print(f"Verified live synthetic deployment: {args.base_url}")
    print(f"Wrote provenance evidence: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
