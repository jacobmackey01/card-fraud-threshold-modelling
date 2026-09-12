from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PCA_FEATURES = [f"V{i}" for i in range(1, 29)]
RAW_INPUT_FEATURES = ["Time", *PCA_FEATURES, "Amount"]
EXPECTED_MODEL_FEATURES = [
    *PCA_FEATURES,
    "Amount",
    "Amount_log1p",
    "Amount_is_zero",
    "Hour_sin",
    "Hour_cos",
]
MODEL_VERSION_PATTERN = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")


class BundleError(RuntimeError):
    """Raised when a release bundle is missing, malformed, or fails integrity checks."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise BundleError(f"Could not read {path}: {exc}") from exc
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise BundleError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise BundleError(f"Expected a JSON object in {path}")
    return parsed, payload


def _require_finite_sequence(value: Any, name: str, expected_length: int) -> np.ndarray:
    if not isinstance(value, list) or len(value) != expected_length:
        raise BundleError(f"{name} must contain exactly {expected_length} values")
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise BundleError(f"{name} must contain numeric values") from exc
    if not np.isfinite(array).all():
        raise BundleError(f"{name} contains a non-finite value")
    return array


@dataclass(frozen=True)
class Prediction:
    score: float
    review_recommended: bool


@dataclass(frozen=True)
class ModelBundle:
    model_version: str
    artifact_sha256: str
    manifest_sha256: str
    model_name: str
    threshold: float
    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    intercept: float
    artifact: Mapping[str, Any]
    manifest: Mapping[str, Any]

    @classmethod
    def load(cls, project_root: Path = PROJECT_ROOT) -> "ModelBundle":
        project_root = project_root.resolve()
        pointer_path = project_root / "artifacts" / "production.json"
        pointer, _ = _read_json(pointer_path)
        version = pointer.get("model_version")
        expected_manifest_hash = pointer.get("manifest_sha256")
        if not isinstance(version, str) or not MODEL_VERSION_PATTERN.fullmatch(version):
            raise BundleError("production.json has an invalid model_version")
        if not isinstance(expected_manifest_hash, str) or len(expected_manifest_hash) != 64:
            raise BundleError("production.json has an invalid manifest_sha256")
        return cls.load_release(
            project_root=project_root,
            model_version=version,
            expected_manifest_hash=expected_manifest_hash,
        )

    @classmethod
    def load_release(
        cls,
        *,
        project_root: Path,
        model_version: str,
        expected_manifest_hash: str | None = None,
    ) -> "ModelBundle":
        if not MODEL_VERSION_PATTERN.fullmatch(model_version):
            raise BundleError(f"Invalid semantic model version: {model_version!r}")
        project_root = project_root.resolve()
        release_dir = project_root / "artifacts" / "releases" / model_version
        manifest, manifest_bytes = _read_json(release_dir / "manifest.json")
        manifest_hash = sha256_bytes(manifest_bytes)
        if expected_manifest_hash is not None and manifest_hash != expected_manifest_hash:
            raise BundleError("Release manifest hash does not match production.json")
        if manifest.get("schema_version") != 1:
            raise BundleError("Unsupported release manifest schema_version")
        if manifest.get("model_version") != model_version:
            raise BundleError("Release directory and manifest model_version do not match")
        artifact_filename = manifest.get("artifact_file")
        if artifact_filename != "fraud_model.json":
            raise BundleError("Release manifest must reference fraud_model.json")

        artifact, artifact_bytes = _read_json(release_dir / artifact_filename)
        artifact_hash = sha256_bytes(artifact_bytes)
        if artifact_hash != manifest.get("artifact_sha256"):
            raise BundleError("Model artifact hash does not match its release manifest")

        feature_names = artifact.get("feature_names")
        if feature_names != EXPECTED_MODEL_FEATURES:
            raise BundleError("Model artifact feature contract is missing, reordered, or unsupported")
        feature_count = len(EXPECTED_MODEL_FEATURES)
        standardizer = artifact.get("standardizer")
        if not isinstance(standardizer, dict):
            raise BundleError("Model artifact is missing its standardizer")
        mean = _require_finite_sequence(standardizer.get("mean"), "standardizer.mean", feature_count)
        scale = _require_finite_sequence(standardizer.get("scale"), "standardizer.scale", feature_count)
        if np.any(scale <= 0):
            raise BundleError("standardizer.scale must be strictly positive")
        weights = _require_finite_sequence(artifact.get("weights"), "weights", feature_count)

        try:
            intercept = float(artifact["intercept"])
            threshold = float(artifact["threshold"])
        except (KeyError, TypeError, ValueError) as exc:
            raise BundleError("Model artifact has an invalid intercept or threshold") from exc
        if not math.isfinite(intercept):
            raise BundleError("Model intercept must be finite")
        if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
            raise BundleError("Model threshold must be between zero and one")
        model_name = artifact.get("model_name")
        if not isinstance(model_name, str) or not model_name.endswith("logistic"):
            raise BundleError("The JSON service currently supports only the audited logistic model format")

        return cls(
            model_version=model_version,
            artifact_sha256=artifact_hash,
            manifest_sha256=manifest_hash,
            model_name=model_name,
            threshold=threshold,
            feature_names=tuple(feature_names),
            mean=mean,
            scale=scale,
            weights=weights,
            intercept=intercept,
            artifact=artifact,
            manifest=manifest,
        )

    @staticmethod
    def engineer_features(transaction: Mapping[str, float]) -> np.ndarray:
        missing = [name for name in RAW_INPUT_FEATURES if name not in transaction]
        if missing:
            raise ValueError(f"Missing required input fields: {missing}")
        try:
            event_time = float(transaction["Time"])
            amount = float(transaction["Amount"])
            pca_values = [float(transaction[name]) for name in PCA_FEATURES]
        except (TypeError, ValueError) as exc:
            raise ValueError("All model inputs must be numeric") from exc
        raw = np.asarray([event_time, amount, *pca_values], dtype=float)
        if not np.isfinite(raw).all():
            raise ValueError("Model inputs must be finite")
        if event_time < 0.0:
            raise ValueError("Time must be non-negative")
        if amount < 0.0:
            raise ValueError("Amount must be non-negative")
        hour = (event_time % 86400.0) / 3600.0
        engineered = [
            *pca_values,
            amount,
            math.log1p(amount),
            float(amount <= 0.0),
            math.sin(2.0 * math.pi * hour / 24.0),
            math.cos(2.0 * math.pi * hour / 24.0),
        ]
        return np.asarray(engineered, dtype=float)

    def predict(self, transaction: Mapping[str, float]) -> Prediction:
        features = self.engineer_features(transaction)
        standardized = (features - self.mean) / self.scale
        logit = float(np.dot(standardized, self.weights) + self.intercept)
        clipped = min(35.0, max(-35.0, logit))
        score = 1.0 / (1.0 + math.exp(-clipped))
        return Prediction(score=score, review_recommended=score >= self.threshold)

    def public_metadata(self) -> dict[str, Any]:
        metrics = self.artifact.get("metrics", {})
        test_metrics = metrics.get("test", {}) if isinstance(metrics, dict) else {}
        return {
            "model_version": self.model_version,
            "model_name": self.model_name,
            "artifact_sha256": self.artifact_sha256,
            "threshold": self.threshold,
            "threshold_strategy": self.artifact.get("threshold_strategy"),
            "input_schema": list(RAW_INPUT_FEATURES),
            "test_metrics": {
                key: test_metrics.get(key)
                for key in ("precision", "recall", "average_precision", "flagged_rate")
            },
            "boundary": (
                "Portfolio case study using anonymised PCA inputs and illustrative costs; "
                "review_recommended is not an autonomous fraud decision."
            ),
        }
