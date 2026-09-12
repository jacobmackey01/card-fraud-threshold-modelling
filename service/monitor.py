from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from service.model import EXPECTED_MODEL_FEATURES, ModelBundle, PCA_FEATURES, PROJECT_ROOT


def _engineer_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"Time", "Amount", *PCA_FEATURES}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Monitoring batch is missing required columns: {missing}")
    numeric = frame[["Time", "Amount", *PCA_FEATURES]].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Monitoring batch contains missing or non-finite model inputs")
    if (numeric["Time"] < 0.0).any() or (numeric["Amount"] < 0.0).any():
        raise ValueError("Time and Amount must be non-negative")
    hour = (numeric["Time"].to_numpy(dtype=float) % 86400.0) / 3600.0
    amount = numeric["Amount"].to_numpy(dtype=float)
    engineered = numeric[PCA_FEATURES].copy()
    engineered["Amount"] = amount
    engineered["Amount_log1p"] = np.log1p(amount)
    engineered["Amount_is_zero"] = (amount <= 0.0).astype(float)
    engineered["Hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    engineered["Hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    return engineered[EXPECTED_MODEL_FEATURES]


def analyze_frame(frame: pd.DataFrame, bundle: ModelBundle) -> dict[str, Any]:
    if frame.empty:
        raise ValueError("Monitoring batch is empty")
    features = _engineer_frame(frame)
    matrix = features.to_numpy(dtype=float)
    standardized = (matrix - bundle.mean) / bundle.scale
    logits = np.clip(standardized @ bundle.weights + bundle.intercept, -35.0, 35.0)
    scores = 1.0 / (1.0 + np.exp(-logits))
    alerts = scores >= bundle.threshold

    drift: list[dict[str, Any]] = []
    warning_features: list[str] = []
    failing_features: list[str] = []
    current_mean = matrix.mean(axis=0)
    current_std = matrix.std(axis=0)
    for index, feature_name in enumerate(bundle.feature_names):
        mean_shift_sd = abs(float(current_mean[index] - bundle.mean[index])) / float(bundle.scale[index])
        scale_ratio = float(current_std[index] / bundle.scale[index])
        status = "pass"
        if mean_shift_sd >= 1.0 or scale_ratio < 0.5 or scale_ratio > 2.0:
            status = "fail"
            failing_features.append(feature_name)
        elif mean_shift_sd >= 0.5 or scale_ratio < 0.67 or scale_ratio > 1.5:
            status = "warn"
            warning_features.append(feature_name)
        drift.append(
            {
                "feature": feature_name,
                "mean_shift_training_sd": round(mean_shift_sd, 6),
                "scale_ratio_to_training": round(scale_ratio, 6),
                "status": status,
            }
        )

    test_metrics = bundle.artifact.get("metrics", {}).get("test", {})
    reference_alert_rate = float(test_metrics.get("flagged_rate", 0.0))
    alert_rate = float(alerts.mean())
    sampling_band = 3.0 * math.sqrt(
        max(reference_alert_rate * (1.0 - reference_alert_rate), 1e-12) / len(frame)
    )
    alert_rate_limit = max(0.005, reference_alert_rate + sampling_band)
    alert_rate_status = "fail" if alert_rate > alert_rate_limit else "pass"

    overall_status = "fail" if failing_features or alert_rate_status == "fail" else "warn" if warning_features else "pass"
    return {
        "schema_version": 1,
        "model_version": bundle.model_version,
        "rows": int(len(frame)),
        "overall_status": overall_status,
        "score_summary": {
            "mean": round(float(scores.mean()), 8),
            "p50": round(float(np.quantile(scores, 0.50)), 8),
            "p95": round(float(np.quantile(scores, 0.95)), 8),
            "p99": round(float(np.quantile(scores, 0.99)), 8),
        },
        "alert_rate": {
            "observed": round(alert_rate, 8),
            "test_reference": round(reference_alert_rate, 8),
            "upper_control_limit": round(alert_rate_limit, 8),
            "status": alert_rate_status,
        },
        "feature_drift": drift,
        "summary": {
            "warning_features": warning_features,
            "failing_features": failing_features,
        },
        "boundary": (
            "Unlabelled input monitoring can detect distribution and queue changes, not recall or fraud losses. "
            "Delayed confirmed labels are required for performance monitoring."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check an unlabelled transaction batch for operational drift.")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fail-on-alert", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = analyze_frame(pd.read_csv(args.csv), ModelBundle.load(PROJECT_ROOT))
    payload = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(f"Wrote monitoring report: {args.output}")
    else:
        print(payload, end="")
    return 2 if args.fail_on_alert and report["overall_status"] != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
