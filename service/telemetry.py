from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from threading import Lock
from typing import Any


LOGGER = logging.getLogger("fraud_service")


def log_event(event: str, **fields: Any) -> None:
    payload = {"event": event, **fields}
    LOGGER.info(json.dumps(payload, separators=(",", ":"), sort_keys=True))


@dataclass
class PredictionMetrics:
    predictions: int = 0
    review_recommendations: int = 0
    errors: int = 0
    score_sum: float = 0.0
    _lock: Lock = field(default_factory=Lock, repr=False)

    def record(self, score: float, review_recommended: bool) -> None:
        with self._lock:
            self.predictions += 1
            self.review_recommendations += int(review_recommended)
            self.score_sum += score

    def record_error(self) -> None:
        with self._lock:
            self.errors += 1

    def snapshot(self) -> dict[str, float | int]:
        with self._lock:
            count = self.predictions
            return {
                "predictions": count,
                "review_recommendations": self.review_recommendations,
                "errors": self.errors,
                "alert_rate": self.review_recommendations / count if count else 0.0,
                "mean_score": self.score_sum / count if count else 0.0,
            }

    def prometheus_text(self, model_version: str) -> str:
        snapshot = self.snapshot()
        labels = f'{{model_version="{model_version}"}}'
        lines = [
            "# HELP fraud_predictions_total Number of scored transactions.",
            "# TYPE fraud_predictions_total counter",
            f"fraud_predictions_total{labels} {snapshot['predictions']}",
            "# HELP fraud_review_recommendations_total Number of transactions sent for review.",
            "# TYPE fraud_review_recommendations_total counter",
            f"fraud_review_recommendations_total{labels} {snapshot['review_recommendations']}",
            "# HELP fraud_prediction_errors_total Number of scoring errors.",
            "# TYPE fraud_prediction_errors_total counter",
            f"fraud_prediction_errors_total{labels} {snapshot['errors']}",
            "# HELP fraud_alert_rate Current process-lifetime review recommendation rate.",
            "# TYPE fraud_alert_rate gauge",
            f"fraud_alert_rate{labels} {snapshot['alert_rate']:.12g}",
            "# HELP fraud_mean_score Current process-lifetime mean model score.",
            "# TYPE fraud_mean_score gauge",
            f"fraud_mean_score{labels} {snapshot['mean_score']:.12g}",
        ]
        return "\n".join(lines) + "\n"
