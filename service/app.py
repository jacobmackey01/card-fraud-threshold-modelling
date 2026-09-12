from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from service.model import BundleError, ModelBundle, PROJECT_ROOT
from service.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionResponse,
    Transaction,
)
from service.telemetry import PredictionMetrics, log_event


logging.basicConfig(level=logging.INFO, format="%(message)s")


def create_app(project_root: Path = PROJECT_ROOT) -> FastAPI:
    bundle = ModelBundle.load(project_root)
    metrics = PredictionMetrics()
    app = FastAPI(
        title="Fraud Review Recommendation API",
        version="1.0.0",
        description=(
            "Scores anonymised ULB credit-card transactions with a versioned portfolio model. "
            "A positive result recommends human review; it is not an autonomous fraud decision."
        ),
    )
    app.state.bundle = bundle
    app.state.metrics = metrics

    @app.exception_handler(BundleError)
    async def bundle_error_handler(_: Request, exc: BundleError) -> JSONResponse:
        metrics.record_error()
        log_event("bundle_error", error_type=type(exc).__name__)
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.get("/health/live", tags=["health"])
    def health_live() -> dict[str, str]:
        return {"status": "alive"}

    @app.get("/health/ready", tags=["health"])
    def health_ready() -> dict[str, str]:
        return {
            "status": "ready",
            "model_version": bundle.model_version,
            "artifact_sha256": bundle.artifact_sha256,
        }

    @app.get("/v1/model", tags=["model"])
    def model_metadata() -> dict[str, object]:
        return bundle.public_metadata()

    @app.get("/metrics", response_class=PlainTextResponse, tags=["monitoring"])
    def prometheus_metrics() -> str:
        return metrics.prometheus_text(bundle.model_version)

    def score_one(transaction: Transaction) -> PredictionResponse:
        started = time.perf_counter()
        request_id = str(uuid.uuid4())
        prediction = bundle.predict(transaction.model_inputs())
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        metrics.record(prediction.score, prediction.review_recommended)
        log_event(
            "prediction",
            request_id=request_id,
            model_version=bundle.model_version,
            review_recommended=prediction.review_recommended,
            score=round(prediction.score, 8),
            latency_ms=round(elapsed_ms, 3),
        )
        return PredictionResponse(
            request_id=request_id,
            transaction_id=transaction.transaction_id,
            model_version=bundle.model_version,
            score=prediction.score,
            threshold=bundle.threshold,
            review_recommended=prediction.review_recommended,
        )

    @app.post("/v1/predictions", response_model=PredictionResponse, tags=["predictions"])
    def predict(transaction: Transaction) -> PredictionResponse:
        return score_one(transaction)

    @app.post("/v1/predictions/batch", response_model=BatchPredictionResponse, tags=["predictions"])
    def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
        return BatchPredictionResponse(
            model_version=bundle.model_version,
            predictions=[score_one(transaction) for transaction in request.transactions],
        )

    return app


app = create_app()
