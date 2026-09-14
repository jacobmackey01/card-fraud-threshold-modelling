from __future__ import annotations

import logging
import os
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from service.demo import DemoFixture, load_demo_fixture
from service.model import BundleError, ModelBundle, PROJECT_ROOT
from service.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    DemoPredictionResponse,
    PredictionResponse,
    Transaction,
)
from service.telemetry import PredictionMetrics, log_event


logging.basicConfig(level=logging.INFO, format="%(message)s")


def _enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes"}


def _deployment_metadata(bundle: ModelBundle, fixture: DemoFixture | None) -> dict[str, object]:
    return {
        "rehearsal": fixture is not None,
        "input_policy": "checked_in_synthetic_fixture_only" if fixture is not None else "validated_feature_vector",
        "git_commit": os.getenv("DEPLOYED_GIT_COMMIT"),
        "container_digest": os.getenv("DEPLOYED_IMAGE_DIGEST"),
        "model_release": bundle.model_version,
        "model_file_sha256": bundle.artifact_sha256,
        "decision_threshold": bundle.threshold,
        "synthetic_fixture_version": fixture.fixture_id if fixture is not None else None,
        "cloud_run_revision": os.getenv("K_REVISION"),
    }


def create_app(
    project_root: Path = PROJECT_ROOT,
    *,
    synthetic_demo_only: bool | None = None,
) -> FastAPI:
    bundle = ModelBundle.load(project_root)
    demo_only = _enabled("SYNTHETIC_DEMO_ONLY") if synthetic_demo_only is None else synthetic_demo_only
    fixture = load_demo_fixture(project_root) if demo_only else None
    deployment_metadata = _deployment_metadata(bundle, fixture)
    metrics = PredictionMetrics()
    app = FastAPI(
        title="Fraud Review Recommendation API",
        version="1.0.0",
        description=(
            "Synthetic deployment rehearsal for a versioned portfolio model. "
            "The public deployment scores one checked-in synthetic fixture only."
            if demo_only
            else "Scores anonymised ULB credit-card transactions with a versioned portfolio model. "
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
    def health_ready() -> dict[str, object]:
        response: dict[str, object] = {
            "status": "ready",
            "model_version": bundle.model_version,
            "artifact_sha256": bundle.artifact_sha256,
            "deployment": deployment_metadata,
        }
        return response

    @app.get("/v1/model", tags=["model"])
    def model_metadata() -> dict[str, object]:
        response = bundle.public_metadata()
        response["deployment"] = deployment_metadata
        return response

    @app.get("/metrics", response_class=PlainTextResponse, tags=["monitoring"])
    def prometheus_metrics() -> str:
        return metrics.prometheus_text(bundle.model_version)

    def score_one(transaction: Transaction, *, fixture_id: str | None = None) -> PredictionResponse:
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
            fixture_id=fixture_id,
            cloud_run_revision=deployment_metadata["cloud_run_revision"],
        )
        return PredictionResponse(
            request_id=request_id,
            transaction_id=transaction.transaction_id,
            model_version=bundle.model_version,
            score=prediction.score,
            threshold=bundle.threshold,
            review_recommended=prediction.review_recommended,
        )

    if fixture is not None:

        @app.post(
            "/v1/demo-prediction",
            response_model=DemoPredictionResponse,
            tags=["synthetic deployment rehearsal"],
        )
        async def demo_prediction(request: Request) -> DemoPredictionResponse:
            if await request.body():
                raise HTTPException(
                    status_code=400,
                    detail="The synthetic demo endpoint does not accept a request body.",
                )
            prediction = score_one(fixture.transaction, fixture_id=fixture.fixture_id)
            return DemoPredictionResponse(**prediction.model_dump(), fixture_id=fixture.fixture_id)

    else:

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
