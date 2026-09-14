from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transaction_id: str | None = Field(default=None, max_length=128)
    Time: float = Field(ge=0.0)
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(ge=0.0)

    @model_validator(mode="after")
    def reject_non_finite_values(self) -> "Transaction":
        for field_name in ("Time", "Amount", *(f"V{i}" for i in range(1, 29))):
            if not math.isfinite(float(getattr(self, field_name))):
                raise ValueError(f"{field_name} must be finite")
        return self

    def model_inputs(self) -> dict[str, float]:
        values = self.model_dump(exclude={"transaction_id"})
        return {name: float(value) for name, value in values.items()}


class PredictionResponse(BaseModel):
    request_id: str
    transaction_id: str | None
    model_version: str
    score: float
    threshold: float
    review_recommended: bool


class DemoPredictionResponse(PredictionResponse):
    fixture_id: str


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transactions: list[Transaction] = Field(min_length=1, max_length=1000)


class BatchPredictionResponse(BaseModel):
    model_version: str
    predictions: list[PredictionResponse]
