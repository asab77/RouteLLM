from typing import Literal

from pydantic import BaseModel, field_validator

from adaptive_llm_gateway.models import InferenceRequest, InferenceResponse
from adaptive_llm_gateway.models.schemas import Identifier
from adaptive_llm_gateway.telemetry.contracts import TelemetrySummary
from adaptive_llm_gateway.evaluation.models import EvaluationSummary
from adaptive_llm_gateway.routing.features import RoutingCategory
from adaptive_llm_gateway.routing.policy import (
    RoutingDecisionReason,
    validate_quality_threshold,
)


class MetricsSummary(TelemetrySummary):
    """Public aggregate telemetry response; no ORM objects."""


class BenchmarkEvaluationSummary(EvaluationSummary):
    """Read-only aggregate evaluation; raw local paths are never exposed."""


class InferencePayload(InferenceRequest):
    """Reuse all domain validation while adding explicit model selection."""

    model_id: Identifier

    def to_domain(self) -> InferenceRequest:
        return InferenceRequest(**self.model_dump(exclude={"model_id"}))


class InferenceResult(InferenceResponse):
    request_id: str


class AdaptiveInferencePayload(InferenceRequest):
    """Public adaptive intent; the gateway owns artifact and candidate configuration."""

    category: RoutingCategory
    quality_threshold: float

    @field_validator("quality_threshold", mode="before")
    @classmethod
    def apply_routing_threshold_contract(cls, value: object) -> float:
        return validate_quality_threshold(value)

    def to_domain(self) -> InferenceRequest:
        return InferenceRequest(
            **self.model_dump(exclude={"category", "quality_threshold"})
        )


class PublicRoutingMetadata(BaseModel):
    selected_model_id: str
    threshold_satisfied: bool
    fallback_used: bool
    reason: RoutingDecisionReason


class AdaptiveInferenceResult(InferenceResponse):
    request_id: str
    routing: PublicRoutingMetadata


class PublicModel(BaseModel):
    """Intentionally omit pricing and provider-internal model names."""

    model_id: str
    provider: str
    context_window: int


class ModelList(BaseModel):
    models: list[PublicModel]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail
    request_id: str
