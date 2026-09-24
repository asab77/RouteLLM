from typing import Literal

from pydantic import BaseModel

from adaptive_llm_gateway.models import InferenceRequest, InferenceResponse
from adaptive_llm_gateway.models.schemas import Identifier
from adaptive_llm_gateway.telemetry.contracts import TelemetrySummary
from adaptive_llm_gateway.evaluation.models import EvaluationSummary


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
