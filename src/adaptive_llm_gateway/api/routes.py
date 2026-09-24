from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Request

from adaptive_llm_gateway.application.service import InferenceService
from adaptive_llm_gateway.telemetry.query import TelemetryQueryService
from adaptive_llm_gateway.evaluation.service import EvaluationService

from .dependencies import get_evaluation_service, get_service
from .schemas import (
    ErrorResponse,
    HealthResponse,
    InferencePayload,
    InferenceResult,
    ModelList,
    MetricsSummary,
    BenchmarkEvaluationSummary,
    PublicModel,
)

router = APIRouter()
Service = Annotated[InferenceService, Depends(get_service)]
Evaluation = Annotated[EvaluationService, Depends(get_evaluation_service)]


@router.get("/health", response_model=HealthResponse, tags=["health"])
async def health() -> HealthResponse:
    return HealthResponse()


@router.get("/v1/models", response_model=ModelList, tags=["models"])
async def list_models(service: Service) -> ModelList:
    return ModelList(models=[
        PublicModel(model_id=model.model_id, provider=model.provider,
                    context_window=model.context_window)
        for model in service.list_models()
    ])


@router.post(
    "/v1/inference", response_model=InferenceResult, tags=["inference"],
    responses={status: {"model": ErrorResponse} for status in (400, 403, 404, 422, 429, 502, 503, 504)},
    openapi_extra={"parameters": [{
        "name": "X-Request-ID", "in": "header", "required": False,
        "description": "Optional correlation label; generated UUID4 when omitted.",
        "schema": {"type": "string", "maxLength": 128,
                   "pattern": "^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"},
    }]},
)
async def inference(payload: InferencePayload, request: Request, service: Service) -> InferenceResult:
    result = await service.generate(payload.model_id, payload.to_domain(), request_id=request.state.request_id)
    return InferenceResult(**result.model_dump(), request_id=request.state.request_id)


@router.get("/v1/metrics/summary", response_model=MetricsSummary, tags=["metrics"],
            responses={503: {"model": ErrorResponse}})
async def metrics_summary(service: Service) -> MetricsSummary:
    summary = await TelemetryQueryService(service.telemetry, service.telemetry_timeout).summary()
    return MetricsSummary(**summary.model_dump())


@router.get("/v1/benchmarks/{run_id}/summary", response_model=BenchmarkEvaluationSummary,
            tags=["benchmarks"], responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}})
async def benchmark_summary(run_id: UUID, evaluation: Evaluation) -> BenchmarkEvaluationSummary:
    summary = await evaluation.summary(run_id)
    return BenchmarkEvaluationSummary(**summary.model_dump())
