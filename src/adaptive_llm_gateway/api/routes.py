from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Request

from adaptive_llm_gateway.application.service import InferenceService
from adaptive_llm_gateway.application.adaptive_config import AdaptiveRuntime
from adaptive_llm_gateway.telemetry.query import TelemetryQueryService
from adaptive_llm_gateway.evaluation.service import EvaluationService

from .dependencies import get_adaptive_runtime, get_evaluation_service, get_service
from .schemas import (
    AdaptiveInferencePayload,
    AdaptiveInferenceResult,
    ErrorResponse,
    HealthResponse,
    InferencePayload,
    InferenceResult,
    ModelList,
    MetricsSummary,
    BenchmarkEvaluationSummary,
    PublicModel,
    PublicRoutingMetadata,
)

router = APIRouter()
Service = Annotated[InferenceService, Depends(get_service)]
Evaluation = Annotated[EvaluationService, Depends(get_evaluation_service)]
Adaptive = Annotated[AdaptiveRuntime, Depends(get_adaptive_runtime)]


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


@router.post(
    "/v1/inference/adaptive",
    response_model=AdaptiveInferenceResult,
    tags=["inference"],
    responses={status: {"model": ErrorResponse} for status in (400, 403, 404, 422, 429, 502, 503, 504)},
)
async def adaptive_inference(
    payload: AdaptiveInferencePayload,
    request: Request,
    runtime: Adaptive,
) -> AdaptiveInferenceResult:
    result = await runtime.service.generate(
        payload.to_domain(),
        category=payload.category,
        quality_threshold=payload.quality_threshold,
        candidate_model_ids=runtime.candidate_model_ids,
        request_id=request.state.request_id,
    )
    decision = result.routing_decision
    return AdaptiveInferenceResult(
        **result.response.model_dump(),
        request_id=request.state.request_id,
        routing=PublicRoutingMetadata(
            selected_model_id=decision.selected_model_id,
            threshold_satisfied=decision.threshold_satisfied,
            fallback_used=decision.fallback_used,
            reason=decision.reason,
        ),
    )


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
