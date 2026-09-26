"""Adaptive routing composition that reuses the explicit inference service."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Sequence

from adaptive_llm_gateway.errors import ModelDisabledError, ProviderUnavailableError
from adaptive_llm_gateway.errors import UnsupportedPredictorCandidateError
from adaptive_llm_gateway.models import InferenceRequest, InferenceResponse, ModelConfig
from adaptive_llm_gateway.models.schemas import DomainModel, Identifier
from adaptive_llm_gateway.routing.features import RoutingCategory
from adaptive_llm_gateway.routing.policy import RoutingDecision
from adaptive_llm_gateway.routing.service import RoutingDecisionService

from .service import InferenceService


class AdaptiveInferenceResult(DomainModel):
    """Generated response plus the unchanged pre-generation routing decision."""

    response: InferenceResponse
    routing_decision: RoutingDecision


class AdaptiveInferenceService:
    """Select one model, then converge on the existing explicit execution path."""

    def __init__(
        self,
        inference_service: InferenceService,
        routing_service: RoutingDecisionService,
    ) -> None:
        self._inference_service = inference_service
        self._routing_service = routing_service

    @classmethod
    def from_trusted_artifact(
        cls,
        inference_service: InferenceService,
        artifact_directory: str | Path,
        *,
        candidate_model_ids: Sequence[Identifier | str] | None = None,
    ) -> AdaptiveInferenceService:
        """Explicitly load an application-owned artifact without building it."""
        # Keep sklearn and pickle loading outside ordinary explicit API imports.
        from adaptive_llm_gateway.routing.predictor import SklearnQualityPredictor

        predictor = SklearnQualityPredictor.from_trusted_artifact(
            Path(artifact_directory)
        )
        instance = cls(inference_service, RoutingDecisionService(predictor))
        if candidate_model_ids is not None:
            candidates = instance._resolve_candidates(candidate_model_ids)
            supported = set(predictor.metadata.known_candidate_ids)
            unsupported = sorted(
                candidate.model_id for candidate in candidates
                if candidate.model_id not in supported
            )
            if unsupported:
                raise UnsupportedPredictorCandidateError(
                    "configured candidate is not supported by the predictor artifact"
                )
        return instance

    async def generate(
        self,
        request: InferenceRequest,
        *,
        category: RoutingCategory | str | None,
        quality_threshold: float | Decimal,
        candidate_model_ids: Sequence[Identifier | str],
        structured_output_required: bool = False,
        request_id: str | None = None,
    ) -> AdaptiveInferenceResult:
        candidates = self._resolve_candidates(candidate_model_ids)
        decision = self._routing_service.route(
            request,
            candidates,
            quality_threshold,
            category_hint=category,
            structured_output_required=structured_output_required,
        )
        response = await self._inference_service.generate(
            decision.selected_model_id,
            request,
            request_id=request_id,
        )
        return AdaptiveInferenceResult(
            response=response,
            routing_decision=decision,
        )

    def _resolve_candidates(
        self,
        candidate_model_ids: Sequence[Identifier | str],
    ) -> tuple[ModelConfig, ...]:
        candidates: list[ModelConfig] = []
        for model_id in candidate_model_ids:
            model = self._inference_service.registry.get(str(model_id))
            if not model.enabled:
                raise ModelDisabledError(f"Model is disabled: {model.model_id!r}")
            if not self._inference_service.resolver.supports(model.provider):
                raise ProviderUnavailableError(
                    "No provider adapter is registered for an adaptive candidate"
                )
            candidates.append(model)
        return tuple(candidates)


__all__ = ["AdaptiveInferenceResult", "AdaptiveInferenceService"]
