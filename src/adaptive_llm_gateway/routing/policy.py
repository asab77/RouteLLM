"""Provider-independent, pre-generation cost-aware routing policy."""

from __future__ import annotations

import math
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Iterable, Protocol, Sequence, runtime_checkable

from pydantic import Field, field_validator

from adaptive_llm_gateway.errors import (
    DuplicateCandidatePredictionError,
    InvalidQualityThresholdError,
    NoEligibleCandidatesError,
)
from adaptive_llm_gateway.models import ModelConfig
from adaptive_llm_gateway.models.schemas import DomainModel, Identifier, Money
from adaptive_llm_gateway.routing.features import RoutingRequestFeatures

Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


def _validated_probability(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise ValueError(f"{field_name} must be a number between 0 and 1")
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError(f"{field_name} must be a finite number between 0 and 1")
    return result


class ModelAcceptabilityPrediction(DomainModel):
    """One predictor output, separate from model pricing and policy."""

    model_id: Identifier
    predicted_acceptability: Probability

    @field_validator("predicted_acceptability", mode="before")
    @classmethod
    def validate_probability(cls, value: object) -> float:
        return _validated_probability(value, field_name="predicted_acceptability")


class CandidatePrediction(ModelAcceptabilityPrediction):
    """All pre-generation values needed by the routing decision layer."""

    projected_cost_usd: Money


class RoutingDecisionReason(StrEnum):
    QUALITY_THRESHOLD_MET = "quality_threshold_met"
    NO_MODEL_MET_THRESHOLD_FALLBACK = "no_model_met_threshold_fallback"


class RoutingDecision(DomainModel):
    """Privacy-safe decision metadata suitable for future telemetry."""

    selected_model_id: Identifier
    selected_predicted_acceptability: Probability
    selected_projected_cost_usd: Money
    quality_threshold: Probability
    threshold_satisfied: bool = Field(strict=True)
    fallback_used: bool = Field(strict=True)
    eligible_candidate_count: int = Field(gt=0, strict=True)
    qualifying_candidate_count: int = Field(ge=0, strict=True)
    reason: RoutingDecisionReason


@runtime_checkable
class QualityPredictor(Protocol):
    """Replaceable batch prediction boundary; implementations own artifacts."""

    def predict(
        self,
        request_features: RoutingRequestFeatures,
        candidates: Sequence[ModelConfig],
    ) -> Sequence[ModelAcceptabilityPrediction]: ...


@runtime_checkable
class CandidateEligibilityFilter(Protocol):
    """Future boundary for registry/capability filtering before policy use."""

    def eligible_candidates(
        self,
        request_features: RoutingRequestFeatures,
        candidates: Sequence[ModelConfig],
    ) -> Sequence[ModelConfig]: ...


class CostAwareRoutingPolicy:
    """Choose the cheapest qualifying candidate, with deterministic fallback."""

    def route(
        self,
        candidates: Iterable[CandidatePrediction],
        quality_threshold: float | Decimal,
    ) -> RoutingDecision:
        candidate_list = tuple(candidates)
        if not candidate_list:
            raise NoEligibleCandidatesError("at least one eligible candidate is required")

        try:
            threshold = _validated_probability(
                quality_threshold, field_name="quality_threshold"
            )
        except ValueError as exc:
            raise InvalidQualityThresholdError(str(exc)) from exc

        candidate_ids = [candidate.model_id for candidate in candidate_list]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise DuplicateCandidatePredictionError(
                "eligible candidate model identifiers must be unique"
            )

        qualifying = tuple(
            candidate for candidate in candidate_list
            if candidate.predicted_acceptability >= threshold
        )
        if qualifying:
            selected = min(
                qualifying,
                key=lambda candidate: (candidate.projected_cost_usd, candidate.model_id),
            )
            reason = RoutingDecisionReason.QUALITY_THRESHOLD_MET
            fallback_used = False
        else:
            selected = min(
                candidate_list,
                key=lambda candidate: (
                    -candidate.predicted_acceptability,
                    candidate.projected_cost_usd,
                    candidate.model_id,
                ),
            )
            reason = RoutingDecisionReason.NO_MODEL_MET_THRESHOLD_FALLBACK
            fallback_used = True

        return RoutingDecision(
            selected_model_id=selected.model_id,
            selected_predicted_acceptability=selected.predicted_acceptability,
            selected_projected_cost_usd=selected.projected_cost_usd,
            quality_threshold=threshold,
            threshold_satisfied=not fallback_used,
            fallback_used=fallback_used,
            eligible_candidate_count=len(candidate_list),
            qualifying_candidate_count=len(qualifying),
            reason=reason,
        )


__all__ = [
    "CandidateEligibilityFilter",
    "CandidatePrediction",
    "CostAwareRoutingPolicy",
    "ModelAcceptabilityPrediction",
    "QualityPredictor",
    "RoutingDecision",
    "RoutingDecisionReason",
    "RoutingRequestFeatures",
]
