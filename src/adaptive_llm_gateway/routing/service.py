"""Thin provider-independent composition of production routing components."""

from __future__ import annotations

from collections import Counter
from decimal import Decimal
from typing import Sequence

from adaptive_llm_gateway.errors import (
    DuplicateCandidatePredictionError,
    PredictorInputCompatibilityError,
)
from adaptive_llm_gateway.models import InferenceRequest, ModelConfig
from adaptive_llm_gateway.pricing import calculate_projected_cost

from .features import ProductionRequestFeatureExtractor, RoutingCategory, RoutingRequestFeatures
from .policy import (
    CandidateEligibilityFilter,
    CandidatePrediction,
    CostAwareRoutingPolicy,
    QualityPredictor,
    RoutingDecision,
)
from .quality_features import resolve_effective_output_allowance


class RoutingDecisionService:
    """Extract, predict, price, and delegate selection without executing a model."""

    def __init__(
        self,
        predictor: QualityPredictor,
        *,
        feature_extractor: ProductionRequestFeatureExtractor | None = None,
        policy: CostAwareRoutingPolicy | None = None,
        eligibility_filter: CandidateEligibilityFilter | None = None,
    ) -> None:
        self._predictor = predictor
        self._feature_extractor = feature_extractor or ProductionRequestFeatureExtractor()
        self._policy = policy or CostAwareRoutingPolicy()
        self._eligibility_filter = eligibility_filter

    def route(
        self,
        request: InferenceRequest,
        candidates: Sequence[ModelConfig],
        quality_threshold: float | Decimal,
        *,
        category_hint: RoutingCategory | str | None = None,
        structured_output_required: bool = False,
    ) -> RoutingDecision:
        features = self._feature_extractor.extract(
            request,
            category_hint=category_hint,
            structured_output_required=structured_output_required,
        )
        return self.route_features(features, candidates, quality_threshold)

    def route_features(
        self,
        request_features: RoutingRequestFeatures,
        candidates: Sequence[ModelConfig],
        quality_threshold: float | Decimal,
    ) -> RoutingDecision:
        eligible = tuple(candidates)
        if self._eligibility_filter is not None:
            eligible = tuple(
                self._eligibility_filter.eligible_candidates(request_features, eligible)
            )
        eligible_ids = [candidate.model_id for candidate in eligible]
        duplicate_eligible = sorted(
            model_id for model_id, count in Counter(eligible_ids).items() if count > 1
        )
        if duplicate_eligible:
            raise DuplicateCandidatePredictionError(
                "eligible candidate model identifiers must be unique"
            )
        predictions = tuple(self._predictor.predict(request_features, eligible))
        expected_ids = eligible_ids
        predicted_ids = [prediction.model_id for prediction in predictions]
        duplicate_ids = sorted(
            model_id for model_id, count in Counter(predicted_ids).items() if count > 1
        )
        if duplicate_ids:
            raise DuplicateCandidatePredictionError(
                "predictor returned duplicate candidate identifiers: "
                + ", ".join(duplicate_ids)
            )
        if set(predicted_ids) != set(expected_ids) or len(predicted_ids) != len(expected_ids):
            raise PredictorInputCompatibilityError(
                "predictor output must correspond exactly to eligible candidates"
            )
        prediction_by_id = {prediction.model_id: prediction for prediction in predictions}
        priced = tuple(
            CandidatePrediction(
                model_id=candidate.model_id,
                predicted_acceptability=prediction_by_id[
                    candidate.model_id
                ].predicted_acceptability,
                projected_cost_usd=calculate_projected_cost(
                    approximate_input_tokens=request_features.approximate_input_tokens,
                    effective_max_output_tokens=resolve_effective_output_allowance(
                        request_features, candidate
                    ),
                    model=candidate,
                ),
            )
            for candidate in eligible
        )
        return self._policy.route(priced, quality_threshold)


__all__ = ["RoutingDecisionService"]
