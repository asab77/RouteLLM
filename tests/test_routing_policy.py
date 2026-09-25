import ast
import inspect
import itertools
from decimal import Decimal

import pytest
from pydantic import ValidationError

from adaptive_llm_gateway.errors import (
    DuplicateCandidatePredictionError,
    InvalidQualityThresholdError,
    NoEligibleCandidatesError,
)
from adaptive_llm_gateway.models import ModelConfig
from adaptive_llm_gateway.pricing import calculate_projected_cost
from adaptive_llm_gateway.routing.policy import (
    CandidateEligibilityFilter,
    CandidatePrediction,
    CostAwareRoutingPolicy,
    ModelAcceptabilityPrediction,
    QualityPredictor,
    RoutingDecisionReason,
    RoutingRequestFeatures,
)


def prediction(model_id: str, probability: float, cost: str) -> CandidatePrediction:
    return CandidatePrediction(
        model_id=model_id,
        predicted_acceptability=probability,
        projected_cost_usd=cost,
    )


def features(category: str | None = None) -> RoutingRequestFeatures:
    return RoutingRequestFeatures(
        category=category,
        prompt_characters=120,
        system_prompt_characters=10,
        approximate_input_tokens=31,
        contains_code=False,
        requests_structured_output=False,
        max_output_tokens=64,
        constraint_indicator_count=2,
        reasoning_indicator_count=1,
    )


def model(model_id: str, provider: str = "provider-x") -> ModelConfig:
    return ModelConfig(
        model_id=model_id,
        provider=provider,
        provider_model_name=f"upstream/{model_id}",
        input_cost_per_1m_tokens="0.123456789123456789",
        output_cost_per_1m_tokens="0.987654321987654321",
        context_window=8192,
    )


def test_cheapest_qualifying_candidate_is_selected():
    decision = CostAwareRoutingPolicy().route((
        prediction("lower-cost", 0.81, "0.0002"),
        prediction("higher-cost", 0.90, "0.0008"),
    ), 0.8)
    assert decision.selected_model_id == "lower-cost"
    assert decision.reason is RoutingDecisionReason.QUALITY_THRESHOLD_MET


def test_higher_probability_does_not_override_cost_after_threshold():
    decision = CostAwareRoutingPolicy().route((
        prediction("cheap", 0.71, "0.01"),
        prediction("expensive", 0.99, "9"),
    ), 0.7)
    assert decision.selected_model_id == "cheap"


def test_fallback_chooses_highest_probability():
    decision = CostAwareRoutingPolicy().route((
        prediction("cheap", 0.6, "0.01"),
        prediction("best", 0.7, "2"),
    ), 0.8)
    assert decision.selected_model_id == "best"
    assert decision.reason is RoutingDecisionReason.NO_MODEL_MET_THRESHOLD_FALLBACK


def test_fallback_probability_tie_chooses_lower_cost():
    decision = CostAwareRoutingPolicy().route((
        prediction("expensive", 0.7, "2"),
        prediction("cheap", 0.7, "0.01"),
    ), 0.8)
    assert decision.selected_model_id == "cheap"


@pytest.mark.parametrize("threshold", [0.5, 0.8])
def test_final_tie_chooses_stable_model_id(threshold):
    probability = 0.8 if threshold == 0.5 else 0.7
    decision = CostAwareRoutingPolicy().route((
        prediction("z-model", probability, "1"),
        prediction("a-model", probability, "1"),
    ), threshold)
    assert decision.selected_model_id == "a-model"


def test_threshold_equality_qualifies():
    decision = CostAwareRoutingPolicy().route(
        (prediction("boundary", 0.75, "1"),), Decimal("0.75")
    )
    assert decision.threshold_satisfied is True
    assert decision.fallback_used is False


def test_empty_candidates_raise_typed_error():
    with pytest.raises(NoEligibleCandidatesError):
        CostAwareRoutingPolicy().route((), 0.8)


@pytest.mark.parametrize("threshold", [-0.01, 1.01, float("nan"), float("inf"), True, "0.8"])
def test_invalid_threshold_is_rejected(threshold):
    with pytest.raises(InvalidQualityThresholdError):
        CostAwareRoutingPolicy().route((prediction("model", 0.5, "1"),), threshold)


@pytest.mark.parametrize("probability", [-0.01, 1.01, float("nan"), float("inf"), True, "0.8"])
def test_invalid_probability_is_rejected(probability):
    with pytest.raises(ValidationError):
        prediction("model", probability, "1")


@pytest.mark.parametrize("cost", ["-0.000001", "NaN", "Infinity"])
def test_invalid_projected_cost_is_rejected(cost):
    with pytest.raises(ValidationError):
        prediction("model", 0.8, cost)


def test_arbitrary_candidate_count_is_supported():
    candidates = tuple(
        prediction(f"model-{index:03}", 0.9, str(100 - index))
        for index in range(100)
    )
    decision = CostAwareRoutingPolicy().route(candidates, 0.5)
    assert decision.selected_model_id == "model-099"
    assert decision.eligible_candidate_count == 100


def test_input_order_does_not_change_selection():
    candidates = (
        prediction("c", 0.8, "1"),
        prediction("a", 0.8, "1"),
        prediction("b", 0.9, "2"),
    )
    selections = {
        CostAwareRoutingPolicy().route(order, 0.75).selected_model_id
        for order in itertools.permutations(candidates)
    }
    assert selections == {"a"}


def test_decimal_cost_ordering_is_exact():
    decision = CostAwareRoutingPolicy().route((
        prediction("slightly-more", 0.9, "0.1000000000000000002"),
        prediction("slightly-less", 0.9, "0.1000000000000000001"),
    ), 0.8)
    assert decision.selected_model_id == "slightly-less"
    assert decision.selected_projected_cost_usd == Decimal("0.1000000000000000001")


def test_decision_reports_selection_metadata():
    decision = CostAwareRoutingPolicy().route((
        prediction("one", 0.9, "0.1"),
        prediction("two", 0.2, "0.2"),
        prediction("three", 0.8, "0.3"),
    ), 0.8)
    assert decision.model_dump() == {
        "selected_model_id": "one",
        "selected_predicted_acceptability": 0.9,
        "selected_projected_cost_usd": Decimal("0.1"),
        "quality_threshold": 0.8,
        "threshold_satisfied": True,
        "fallback_used": False,
        "eligible_candidate_count": 3,
        "qualifying_candidate_count": 2,
        "reason": RoutingDecisionReason.QUALITY_THRESHOLD_MET,
    }


def test_fallback_decision_reports_state_and_zero_qualifiers():
    decision = CostAwareRoutingPolicy().route(
        (prediction("one", 0.4, "0.1"),), 0.8
    )
    assert decision.threshold_satisfied is False
    assert decision.fallback_used is True
    assert decision.qualifying_candidate_count == 0


def test_duplicate_candidate_ids_are_rejected():
    with pytest.raises(DuplicateCandidatePredictionError):
        CostAwareRoutingPolicy().route((
            prediction("same", 0.8, "1"),
            prediction("same", 0.9, "2"),
        ), 0.5)


def test_provider_names_do_not_affect_policy():
    candidates = (prediction("generic-a", 0.9, "1"), prediction("generic-b", 0.9, "2"))
    assert CostAwareRoutingPolicy().route(candidates, 0.5).selected_model_id == "generic-a"
    assert "provider" not in CandidatePrediction.model_fields


def test_current_experimental_model_names_are_not_required():
    decision = CostAwareRoutingPolicy().route((
        prediction("local-model-alpha", 0.8, "1"),
        prediction("future-model-beta", 0.9, "2"),
    ), 0.7)
    assert decision.selected_model_id == "local-model-alpha"


def test_quality_predictor_is_replaceable_with_fake():
    class FakePredictor:
        def predict(self, request_features, candidates):
            return tuple(ModelAcceptabilityPrediction(
                model_id=item.model_id, predicted_acceptability=0.9
            ) for item in candidates)

    predictor: QualityPredictor = FakePredictor()
    outputs = predictor.predict(features("qa"), (model("alpha"), model("beta")))
    assert isinstance(predictor, QualityPredictor)
    assert [item.model_id for item in outputs] == ["alpha", "beta"]


def test_eligibility_boundary_is_replaceable_with_fake():
    class EnabledOnly:
        def eligible_candidates(self, request_features, candidates):
            return tuple(item for item in candidates if item.enabled)

    eligibility: CandidateEligibilityFilter = EnabledOnly()
    disabled = model("disabled").model_copy(update={"enabled": False})
    assert [item.model_id for item in eligibility.eligible_candidates(
        features(), (disabled, model("enabled"))
    )] == ["enabled"]


def test_category_boundary_is_optional_and_validated():
    assert features().category is None
    assert features("  reasoning  ").category == "reasoning"
    with pytest.raises(ValidationError):
        features("   ")


def test_projected_cost_reuses_exact_canonical_pricing_formula():
    configured = model("exact")
    result = calculate_projected_cost(
        approximate_input_tokens=31,
        effective_max_output_tokens=64,
        model=configured,
    )
    expected = (
        Decimal(31) * configured.input_cost_per_1m_tokens
        + Decimal(64) * configured.output_cost_per_1m_tokens
    ) / Decimal(1_000_000)
    assert result == expected


def test_production_policy_has_no_experiment_or_provider_dependencies():
    import adaptive_llm_gateway.routing.policy as policy_module

    tree = ast.parse(inspect.getsource(policy_module))
    imports = {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert not any("benchmark" in name or "provider" in name for name in imports)
    source = inspect.getsource(policy_module).lower()
    assert "foundation" not in source
    assert "benchmark-results" not in source
    assert "scikit" not in source
