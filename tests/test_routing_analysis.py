from decimal import Decimal
from uuid import UUID

import pytest
from pydantic import ValidationError

from adaptive_llm_gateway.benchmarks.features import RequestFeatures
from adaptive_llm_gateway.routing.analysis import (
    ALWAYS_CHEAPEST,
    ALWAYS_STRONGEST,
    GEMINI,
    LUNA,
    NEMOTRON,
    ORACLE_CHEAPEST_ACCEPTABLE,
    RANDOM_SEEDED,
    RULE_BASED_V1,
    SONNET,
    AlwaysCheapestPolicy,
    CandidateOption,
    FixedCandidatePolicy,
    FrozenRoutingData,
    PolicyRequest,
    RandomSeededPolicy,
    RoutingDecision,
    RowObservation,
    RuleBasedV1Policy,
    _comparison,
    evaluate_decisions,
    oracle_cheapest_acceptable,
    resolve_strongest,
)


def features(category="qa", **updates):
    values = {
        "category": category,
        "prompt_characters": 100,
        "system_prompt_characters": 20,
        "approximate_input_tokens": 30,
        "contains_code": False,
        "requests_structured_output": False,
        "max_output_tokens": 64,
        "constraint_indicator_count": 1,
        "reasoning_indicator_count": 0,
    }
    values.update(updates)
    return RequestFeatures(**values)


def option(candidate_id, input_price="1", output_price="1", limit=100):
    return CandidateOption(
        candidate_id=candidate_id,
        input_cost_per_1m_tokens=input_price,
        output_cost_per_1m_tokens=output_price,
        effective_max_output_tokens=limit,
    )


def observation(task_id, candidate_id, *, acceptable=True, quality=1.0, cost="0.001",
                label_status="valid", category="qa", input_price="1", output_price="1",
                latency=100.0, difficulty="easy"):
    return RowObservation(
        task_id=task_id,
        candidate_id=candidate_id,
        features=features(category),
        analysis_difficulty=difficulty,
        option=option(candidate_id, input_price, output_price),
        label_status=label_status,
        acceptable=acceptable if label_status == "valid" else None,
        quality_score=quality if label_status == "valid" else None,
        realized_cost_usd=cost,
        latency_ms=latency,
    )


def request(task_id="task-1", category="qa", candidates=None, **feature_updates):
    return PolicyRequest(
        task_id=task_id,
        features=features(category, **feature_updates),
        candidates=tuple(candidates or (option("cheap", "0.1", "0.2"), option("expensive", "2", "10"))),
    )


def test_cheapest_uses_pre_generation_projected_cost_not_realized_cost():
    policy_request = request(candidates=(
        option("cheap", "0.1", "0.2", 200),
        option("expensive", "2", "10", 10),
    ))
    decision = AlwaysCheapestPolicy().select(policy_request)
    assert decision.policy == ALWAYS_CHEAPEST
    assert decision.candidate_id == "cheap"
    assert "realized_cost" not in PolicyRequest.model_fields
    assert "output_tokens" not in PolicyRequest.model_fields


def test_policy_request_rejects_label_difficulty_and_outcome_leakage():
    payload = request().model_dump()
    for forbidden in (
        "acceptable", "quality_score", "difficulty", "latency_ms", "realized_cost_usd",
        "output_tokens", "reasoning_tokens", "evaluation_status", "ground_truth",
    ):
        with pytest.raises(ValidationError):
            PolicyRequest.model_validate({**payload, forbidden: 1})


def test_strongest_resolves_by_rate_then_quality_cost_and_id():
    higher_rate = [
        observation("t1", "a", acceptable=False, quality=0.9),
        observation("t2", "a", acceptable=True, quality=0.9),
        observation("t1", "b", acceptable=True, quality=0.6),
        observation("t2", "b", acceptable=True, quality=0.6),
    ]
    assert resolve_strongest(higher_rate)[0] == "b"

    quality_tie_break = [
        observation("t1", "a", quality=0.8), observation("t1", "b", quality=0.9),
    ]
    assert resolve_strongest(quality_tie_break)[0] == "b"

    cost_tie_break = [
        observation("t1", "a", quality=0.9, input_price="0.1", output_price="0.2"),
        observation("t1", "b", quality=0.9, input_price="2", output_price="10"),
    ]
    assert resolve_strongest(cost_tie_break)[0] == "a"

    stable_id_tie_break = [
        observation("t1", "a", quality=0.9), observation("t1", "b", quality=0.9),
    ]
    assert resolve_strongest(stable_id_tie_break)[0] == "a"


def test_fixed_strongest_policy_is_explicitly_resolved_before_selection():
    decision = FixedCandidatePolicy("expensive").select(request())
    assert decision.policy == ALWAYS_STRONGEST
    assert decision.candidate_id == "expensive"


def test_random_seed_is_deterministic_per_request_and_uses_no_outcomes():
    policy = RandomSeededPolicy(seed=42)
    first = [policy.select(request(f"task-{index}")) for index in range(20)]
    second = [policy.select(request(f"task-{index}")) for index in range(20)]
    assert first == second
    assert all(item.policy == RANDOM_SEEDED for item in first)
    assert {item.candidate_id for item in first} == {"cheap", "expensive"}


@pytest.mark.parametrize(("category", "updates", "expected"), [
    ("coding", {}, SONNET),
    ("classification", {"contains_code": True}, SONNET),
    ("reasoning", {}, SONNET),
    ("classification", {"reasoning_indicator_count": 3}, SONNET),
    ("json", {"requests_structured_output": True}, LUNA),
    ("qa", {}, NEMOTRON),
    ("summarization", {}, LUNA),
])
def test_rule_based_v1_uses_only_request_visible_features(category, updates, expected):
    candidates = tuple(option(candidate) for candidate in (NEMOTRON, LUNA, GEMINI, SONNET))
    decision = RuleBasedV1Policy().select(request(
        category=category, candidates=candidates, **updates))
    assert decision.policy == RULE_BASED_V1
    assert decision.candidate_id == expected


def frozen_data(observations):
    task_ids = sorted({row.task_id for row in observations})
    requests = tuple(PolicyRequest(
        task_id=task_id,
        features=next(row.features for row in observations if row.task_id == task_id),
        candidates=tuple(row.option for row in observations if row.task_id == task_id),
    ) for task_id in task_ids)
    return FrozenRoutingData(
        run_id=UUID("61707aba-5ab2-4c16-8ec3-eed74555d69c"),
        dataset_sha256="dataset",
        protocol_sha256="protocol",
        observations=tuple(observations),
        requests=requests,
    )


def test_oracle_selects_cheapest_acceptable_with_stable_tie_break():
    rows = [
        observation("t1", "a", cost="0.001"),
        observation("t1", "b", cost="0.001"),
    ]
    decisions, states = oracle_cheapest_acceptable(frozen_data(rows))
    assert decisions[0].candidate_id == "a"
    assert states == {"fully_observed": 1}


def test_oracle_preserves_partial_observation_and_no_acceptable_state():
    rows = [
        observation("t1", "a", label_status="missing"),
        observation("t1", "b", acceptable=True, cost="0.002"),
        observation("t2", "a", acceptable=False, quality=0.2),
        observation("t2", "b", acceptable=False, quality=0.3),
    ]
    decisions, states = oracle_cheapest_acceptable(frozen_data(rows))
    by_task = {item.task_id: item for item in decisions}
    assert by_task["t1"].candidate_id == "b"
    assert "partially_observed" in by_task["t1"].reason
    assert by_task["t2"].candidate_id is None
    assert states == {"partially_observed": 1, "no_acceptable_candidate": 1}


def test_missing_selected_label_is_excluded_from_quality_denominators_and_cost_retained():
    rows = [
        observation("t1", "a", acceptable=True, quality=0.8, cost="1", latency=10),
        observation("t2", "a", label_status="missing", cost="2", latency=30),
    ]
    decisions = [
        RoutingDecision(policy="TEST", task_id="t1", candidate_id="a", reason="test"),
        RoutingDecision(policy="TEST", task_id="t2", candidate_id="a", reason="test"),
    ]
    summary, _ = evaluate_decisions(rows, decisions)
    assert summary["total_requests"] == summary["selected_rows"] == 2
    assert summary["valid_selected_labels"] == 1
    assert summary["missing_selected_labels"] == 1
    assert summary["valid_label_coverage"] == 0.5
    assert summary["acceptable"] == 1 and summary["unacceptable"] == 0
    assert summary["acceptable_rate_among_valid"] == 1
    assert summary["mean_quality_among_valid"] == 0.8
    assert summary["total_realized_cost_usd"] == "3"
    assert summary["average_realized_cost_per_request_usd"] == "1.5"
    assert summary["mean_latency_ms"] == 20


def test_no_acceptable_decision_keeps_request_denominator_and_zero_request_cost():
    rows = [observation("t1", "a", acceptable=False, cost="1")]
    decisions = [RoutingDecision(
        policy=ORACLE_CHEAPEST_ACCEPTABLE, task_id="t1",
        candidate_id=None, reason="no_acceptable_candidate")]
    summary, _ = evaluate_decisions(rows, decisions)
    assert summary["total_requests"] == 1 and summary["selected_rows"] == 0
    assert summary["valid_label_coverage"] == 0
    assert summary["total_realized_cost_usd"] == "0"
    assert summary["model_distribution"]["no_acceptable_candidate"] == {
        "count": 1, "percentage_of_requests": 1.0}


def test_comparison_uses_strongest_cost_and_acceptable_rate_denominators():
    summaries = {
        ALWAYS_STRONGEST: {
            "total_realized_cost_usd": "10", "acceptable_rate_among_valid": 0.8},
        ALWAYS_CHEAPEST: {
            "total_realized_cost_usd": "4", "acceptable_rate_among_valid": 0.6},
    }
    compared = _comparison(summaries)[ALWAYS_CHEAPEST]
    assert compared["cost_reduction_vs_always_strongest"] == 0.6
    assert compared["acceptable_rate_difference_vs_always_strongest"] == pytest.approx(-0.2)
    assert compared["acceptable_rate_retention_vs_always_strongest"] == pytest.approx(0.75)


def test_evaluation_output_is_deterministic_when_inputs_are_reordered():
    rows = [
        observation("t2", "a", acceptable=False, quality=0.2, cost="2"),
        observation("t1", "a", acceptable=True, quality=0.9, cost="1"),
    ]
    decisions = [
        RoutingDecision(policy="TEST", task_id="t2", candidate_id="a", reason="test"),
        RoutingDecision(policy="TEST", task_id="t1", candidate_id="a", reason="test"),
    ]
    first = evaluate_decisions(rows, decisions)
    second = evaluate_decisions(reversed(rows), reversed(decisions))
    assert first == second
