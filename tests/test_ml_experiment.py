from collections import Counter
from decimal import Decimal
from uuid import UUID

import pytest
from pydantic import ValidationError

from adaptive_llm_gateway.routing.analysis import (
    GEMINI, LUNA, NEMOTRON, SONNET, CandidateOption, FrozenRoutingData,
    PolicyRequest, RoutingDecision, RowObservation, evaluate_decisions,
)
from adaptive_llm_gateway.routing.ml_experiment import (
    LOGREG_BALANCED,
    LOGREG_UNWEIGHTED,
    THRESHOLDS,
    _json,
    _predictive_metrics,
    fold_local_strongest_decisions,
    generate_oof_predictions,
    select_cost_aware_candidate,
)
from adaptive_llm_gateway.routing.ml_features import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    FORBIDDEN_PREDICTIVE_FIELDS,
    NUMERIC_FEATURES,
    PREDICTIVE_FEATURES,
    MLExperimentDataset,
    MLExperimentRow,
    build_outer_folds,
    build_pipeline,
    feature_matrix,
)
from adaptive_llm_gateway.benchmarks.features import RequestFeatures
from adaptive_llm_gateway.routing.ml_features import FoldAssignment

CATEGORIES = ("qa", "classification", "extraction", "json", "reasoning", "coding", "summarization")
CANDIDATES = (NEMOTRON, LUNA, GEMINI, SONNET)


def governed_features(category, candidate, task_index):
    candidate_index = CANDIDATES.index(candidate)
    return {
        "category": category,
        "candidate_id": candidate,
        "upstream_provider_pin": ("nvidia", "openai", "google", "anthropic")[candidate_index],
        "reasoning_effort": "none" if candidate == NEMOTRON else "low",
        "prompt_characters": float(100 + task_index),
        "approximate_input_tokens": float(20 + task_index),
        "requested_max_output_tokens": 160.0,
        "constraint_indicator_count": float(task_index % 4),
        "reasoning_indicator_count": float(task_index % 3),
        "configured_input_cost_per_1m_tokens": (0.05, 0.1, 0.5, 2.0)[candidate_index],
        "configured_output_cost_per_1m_tokens": (0.15, 0.5, 3.0, 10.0)[candidate_index],
        "context_window": float((256000, 1050000, 1000000, 1000000)[candidate_index]),
        "effective_max_output_tokens": 256.0 if candidate == GEMINI and category == "reasoning" else 160.0,
        "contains_code": category == "coding",
        "requests_structured_output": category in {"coding", "json", "extraction"},
        "supports_temperature": candidate != SONNET,
    }


def synthetic_dataset():
    rows = []
    missing_pairs = {(f"{category}-0", GEMINI) for category in CATEGORIES}
    missing_pairs.add(("reasoning-1", SONNET))
    for category_index, category in enumerate(CATEGORIES):
        for task_index in range(8):
            task_id = f"{category}-{task_index}"
            for candidate_index, candidate in enumerate(CANDIDATES):
                missing = (task_id, candidate) in missing_pairs
                acceptable = ((category_index + task_index + candidate_index * 2) % 5) < 3
                rows.append(MLExperimentRow(
                    task_id=task_id,
                    candidate_id=candidate,
                    category=category,
                    features=governed_features(category, candidate, task_index),
                    label_status="missing" if missing else "valid",
                    acceptable=None if missing else acceptable,
                    quality_score=None if missing else (1.0 if acceptable else 0.2),
                    analysis_difficulty=("easy", "medium", "hard")[task_index % 3],
                    realized_cost_usd=Decimal("0.01") * (candidate_index + 1),
                    latency_ms=float(100 * (candidate_index + 1)),
                ))
    return MLExperimentDataset(
        run_id=UUID("61707aba-5ab2-4c16-8ec3-eed74555d69c"),
        rows=tuple(rows),
    )


def test_governed_feature_matrix_contains_only_predeclared_fields():
    dataset = synthetic_dataset()
    assert set(dataset.rows[0].features) == set(PREDICTIVE_FEATURES)
    assert not set(PREDICTIVE_FEATURES) & FORBIDDEN_PREDICTIVE_FIELDS
    assert "task_id" not in PREDICTIVE_FEATURES
    assert "analysis_difficulty" not in PREDICTIVE_FEATURES
    assert "acceptable" not in PREDICTIVE_FEATURES
    assert feature_matrix(dataset.rows[:2]).shape == (2, len(PREDICTIVE_FEATURES))


@pytest.mark.parametrize("forbidden", sorted(FORBIDDEN_PREDICTIVE_FIELDS))
def test_predictive_row_rejects_every_forbidden_extra_feature(forbidden):
    dataset = synthetic_dataset()
    payload = dataset.rows[0].model_dump()
    payload["features"] = {**payload["features"], forbidden: 1}
    with pytest.raises(ValidationError):
        MLExperimentRow.model_validate(payload)


def test_outer_folds_are_grouped_balanced_complete_and_deterministic():
    dataset = synthetic_dataset()
    first = build_outer_folds(dataset)
    second = build_outer_folds(dataset)
    assert first == second
    assert len(first) == 4
    assert all(len(fold.task_ids) == 14 for fold in first)
    assert all(fold.category_counts == {category: 2 for category in sorted(CATEGORIES)}
               for fold in first)
    tasks = [task_id for fold in first for task_id in fold.task_ids]
    assert len(tasks) == len(set(tasks)) == 56


def test_missing_targets_are_excluded_from_fit_but_receive_oof_predictions():
    dataset = synthetic_dataset()
    records, _, fit_audit, _ = generate_oof_predictions(dataset)
    assert len(records) == 448
    for variant in (LOGREG_UNWEIGHTED, LOGREG_BALANCED):
        selected = [record for record in records if record["variant"] == variant]
        assert len(selected) == 224
        assert len({(record["task_id"], record["candidate_id"]) for record in selected}) == 224
        assert sum(record["label_status"] == "missing" for record in selected) == 8
        assert sum(fold["missing_rows_excluded_from_fit"] for fold in fit_audit[variant]) == 24
        assert all(fold["test_rows_predicted"] == 56 for fold in fit_audit[variant])


def test_preprocessing_is_constructed_and_fitted_inside_each_fold():
    dataset = synthetic_dataset()
    _, _, fit_audit, folds = generate_oof_predictions(dataset)
    for variant in fit_audit:
        assert len(fit_audit[variant]) == 4
        for audit, assignment in zip(fit_audit[variant], folds):
            assert audit["preprocessor_instance_is_fold_local"] is True
            assert set(audit["train_task_ids"]).isdisjoint(assignment.task_ids)
            assert set(audit["test_task_ids"]) == set(assignment.task_ids)
            assert audit["train_groups"] == 42 and audit["test_groups"] == 14


def test_training_results_are_deterministic():
    dataset = synthetic_dataset()
    first = generate_oof_predictions(dataset)[0]
    second = generate_oof_predictions(dataset)[0]
    assert first == second


def test_pipeline_has_required_fold_local_transformers_and_two_weight_options():
    unweighted = build_pipeline(None)
    balanced = build_pipeline("balanced")
    assert unweighted.named_steps["classifier"].class_weight is None
    assert balanced.named_steps["classifier"].class_weight == "balanced"
    assert unweighted.named_steps["classifier"].l1_ratio == 0
    names = [name for name, _, _ in unweighted.named_steps["preprocess"].transformers]
    assert names == ["categorical", "numeric", "boolean"]
    assert len(CATEGORICAL_FEATURES) + len(NUMERIC_FEATURES) + len(BOOLEAN_FEATURES) == len(PREDICTIVE_FEATURES)


def test_cost_aware_selection_uses_projected_cost_not_realized_cost():
    dataset = synthetic_dataset()
    task_id = "qa-1"
    rows = [row for row in dataset.rows if row.task_id == task_id][:2]
    cheap, expensive = rows
    cheap = cheap.model_copy(update={"realized_cost_usd": Decimal("999")})
    expensive = expensive.model_copy(update={"realized_cost_usd": Decimal("0")})
    index = {(row.task_id, row.candidate_id): row for row in (cheap, expensive)}
    candidates = [
        {"candidate_id": cheap.candidate_id, "predicted_probability": 0.7},
        {"candidate_id": expensive.candidate_id, "predicted_probability": 0.8},
    ]
    selected, fallback = select_cost_aware_candidate(task_id, candidates, index, 0.6)
    assert selected["candidate_id"] == cheap.candidate_id
    assert fallback is False


def test_fallback_selects_highest_probability_then_projected_cost():
    dataset = synthetic_dataset()
    task_id = "qa-1"
    rows = [row for row in dataset.rows if row.task_id == task_id][:3]
    index = {(row.task_id, row.candidate_id): row for row in rows}
    candidates = [
        {"candidate_id": rows[0].candidate_id, "predicted_probability": 0.2},
        {"candidate_id": rows[1].candidate_id, "predicted_probability": 0.4},
        {"candidate_id": rows[2].candidate_id, "predicted_probability": 0.3},
    ]
    selected, fallback = select_cost_aware_candidate(task_id, candidates, index, 0.9)
    assert selected["candidate_id"] == rows[1].candidate_id
    assert fallback is True


def test_predictive_metrics_exclude_missing_targets():
    records = [
        {"label_status": "valid", "target": True, "predicted_probability": 0.8},
        {"label_status": "valid", "target": False, "predicted_probability": 0.2},
        {"label_status": "missing", "target": None, "predicted_probability": 0.99},
    ]
    metrics = _predictive_metrics(records)
    assert metrics["rows"] == 2
    assert metrics["positive"] == metrics["negative"] == 1


def test_selected_missing_label_is_not_counted_as_negative():
    feature = RequestFeatures(
        category="qa", prompt_characters=1, system_prompt_characters=1,
        approximate_input_tokens=1, contains_code=False,
        requests_structured_output=False, max_output_tokens=16,
        constraint_indicator_count=0, reasoning_indicator_count=0)
    observations = [
        RowObservation(
            task_id="t1", candidate_id=NEMOTRON, features=feature,
            analysis_difficulty="easy",
            option=CandidateOption(candidate_id=NEMOTRON,
                input_cost_per_1m_tokens="0.05", output_cost_per_1m_tokens="0.15",
                effective_max_output_tokens=16),
            label_status="missing", acceptable=None, quality_score=None,
            realized_cost_usd="0.001", latency_ms=10),
    ]
    summary, _ = evaluate_decisions(observations, [RoutingDecision(
        policy="TEST", task_id="t1", candidate_id=NEMOTRON, reason="test")])
    assert summary["missing_selected_labels"] == 1
    assert summary["valid_selected_labels"] == 0
    assert summary["unacceptable"] == 0


def test_threshold_grid_is_exactly_frozen():
    assert THRESHOLDS == (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)


def test_fold_target_counts_cover_only_each_folds_held_out_requests():
    dataset = synthetic_dataset()
    folds = build_outer_folds(dataset)
    rows_by_task = {task_id: [row for row in dataset.rows if row.task_id == task_id]
                    for task_id in {row.task_id for row in dataset.rows}}
    for fold in folds:
        rows = [row for task_id in fold.task_ids for row in rows_by_task[task_id]]
        assert fold.positive_labels == sum(row.acceptable is True for row in rows)
        assert fold.negative_labels == sum(row.acceptable is False for row in rows)
        assert fold.missing_labels == sum(row.label_status == "missing" for row in rows)
        assert Counter(row.category for row in rows) == Counter({category: 8 for category in CATEGORIES})


def test_fold_local_strongest_uses_training_labels_and_matches_test_request_set():
    feature = RequestFeatures(
        category="qa", prompt_characters=1, system_prompt_characters=1,
        approximate_input_tokens=1, contains_code=False,
        requests_structured_output=False, max_output_tokens=16,
        constraint_indicator_count=0, reasoning_indicator_count=0)
    options = (
        CandidateOption(candidate_id="a", input_cost_per_1m_tokens="1",
                        output_cost_per_1m_tokens="1", effective_max_output_tokens=16),
        CandidateOption(candidate_id="b", input_cost_per_1m_tokens="2",
                        output_cost_per_1m_tokens="2", effective_max_output_tokens=16),
    )
    observations = []
    for task_id, labels in (("train-1", (True, False)), ("train-2", (True, False)),
                            ("test", (False, True))):
        for option_value, acceptable in zip(options, labels):
            observations.append(RowObservation(
                task_id=task_id, candidate_id=option_value.candidate_id,
                features=feature, analysis_difficulty="easy", option=option_value,
                label_status="valid", acceptable=acceptable,
                quality_score=1.0 if acceptable else 0.0,
                realized_cost_usd="0.001", latency_ms=1))
    frozen = FrozenRoutingData(
        run_id=UUID("61707aba-5ab2-4c16-8ec3-eed74555d69c"),
        dataset_sha256="dataset", protocol_sha256="protocol",
        observations=tuple(observations),
        requests=tuple(PolicyRequest(task_id=task_id, features=feature, candidates=options)
                       for task_id in ("train-1", "train-2", "test")),
    )
    fold = FoldAssignment(
        fold=0, task_ids=("test",), category_counts={"qa": 1},
        positive_labels=1, negative_labels=1, missing_labels=0)
    decisions, resolved = fold_local_strongest_decisions(frozen, (fold,))
    assert [decision.task_id for decision in decisions] == ["test"]
    assert decisions[0].candidate_id == "a"
    assert resolved["0"]["candidate_id"] == "a"


def test_artifact_json_normalization_is_deterministic():
    left = _json({"z": 0.123456789012345, "a": {"b": 2, "a": 1}})
    right = _json({"a": {"a": 1, "b": 2}, "z": 0.123456789012345})
    assert left == right
    assert "0.123456789012" in left
