import hashlib
from decimal import Decimal
from pathlib import Path
from uuid import UUID

import pytest

from adaptive_llm_gateway.routing.analysis import GEMINI, LUNA, NEMOTRON, SONNET
from adaptive_llm_gateway.routing.ml_diagnostics import (
    CANDIDATE_ONLY,
    CATEGORY_CANDIDATE_INTERACTION,
    DIAGNOSTIC_VARIANTS,
    FULL_ADDITIVE,
    INTERACTION_FEATURE,
    REPRESENTATIONS,
    REQUEST_ONLY,
    _rank_variant,
    candidate_probability_diagnostics,
    diagnostic_feature_matrix,
    disagreement_tasks,
    gemini_diagnostics,
    generate_diagnostic_predictions,
    run_diagnostics,
)
from adaptive_llm_gateway.routing.ml_features import (
    FORBIDDEN_PREDICTIVE_FIELDS,
    MLExperimentDataset,
    MLExperimentRow,
    build_outer_folds,
    load_ml_dataset,
)

RUN_ID = UUID("61707aba-5ab2-4c16-8ec3-eed74555d69c")
RESULT_ROOT = Path("benchmark-results")
CATEGORIES = ("qa", "classification", "extraction", "json", "reasoning", "coding", "summarization")
CANDIDATES = (NEMOTRON, LUNA, GEMINI, SONNET)


def _features(category, candidate, task_index):
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


@pytest.fixture(scope="module")
def synthetic_dataset():
    rows = []
    missing = {(f"{category}-0", GEMINI) for category in CATEGORIES}
    missing.add(("reasoning-1", SONNET))
    for category_index, category in enumerate(CATEGORIES):
        for task_index in range(8):
            task_id = f"{category}-{task_index}"
            for candidate_index, candidate in enumerate(CANDIDATES):
                is_missing = (task_id, candidate) in missing
                acceptable = ((category_index + task_index + candidate_index * 2) % 5) < 3
                rows.append(MLExperimentRow(
                    task_id=task_id,
                    candidate_id=candidate,
                    category=category,
                    features=_features(category, candidate, task_index),
                    label_status="missing" if is_missing else "valid",
                    acceptable=None if is_missing else acceptable,
                    quality_score=None if is_missing else (1.0 if acceptable else 0.2),
                    analysis_difficulty=("easy", "medium", "hard")[task_index % 3],
                    realized_cost_usd=Decimal("0.01") * (candidate_index + 1),
                    latency_ms=float(100 * (candidate_index + 1)),
                ))
    return MLExperimentDataset(run_id=RUN_ID, rows=tuple(rows))


@pytest.fixture(scope="module")
def real_dataset():
    return load_ml_dataset(RESULT_ROOT, RUN_ID)


@pytest.fixture(scope="module")
def real_diagnostics():
    return run_diagnostics(RESULT_ROOT, RUN_ID)


def test_feature_representations_are_strictly_isolated(synthetic_dataset):
    candidate = REPRESENTATIONS[CANDIDATE_ONLY]
    request = REPRESENTATIONS[REQUEST_ONLY]
    interaction = REPRESENTATIONS[CATEGORY_CANDIDATE_INTERACTION]
    assert candidate.features == ("candidate_id",)
    assert "candidate_id" not in request.features
    assert "upstream_provider_pin" not in request.features
    assert "reasoning_effort" not in request.features
    assert "configured_input_cost_per_1m_tokens" not in request.features
    assert "configured_output_cost_per_1m_tokens" not in request.features
    assert "context_window" not in request.features
    assert "effective_max_output_tokens" not in request.features
    assert set(REPRESENTATIONS[FULL_ADDITIVE].features) == set(synthetic_dataset.rows[0].features)
    assert interaction.features[:-1] != REPRESENTATIONS[FULL_ADDITIVE].features
    assert set(interaction.features) - set(REPRESENTATIONS[FULL_ADDITIVE].features) == {
        INTERACTION_FEATURE}
    assert all(not set(item.features) & FORBIDDEN_PREDICTIVE_FIELDS
               for item in REPRESENTATIONS.values())


def test_interaction_value_is_only_category_by_candidate(synthetic_dataset):
    row = synthetic_dataset.rows[0]
    matrix = diagnostic_feature_matrix([row], CATEGORY_CANDIDATE_INTERACTION)
    index = REPRESENTATIONS[CATEGORY_CANDIDATE_INTERACTION].features.index(INTERACTION_FEATURE)
    assert matrix[0, index] == f"{row.category}::{row.candidate_id}"
    assert sum("_" in name and name == INTERACTION_FEATURE
               for name in REPRESENTATIONS[CATEGORY_CANDIDATE_INTERACTION].features) == 1


def test_all_variants_reuse_same_grouped_outer_folds_and_predict_missing(synthetic_dataset):
    records, audit, _, folds = generate_diagnostic_predictions(synthetic_dataset)
    assert folds == build_outer_folds(synthetic_dataset)
    for variant in DIAGNOSTIC_VARIANTS:
        selected = [item for item in records if item["variant"] == variant]
        assert len(selected) == 224
        assert sum(item["label_status"] == "missing" for item in selected) == 8
        assert sum(item["missing_rows_excluded_from_fit"] for item in audit[variant]) == 24
        for entry, assignment in zip(audit[variant], folds):
            assert set(entry["train_task_ids"]).isdisjoint(entry["test_task_ids"])
            assert set(entry["test_task_ids"]) == set(assignment.task_ids)


def test_pairwise_ranking_counts_correct_incorrect_and_ties(real_dataset):
    task_id = disagreement_tasks(real_dataset)[0]
    rows = [row for row in real_dataset.rows if row.task_id == task_id]
    index = {(row.task_id, row.candidate_id): row for row in rows}
    probabilities = {}
    positive_rank = 0.9
    negative_rank = 0.1
    for row in rows:
        probabilities[row.candidate_id] = positive_rank if row.acceptable else negative_rank
    records = [{
        "task_id": row.task_id,
        "candidate_id": row.candidate_id,
        "predicted_probability": probabilities[row.candidate_id],
        "label_status": row.label_status,
        "target": row.acceptable,
    } for row in rows]
    ranked = _rank_variant(records, index)
    assert ranked["pairwise"]["correct"] == ranked["pairwise"]["pairs"]
    assert ranked["pairwise"]["ties"] == 0
    for item in records:
        item["predicted_probability"] = 0.5
    tied = _rank_variant(records, index)
    assert tied["pairwise"]["ties"] == tied["pairwise"]["pairs"]
    assert tied["pairwise"]["strict_accuracy"] == 0
    assert tied["pairwise"]["tie_adjusted_accuracy"] == 0.5


def test_frozen_dataset_has_exactly_39_disagreement_requests(real_dataset):
    assert len(disagreement_tasks(real_dataset)) == 39


def test_candidate_probability_distributions_account_for_every_request(real_diagnostics):
    _, results = real_diagnostics
    probabilities = results["probabilities"]["by_variant_and_candidate"]
    for variant in DIAGNOSTIC_VARIANTS:
        assert set(probabilities[variant]) == set(CANDIDATES)
        assert all(summary["requests"] == 56 for summary in probabilities[variant].values())
        assert all(0 <= summary["min"] <= summary["max"] <= 1
                   for summary in probabilities[variant].values())


def test_rule_comparison_accounting_is_complete(real_diagnostics):
    _, results = real_diagnostics
    rules = results["rules"]
    assert sum(item["request_count"] for item in rules["branch_analysis"].values()) == 56
    for variant in (FULL_ADDITIVE, CATEGORY_CANDIDATE_INTERACTION):
        comparisons = [rules["ml_vs_rule"][variant]["top_ranked"]]
        comparisons.extend(rules["ml_vs_rule"][variant]["threshold_selected"].values())
        for comparison in comparisons:
            assert comparison["agreement_count"] + comparison["disagreement_count"] == 56
            assert sum(comparison["disagreement_outcomes"].values()) == comparison["disagreement_count"]


def test_gemini_diagnostic_accounting_is_complete(real_dataset, real_diagnostics):
    _, results = real_diagnostics
    gemini = results["gemini"]
    assert gemini["valid_labels"] == 51
    assert gemini["positive_labels"] == 9
    for variant in DIAGNOSTIC_VARIANTS:
        detail = gemini["by_variant"][variant]
        assert sum(detail["rank_distribution_stable_tiebreak"].values()) == 56
        counts = list(detail["threshold_exceed_counts"].values())
        assert counts == sorted(counts, reverse=True)
    assert gemini == gemini_diagnostics(
        real_dataset,
        generate_diagnostic_predictions(real_dataset)[0],
    )


def test_artifacts_are_deterministic_and_phase_7b_artifacts_unchanged(real_diagnostics):
    paths, _ = real_diagnostics
    phase7 = RESULT_ROOT / str(RUN_ID) / "phase-7"
    protected = [
        phase7 / "fold-assignments.json",
        phase7 / "oof-predictions.jsonl",
        phase7 / "predictive-metrics.json",
        phase7 / "routing-sensitivity.json",
        phase7 / "coefficient-summary.json",
        phase7 / "phase-7b-report.md",
        phase7 / "ml-router-design.json",
    ]
    before_new = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                  for name, path in paths.items()}
    before_protected = {path: hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in protected}
    run_diagnostics(RESULT_ROOT, RUN_ID)
    after_new = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                 for name, path in paths.items()}
    after_protected = {path: hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in protected}
    assert before_new == after_new
    assert before_protected == after_protected
