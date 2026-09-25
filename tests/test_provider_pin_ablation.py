import hashlib
import inspect
from decimal import Decimal
from pathlib import Path
from uuid import UUID

import pytest

from adaptive_llm_gateway.routing.analysis import (
    FOUNDATION_V2_SHA256,
    FOUNDATION_V3_PROTOCOL_SHA256,
    FOUNDATION_V3_SHA256,
)
from adaptive_llm_gateway.routing.ml_diagnostics import (
    CATEGORY_CANDIDATE_INTERACTION,
    INTERACTION_FEATURE,
    REPRESENTATIONS,
)
from adaptive_llm_gateway.routing.ml_experiment import DESIGN_SHA256, THRESHOLDS
from adaptive_llm_gateway.routing.ml_features import build_outer_folds, load_ml_dataset
from adaptive_llm_gateway.routing.provider_pin_ablation import (
    INTERACTION_NO_PROVIDER_PIN,
    NO_PIN_REPRESENTATION,
    generate_predictions,
    provider_pin_mapping,
    run_ablation,
)

RUN_ID = UUID("61707aba-5ab2-4c16-8ec3-eed74555d69c")
ROOT = Path("benchmark-results")


@pytest.fixture(scope="module")
def dataset():
    return load_ml_dataset(ROOT, RUN_ID)


@pytest.fixture(scope="module")
def generated(dataset):
    return generate_predictions(dataset)


@pytest.fixture(scope="module")
def ablation():
    return run_ablation(ROOT, RUN_ID)


def test_accepted_baseline_reproduces(ablation):
    _, result = ablation
    reproduction = result["baseline_reproduction"]
    assert reproduction["stored_rows"] == reproduction["regenerated_rows"] == 224
    assert reproduction["exactly_reproduced_at_stored_precision"] is True


def test_exact_phase_7_grouped_folds_are_reused(dataset, generated):
    _, audit, _, folds = generated
    assert folds == build_outer_folds(dataset)
    for variant_audit in audit.values():
        assert [entry["test_task_ids"] for entry in variant_audit] == [
            sorted(fold.task_ids) for fold in folds
        ]


def test_exact_labels_and_missing_statuses_are_reused(dataset, generated):
    records, _, _, _ = generated
    expected = {
        (row.task_id, row.candidate_id): (row.label_status, row.acceptable)
        for row in dataset.rows
    }
    for record in records:
        assert (record["label_status"], record["target"]) == expected[
            (record["task_id"], record["candidate_id"])
        ]
    assert sum(row.label_status == "missing" for row in dataset.rows) == 8
    assert all(
        record["target"] is None
        for record in records
        if record["label_status"] == "missing"
    )


def test_provider_pin_is_present_in_the_accepted_baseline():
    assert "upstream_provider_pin" in REPRESENTATIONS[
        CATEGORY_CANDIDATE_INTERACTION
    ].features


def test_provider_pin_is_the_only_removed_feature():
    accepted = REPRESENTATIONS[CATEGORY_CANDIDATE_INTERACTION].features
    assert "upstream_provider_pin" not in NO_PIN_REPRESENTATION.features
    assert set(accepted) - set(NO_PIN_REPRESENTATION.features) == {
        "upstream_provider_pin"
    }
    assert len(accepted) == len(NO_PIN_REPRESENTATION.features) + 1


def test_all_other_predictive_features_keep_order_and_interaction():
    accepted = tuple(
        feature
        for feature in REPRESENTATIONS[CATEGORY_CANDIDATE_INTERACTION].features
        if feature != "upstream_provider_pin"
    )
    assert NO_PIN_REPRESENTATION.features == accepted
    assert INTERACTION_FEATURE in NO_PIN_REPRESENTATION.categorical
    assert NO_PIN_REPRESENTATION.numeric == REPRESENTATIONS[
        CATEGORY_CANDIDATE_INTERACTION
    ].numeric
    assert NO_PIN_REPRESENTATION.boolean == REPRESENTATIONS[
        CATEGORY_CANDIDATE_INTERACTION
    ].boolean


def test_preprocessing_and_logistic_regression_are_reused(generated):
    _, audits, coefficients, _ = generated
    assert set(audits) == {
        CATEGORY_CANDIDATE_INTERACTION,
        INTERACTION_NO_PROVIDER_PIN,
    }
    assert all(len(value) == 4 for value in coefficients.values())
    assert INTERACTION_NO_PROVIDER_PIN not in REPRESENTATIONS


def test_frozen_inputs_remain_unchanged():
    paths = {
        "v2": Path("benchmarks/datasets/foundation-v2.json"),
        "v3": Path("benchmarks/datasets/foundation-v3.json"),
        "protocol": Path("benchmarks/protocols/foundation-v3.json"),
    }
    assert hashlib.sha256(paths["v2"].read_bytes()).hexdigest() == FOUNDATION_V2_SHA256
    assert hashlib.sha256(paths["v3"].read_bytes()).hexdigest() == FOUNDATION_V3_SHA256
    assert (
        hashlib.sha256(paths["protocol"].read_bytes()).hexdigest()
        == FOUNDATION_V3_PROTOCOL_SHA256
    )
    assert DESIGN_SHA256 == "abb18acee90efd773a55eedd0f75ae69cb487b8b2f667c8d1aae336b0e8e434b"


def test_analysis_has_no_provider_execution_dependency():
    import adaptive_llm_gateway.routing.provider_pin_ablation as module

    source = inspect.getsource(module)
    assert "adaptive_llm_gateway.providers" not in source
    assert "InferenceService" not in source


def test_metrics_and_generated_artifacts_are_deterministic(ablation):
    first_paths, first_result = ablation
    first_hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in first_paths.items()
    }
    second_paths, second_result = run_ablation(ROOT, RUN_ID)
    second_hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in second_paths.items()
    }
    assert first_result == second_result
    assert first_hashes == second_hashes


def test_routing_threshold_grid_is_unchanged(ablation):
    _, result = ablation
    assert THRESHOLDS == (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
    assert result["routing"]["threshold_grid"] == list(THRESHOLDS)


def test_projected_cost_totals_use_the_governed_row_property(dataset, ablation):
    _, result = ablation
    index = {(row.task_id, row.candidate_id): row for row in dataset.rows}
    for variant in result["routing"]["variants"].values():
        for summary in variant.values():
            expected = sum(
                (
                    index[(selection["task_id"], selection["candidate_id"])].projected_cost_usd
                    for selection in summary["selections"]
                ),
                Decimal(0),
            )
            assert Decimal(summary["total_projected_cost_usd"]) == expected


def test_candidate_provider_mapping_is_deterministic_and_one_to_one(dataset):
    first = provider_pin_mapping(dataset)
    assert first == provider_pin_mapping(dataset)
    assert first["each_candidate_has_exactly_one_pin"] is True
    assert first["within_candidate_provider_variation"] is False
    assert first["relationship"] == "one_to_one"
    assert {entry["rows"] for entry in first["frequency_table"]} == {56}


def test_ranking_and_decision_accounting_cover_every_request(ablation):
    _, result = ablation
    ranking = result["ranking"]
    assert ranking["top_ranked_candidate_agreement"] == 56
    assert ranking["top_ranked_candidate_disagreement"] == 0
    for variant in result["routing"]["variants"].values():
        for summary in variant.values():
            assert summary["selected_rows"] == 56
            assert (
                summary["valid_selected_labels"] + summary["missing_selected_labels"]
                == 56
            )


def test_ablation_is_supported_only_as_a_phase_8c_candidate(ablation):
    _, result = ablation
    decision = result["decision_framework"]
    assert decision["Q7"] == "SUPPORTED_FOR_PHASE_8C"
    assert decision["conclusion"] == "SUPPORTED_FOR_PHASE_8C"
    assert decision["scope"] == (
        "Candidate formulation only; this is not production validation."
    )
