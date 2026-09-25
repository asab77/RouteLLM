"""Offline Phase 8D validation of the composed production routing path."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

from adaptive_llm_gateway.benchmarks.models import load_dataset
from adaptive_llm_gateway.pricing import calculate_projected_cost
from adaptive_llm_gateway.providers.gateway_config import CANDIDATE_MODELS

from .features import ProductionRequestFeatureExtractor
from .ml_experiment import THRESHOLDS
from .policy import CandidatePrediction, CostAwareRoutingPolicy
from .predictor import SklearnQualityPredictor, load_trusted_quality_artifact
from .quality_features import (
    canonical_feature_matrix,
    canonical_from_production,
    canonicalize_category,
    resolve_effective_output_allowance,
)
from .service import RoutingDecisionService

DEFAULT_ARTIFACT_DIRECTORY = Path(
    "artifacts/routing-quality/interaction-no-provider-pin-v1"
)
DEFAULT_REPORT_PATH = Path("artifacts/routing-validation/phase-8d-report.json")
FOUNDATION_V3_PATH = Path("benchmarks/datasets/foundation-v3.json")


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_production_routing_path(
    artifact_directory: Path,
    *,
    dataset_path: Path = FOUNDATION_V3_PATH,
    report_path: Path | None = DEFAULT_REPORT_PATH,
) -> dict[str, object]:
    """Compare composed routing with direct final-artifact reference semantics."""
    pipeline, metadata = load_trusted_quality_artifact(artifact_directory)
    predictor = SklearnQualityPredictor.from_trusted_artifact(artifact_directory)
    extractor = ProductionRequestFeatureExtractor()
    policy = CostAwareRoutingPolicy()
    service = RoutingDecisionService(
        predictor,
        feature_extractor=extractor,
        policy=policy,
    )
    dataset = load_dataset(dataset_path)
    prediction_count = prediction_parity = 0
    maximum_probability_difference = 0.0
    decision_count = decision_parity = 0
    selected_model_parity = projected_cost_parity = 0
    reason_parity = qualifying_count_parity = fallback_state_parity = 0
    repeated_deterministic = True
    candidate_order_independent = True
    threshold_diagnostics = {
        f"{threshold:.2f}": {
            "selected_model_distribution": Counter(),
            "fallback_count": 0,
            "total_projected_cost_usd": Decimal(0),
        }
        for threshold in THRESHOLDS
    }
    for task in dataset.tasks:
        request = task.to_request()
        category = canonicalize_category(task.category)
        structured = task.expected_output_type in {"json", "code"}
        features = extractor.extract(
            request,
            category_hint=category,
            structured_output_required=structured,
        )
        production_predictions = predictor.predict(features, CANDIDATE_MODELS)
        canonical = tuple(
            canonical_from_production(features, candidate)
            for candidate in CANDIDATE_MODELS
        )
        reference_probabilities = pipeline.predict_proba(
            canonical_feature_matrix(canonical)
        )[:, 1]
        reference_predictions = tuple(
            CandidatePrediction(
                model_id=candidate.model_id,
                predicted_acceptability=float(probability),
                projected_cost_usd=calculate_projected_cost(
                    approximate_input_tokens=features.approximate_input_tokens,
                    effective_max_output_tokens=resolve_effective_output_allowance(
                        features, candidate
                    ),
                    model=candidate,
                ),
            )
            for candidate, probability in zip(CANDIDATE_MODELS, reference_probabilities)
        )
        if [item.model_id for item in production_predictions] != [
            item.model_id for item in reference_predictions
        ]:
            raise ValueError("production prediction candidate association mismatch")
        for production, reference in zip(production_predictions, reference_predictions):
            difference = abs(
                production.predicted_acceptability - reference.predicted_acceptability
            )
            prediction_count += 1
            maximum_probability_difference = max(maximum_probability_difference, difference)
            prediction_parity += difference <= 1e-15
        previous_qualifying: set[str] | None = None
        for threshold in THRESHOLDS:
            key = f"{threshold:.2f}"
            qualifying = {
                item.model_id for item in reference_predictions
                if item.predicted_acceptability >= threshold
            }
            if previous_qualifying is not None and not qualifying <= previous_qualifying:
                raise ValueError("qualifying candidate set grew as threshold increased")
            previous_qualifying = qualifying
            actual = service.route(
                request,
                CANDIDATE_MODELS,
                threshold,
                category_hint=category,
                structured_output_required=structured,
            )
            expected = policy.route(reference_predictions, threshold)
            repeated = service.route(
                request,
                CANDIDATE_MODELS,
                threshold,
                category_hint=category,
                structured_output_required=structured,
            )
            reordered = service.route(
                request,
                tuple(reversed(CANDIDATE_MODELS)),
                threshold,
                category_hint=category,
                structured_output_required=structured,
            )
            decision_count += 1
            selected_model_parity += actual.selected_model_id == expected.selected_model_id
            projected_cost_parity += (
                actual.selected_projected_cost_usd == expected.selected_projected_cost_usd
            )
            reason_parity += actual.reason is expected.reason
            qualifying_count_parity += (
                actual.qualifying_candidate_count == expected.qualifying_candidate_count
            )
            fallback_state_parity += (
                actual.fallback_used == expected.fallback_used
                and actual.threshold_satisfied == expected.threshold_satisfied
            )
            if actual != expected:
                raise ValueError("composed routing decision differs from direct reference")
            decision_parity += 1
            repeated_deterministic = repeated_deterministic and repeated == actual
            candidate_order_independent = candidate_order_independent and reordered == actual
            if actual.selected_model_id not in {item.model_id for item in CANDIDATE_MODELS}:
                raise ValueError("routing selected a candidate outside the eligible set")
            summary = threshold_diagnostics[key]
            summary["selected_model_distribution"][actual.selected_model_id] += 1
            summary["fallback_count"] += int(actual.fallback_used)
            summary["total_projected_cost_usd"] += actual.selected_projected_cost_usd
    serializable_thresholds = {}
    for key, values in threshold_diagnostics.items():
        total_cost = values["total_projected_cost_usd"]
        serializable_thresholds[key] = {
            "selected_model_distribution": dict(
                sorted(values["selected_model_distribution"].items())
            ),
            "fallback_count": values["fallback_count"],
            "total_projected_cost_usd": str(total_cost),
            "average_projected_cost_usd": str(total_cost / Decimal(len(dataset.tasks))),
        }
    report = {
        "phase": "8D",
        "scope": "full-fit artifact integration diagnostics; not OOF generalization evidence",
        "artifact_model_sha256": metadata.model_sha256,
        "reference_requests": len(dataset.tasks),
        "candidate_predictions": prediction_count,
        "prediction_parity_count": prediction_parity,
        "maximum_probability_difference": maximum_probability_difference,
        "threshold_grid": list(THRESHOLDS),
        "routing_decisions": decision_count,
        "decision_parity_count": decision_parity,
        "selected_model_parity_count": selected_model_parity,
        "projected_cost_parity_count": projected_cost_parity,
        "routing_reason_parity_count": reason_parity,
        "qualifying_count_parity_count": qualifying_count_parity,
        "fallback_state_parity_count": fallback_state_parity,
        "repeated_run_deterministic": repeated_deterministic,
        "candidate_order_independent": candidate_order_independent,
        "per_threshold": serializable_thresholds,
    }
    if report_path is not None:
        _atomic_write(
            report_path,
            json.dumps(report, indent=2, sort_keys=True) + "\n",
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the offline production routing path with a trusted artifact"
    )
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT_DIRECTORY)
    parser.add_argument("--dataset", type=Path, default=FOUNDATION_V3_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()
    report = validate_production_routing_path(
        args.artifact,
        dataset_path=args.dataset,
        report_path=args.report,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
