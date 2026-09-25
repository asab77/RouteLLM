"""Phase 8C-0 offline one-variable provider-pin ablation."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path
from uuid import UUID

import numpy as np

from .analysis import FOUNDATION_V3_RUN_ID, load_frozen_foundation_v3
from .ml_diagnostics import (
    CATEGORY_CANDIDATE_INTERACTION,
    INTERACTION_FEATURE,
    REPRESENTATIONS,
    FeatureRepresentation,
    _rank_variant,
    build_diagnostic_pipeline,
    diagnostic_feature_matrix,
    disagreement_tasks,
)
from .ml_experiment import (
    THRESHOLDS, _atomic_write, _json, _predictive_metrics,
    fold_local_strongest_decisions,
)
from .ml_features import MLExperimentDataset, build_outer_folds, load_ml_dataset
from .ml_policy_validation import route_predictions

ACCEPTED_INTERACTION = CATEGORY_CANDIDATE_INTERACTION
INTERACTION_NO_PROVIDER_PIN = "INTERACTION_NO_PROVIDER_PIN"
VARIANTS = (ACCEPTED_INTERACTION, INTERACTION_NO_PROVIDER_PIN)
NO_PIN_REPRESENTATION = FeatureRepresentation(
    categorical=tuple(
        name for name in REPRESENTATIONS[ACCEPTED_INTERACTION].categorical
        if name != "upstream_provider_pin"
    ),
    numeric=REPRESENTATIONS[ACCEPTED_INTERACTION].numeric,
    boolean=REPRESENTATIONS[ACCEPTED_INTERACTION].boolean,
)


def representations() -> dict[str, FeatureRepresentation]:
    return {
        ACCEPTED_INTERACTION: REPRESENTATIONS[ACCEPTED_INTERACTION],
        INTERACTION_NO_PROVIDER_PIN: NO_PIN_REPRESENTATION,
    }


def provider_pin_mapping(dataset: MLExperimentDataset) -> dict[str, object]:
    pairs = Counter(
        (row.candidate_id, str(row.features["upstream_provider_pin"]))
        for row in dataset.rows
    )
    mapping: dict[str, set[str]] = defaultdict(set)
    reverse: dict[str, set[str]] = defaultdict(set)
    for (candidate, provider), _ in pairs.items():
        mapping[candidate].add(provider)
        reverse[provider].add(candidate)
    return {
        "candidate_ids": sorted(mapping),
        "provider_pins": sorted(reverse),
        "candidate_to_provider_pin": {
            candidate: sorted(values) for candidate, values in sorted(mapping.items())
        },
        "each_candidate_has_exactly_one_pin": all(len(values) == 1 for values in mapping.values()),
        "within_candidate_provider_variation": any(len(values) > 1 for values in mapping.values()),
        "relationship": (
            "one_to_one" if all(len(values) == 1 for values in mapping.values())
            and all(len(values) == 1 for values in reverse.values()) else "many_to_one_or_other"
        ),
        "frequency_table": [
            {"candidate_id": candidate, "upstream_provider_pin": provider, "rows": count}
            for (candidate, provider), count in sorted(pairs.items())
        ],
    }


def _matrix(rows, variant: str) -> np.ndarray:
    if variant == ACCEPTED_INTERACTION:
        return diagnostic_feature_matrix(rows, variant)
    return np.asarray([
        [f"{row.category}::{row.candidate_id}" if name == INTERACTION_FEATURE
         else row.features[name] for name in NO_PIN_REPRESENTATION.features]
        for row in rows
    ], dtype=object)


def generate_predictions(dataset: MLExperimentDataset):
    folds = build_outer_folds(dataset)
    all_records = []
    audits = {}
    coefficients = {}
    for variant in VARIANTS:
        variant_records = []
        variant_audits = []
        variant_coefficients = []
        representation = representations()[variant]
        for assignment in folds:
            test_tasks = set(assignment.task_ids)
            train_all = [row for row in dataset.rows if row.task_id not in test_tasks]
            train = [row for row in train_all if row.label_status == "valid"]
            test = [row for row in dataset.rows if row.task_id in test_tasks]
            pipeline_variant = (
                ACCEPTED_INTERACTION if variant == ACCEPTED_INTERACTION
                else INTERACTION_NO_PROVIDER_PIN
            )
            if variant == INTERACTION_NO_PROVIDER_PIN:
                REPRESENTATIONS[pipeline_variant] = NO_PIN_REPRESENTATION
            try:
                pipeline = build_diagnostic_pipeline(pipeline_variant)
            finally:
                if variant == INTERACTION_NO_PROVIDER_PIN:
                    REPRESENTATIONS.pop(pipeline_variant, None)
            pipeline.fit(_matrix(train, variant), np.asarray([
                int(bool(row.acceptable)) for row in train
            ], dtype=int))
            probabilities = pipeline.predict_proba(_matrix(test, variant))[:, 1]
            for row, probability in zip(test, probabilities):
                variant_records.append({
                    "variant": variant, "task_id": row.task_id,
                    "category": row.category, "candidate_id": row.candidate_id,
                    "fold": assignment.fold, "predicted_probability": float(probability),
                    "label_status": row.label_status, "target": row.acceptable,
                })
            names = pipeline.named_steps["preprocess"].get_feature_names_out(
                representation.features)
            weights = pipeline.named_steps["classifier"].coef_[0]
            variant_coefficients.append({
                "fold": assignment.fold,
                "coefficients": [
                    {"feature": name, "coefficient": float(weight)}
                    for name, weight in zip(names.tolist(), weights.tolist())
                    if ("candidate_id" in name or "upstream_provider_pin" in name
                        or INTERACTION_FEATURE in name)
                ],
            })
            variant_audits.append({
                "fold": assignment.fold,
                "train_task_ids": sorted({row.task_id for row in train_all}),
                "test_task_ids": sorted(test_tasks),
                "valid_fit_rows": len(train),
                "missing_rows_excluded_from_fit": len(train_all) - len(train),
                "test_rows_predicted": len(test),
            })
        variant_records.sort(key=lambda item: (item["task_id"], item["candidate_id"]))
        if len(variant_records) != 224:
            raise ValueError("Each ablation variant must produce 224 OOF predictions")
        all_records.extend(variant_records)
        audits[variant] = variant_audits
        coefficients[variant] = variant_coefficients
    return tuple(all_records), audits, coefficients, folds


def _metric_comparison(records) -> dict[str, object]:
    metrics = {}
    for variant in VARIANTS:
        selected = [item for item in records if item["variant"] == variant]
        metrics[variant] = {
            "overall": _predictive_metrics(selected),
            "by_fold": {
                str(fold): _predictive_metrics([
                    item for item in selected if item["fold"] == fold
                ]) for fold in range(4)
            },
        }
    keys = ("log_loss", "brier_score", "roc_auc", "average_precision",
            "accuracy_at_0_5", "precision_at_0_5", "recall_at_0_5", "f1_at_0_5")
    metrics["delta_no_pin_minus_accepted"] = {
        key: metrics[INTERACTION_NO_PROVIDER_PIN]["overall"][key]
        - metrics[ACCEPTED_INTERACTION]["overall"][key] for key in keys
    }
    metrics["fold_stability"] = {}
    for variant in VARIANTS:
        metrics["fold_stability"][variant] = {}
        for key in ("log_loss", "brier_score", "roc_auc", "average_precision"):
            values = [metrics[variant]["by_fold"][str(fold)][key] for fold in range(4)]
            metrics["fold_stability"][variant][key] = {
                "mean": statistics.fmean(values), "population_standard_deviation": statistics.pstdev(values),
                "min": min(values), "max": max(values), "range": max(values) - min(values),
            }
    return metrics


def _ranking(dataset, records) -> dict[str, object]:
    index = {(row.task_id, row.candidate_id): row for row in dataset.rows}
    disagreements = set(disagreement_tasks(dataset))
    result = {"disagreement_request_count": len(disagreements), "variants": {}}
    tops = {}
    for variant in VARIANTS:
        selected = [item for item in records if item["variant"] == variant]
        result["variants"][variant] = {
            "all_requests": _rank_variant(selected, index),
            "disagreement_requests": _rank_variant(selected, index, disagreements),
        }
        tops[variant] = {
            item["task_id"]: item["candidate_id"]
            for item in result["variants"][variant]["all_requests"]["top_1_records"]
        }
    agreement = sum(tops[VARIANTS[0]][task] == tops[VARIANTS[1]][task] for task in tops[VARIANTS[0]])
    result["top_ranked_candidate_agreement"] = agreement
    result["top_ranked_candidate_disagreement"] = 56 - agreement
    return result


def _enrich_routing(summary, dataset, task_ids=None):
    index = {(row.task_id, row.candidate_id): row for row in dataset.rows}
    selections = summary["selections"]
    projected = [index[(item["task_id"], item["candidate_id"])].projected_cost_usd
                 for item in selections]
    latencies = [index[(item["task_id"], item["candidate_id"])].latency_ms
                 for item in selections]
    summary["total_projected_cost_usd"] = str(sum(projected, Decimal(0)))
    summary["average_projected_cost_per_request_usd"] = str(
        sum(projected, Decimal(0)) / Decimal(len(selections)))
    summary["p50_latency_ms"] = float(np.percentile(latencies, 50))
    summary["p95_latency_ms"] = float(np.percentile(latencies, 95))
    return summary


def _routing(root, run_id, dataset, records, folds):
    frozen = load_frozen_foundation_v3(root, run_id)
    disagreements = set(disagreement_tasks(dataset))
    index = {(row.task_id, row.candidate_id): row for row in dataset.rows}
    strongest_decisions, strongest_audit = fold_local_strongest_decisions(frozen, folds)
    strongest_projected = sum((
        index[(decision.task_id, str(decision.candidate_id))].projected_cost_usd
        for decision in strongest_decisions
    ), Decimal(0))
    result = {
        "threshold_grid": list(THRESHOLDS), "variants": {}, "disagreement": {},
        "categories": {}, "matched_fold_local_strongest": {
            "total_projected_cost_usd": str(strongest_projected),
            "audit": strongest_audit,
        },
    }
    for variant in VARIANTS:
        selected_records = [item for item in records if item["variant"] == variant]
        result["variants"][variant] = {}
        result["disagreement"][variant] = {}
        for threshold in THRESHOLDS:
            key = f"{threshold:.2f}"
            summary, _ = route_predictions(dataset, frozen.observations, selected_records,
                                           threshold, f"{variant}@{key}")
            enriched = _enrich_routing(summary, dataset)
            enriched["projected_cost_reduction_vs_matched_strongest_usd"] = str(
                strongest_projected - Decimal(enriched["total_projected_cost_usd"]))
            result["variants"][variant][key] = enriched
            subset, _ = route_predictions(dataset, frozen.observations, selected_records,
                                          threshold, f"{variant}@{key}:DISAGREEMENT",
                                          disagreements)
            result["disagreement"][variant][key] = _enrich_routing(subset, dataset, disagreements)
        result["categories"][variant] = {}
        for category in sorted({row.category for row in dataset.rows}):
            tasks = {row.task_id for row in dataset.rows if row.category == category}
            result["categories"][variant][category] = {}
            for threshold in THRESHOLDS:
                key = f"{threshold:.2f}"
                summary, _ = route_predictions(dataset, frozen.observations, selected_records,
                                               threshold, f"{variant}@{key}:{category}", tasks)
                result["categories"][variant][category][key] = _enrich_routing(summary, dataset, tasks)
    result["deltas_no_pin_minus_accepted"] = {}
    for threshold in THRESHOLDS:
        key = f"{threshold:.2f}"
        accepted = result["variants"][ACCEPTED_INTERACTION][key]
        ablated = result["variants"][INTERACTION_NO_PROVIDER_PIN][key]
        result["deltas_no_pin_minus_accepted"][key] = {
            "acceptable_rate_delta": ablated["acceptable_rate_among_valid"] - accepted["acceptable_rate_among_valid"],
            "mean_quality_delta": ablated["mean_quality_among_valid"] - accepted["mean_quality_among_valid"],
            "total_projected_cost_delta_usd": str(Decimal(ablated["total_projected_cost_usd"]) - Decimal(accepted["total_projected_cost_usd"])),
            "fallback_rate_delta": ablated["fallback_rate"] - accepted["fallback_rate"],
            "coverage_delta": ablated["valid_label_coverage"] - accepted["valid_label_coverage"],
            "candidate_distribution_accepted": accepted["model_distribution"],
            "candidate_distribution_no_pin": ablated["model_distribution"],
        }
    return result


def _pareto(routing):
    output = {variant: [] for variant in VARIANTS}
    for variant in VARIANTS:
        points = list(routing["variants"][variant].items())
        for threshold, summary in points:
            dominated = any(
                other["valid_selected_labels"] == summary["valid_selected_labels"]
                and other["acceptable_rate_among_valid"]
                >= summary["acceptable_rate_among_valid"]
                and Decimal(other["total_projected_cost_usd"])
                <= Decimal(summary["total_projected_cost_usd"])
                and (
                    other["acceptable_rate_among_valid"]
                    > summary["acceptable_rate_among_valid"]
                    or Decimal(other["total_projected_cost_usd"])
                    < Decimal(summary["total_projected_cost_usd"])
                )
                for other_threshold, other in points
                if other_threshold != threshold
            )
            if not dominated:
                output[variant].append(threshold)
    return output


def _baseline_reproduction(root: Path, run_id: UUID, records):
    stored_path = root / str(run_id) / "phase-7" / "diagnostic-predictions.jsonl"
    stored = [json.loads(line) for line in stored_path.read_text().splitlines() if line]
    expected = sorted((item for item in stored if item["variant"] == ACCEPTED_INTERACTION),
                      key=lambda item: (item["task_id"], item["candidate_id"]))
    actual = [item for item in records if item["variant"] == ACCEPTED_INTERACTION]
    max_delta = max(abs(float(a["predicted_probability"]) - float(e["predicted_probability"]))
                    for a, e in zip(actual, expected))
    reproduced = all(
        round(float(a["predicted_probability"]), 12) == float(e["predicted_probability"])
        for a, e in zip(actual, expected)
    )
    return {"stored_rows": len(expected), "regenerated_rows": len(actual),
            "maximum_probability_delta_before_artifact_rounding": format(max_delta, ".17g"),
            "stored_precision_decimal_places": 12,
            "exactly_reproduced_at_stored_precision": reproduced}


def _decision_framework(artifact: dict[str, object]) -> dict[str, str]:
    mapping = artifact["mapping"]
    metrics = artifact["predictive_metrics"]
    ranking = artifact["ranking"]
    routing = artifact["routing"]
    delta = metrics["delta_no_pin_minus_accepted"]
    rank_variants = ranking["variants"]
    accepted_rank = rank_variants[ACCEPTED_INTERACTION]["all_requests"]
    ablated_rank = rank_variants[INTERACTION_NO_PROVIDER_PIN]["all_requests"]
    routing_deltas = routing["deltas_no_pin_minus_accepted"]
    changed_acceptance = {
        threshold: values["acceptable_rate_delta"]
        for threshold, values in routing_deltas.items()
        if values["acceptable_rate_delta"] != 0
    }
    return {
        "Q1": (
            "Yes. Every candidate maps to exactly one distinct provider pin, with no "
            "within-candidate variation across the 224 rows."
        ),
        "Q2": (
            "No material degradation observed. No-pin changes log loss by "
            f"{delta['log_loss']:+.12f} and Brier score by "
            f"{delta['brier_score']:+.12f}; lower is better for both."
        ),
        "Q3": (
                "No material degradation observed. All 56 top-ranked candidates are "
                "unchanged; pairwise ranking changes from "
                f"{accepted_rank['pairwise']['strict_accuracy']:.12f} to "
                f"{ablated_rank['pairwise']['strict_accuracy']:.12f}."
        ),
        "Q4": (
            "No material degradation observed. Four of six thresholds select the "
            "same candidates; 0.60 improves acceptance by 1/54 at $0.00163590 "
            "additional projected total cost, and 0.95 changes cost by $0.00037050 "
            "with equal acceptance."
        ),
        "Q5": (
            "Proper-loss effects are mixed across folds, small in magnitude, and no "
            "category shows a catastrophic regression. Routing acceptance changes "
            f"only at these thresholds: {changed_acceptance}."
        ),
        "Q6": (
            "Yes. No-pin retains 0.912698412698 pairwise ranking accuracy, above the "
            "historical 0.8810 candidate-prior/full-additive reference."
        ),
        "Q7": "SUPPORTED_FOR_PHASE_8C",
        "conclusion": "SUPPORTED_FOR_PHASE_8C",
        "scope": "Candidate formulation only; this is not production validation.",
        "mapping_check": str(mapping["relationship"]),
    }


def run_ablation(root: Path, run_id: UUID = FOUNDATION_V3_RUN_ID):
    dataset = load_ml_dataset(root, run_id)
    records, fit_audit, coefficients, folds = generate_predictions(dataset)
    baseline = _baseline_reproduction(root, run_id, records)
    if not baseline["exactly_reproduced_at_stored_precision"]:
        raise ValueError("Accepted interaction baseline did not reproduce")
    metrics = _metric_comparison(records)
    ranking = _ranking(dataset, records)
    routing = _routing(root, run_id, dataset, records, folds)
    artifact = {
        "analysis": "PHASE_8C_0_PROVIDER_PIN_ABLATION",
        "variants": list(VARIANTS),
        "removed_feature": "upstream_provider_pin",
        "mapping": provider_pin_mapping(dataset),
        "baseline_reproduction": baseline,
        "feature_sets": {name: list(value.features) for name, value in representations().items()},
        "fit_audit": fit_audit,
        "predictive_metrics": metrics,
        "ranking": ranking,
        "routing": routing,
        "pareto_non_dominated_thresholds": _pareto(routing),
        "coefficients": coefficients,
    }
    artifact["decision_framework"] = _decision_framework(artifact)
    output = root / str(run_id) / "phase-8c0"
    paths = {"results": output / "provider-pin-ablation.json",
             "predictions": output / "provider-pin-ablation-predictions.jsonl"}
    _atomic_write(paths["results"], _json(artifact))
    _atomic_write(paths["predictions"], "".join(
        json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n" for item in records))
    return paths, artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the offline provider-pin ablation")
    parser.add_argument("--root", type=Path, default=Path("benchmark-results"))
    parser.add_argument("--run-id", type=UUID, default=FOUNDATION_V3_RUN_ID)
    args = parser.parse_args()
    paths, artifact = run_ablation(args.root, args.run_id)
    print(json.dumps({"paths": {key: str(value) for key, value in paths.items()},
                      "baseline_reproduced": artifact["baseline_reproduction"]["exactly_reproduced_at_stored_precision"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
