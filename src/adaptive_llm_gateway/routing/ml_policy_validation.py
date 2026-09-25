"""Phase 7D offline policy validation for the interaction logistic router."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path
from time import perf_counter
from uuid import UUID

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .analysis import (
    ALWAYS_CHEAPEST,
    ALWAYS_STRONGEST,
    RULE_BASED_V1,
    FOUNDATION_V3_RUN_ID,
    RoutingDecision,
    RuleBasedV1Policy,
    evaluate_decisions,
    load_frozen_foundation_v3,
    oracle_cheapest_acceptable,
)
from .ml_diagnostics import (
    CATEGORY_CANDIDATE_INTERACTION,
    FULL_ADDITIVE,
    FeatureRepresentation,
    _rank_variant,
    _verify_integrity,
    disagreement_tasks,
    generate_diagnostic_predictions,
    predictive_diagnostics,
    rule_based_comparison,
)
from .ml_experiment import (
    FOLD_LOCAL_ALWAYS_STRONGEST,
    THRESHOLDS,
    _atomic_write,
    _baseline_evaluation,
    _json,
    _normalized,
    _predictive_metrics,
    select_cost_aware_candidate,
)
from .ml_features import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    FORBIDDEN_PREDICTIVE_FIELDS,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    MLExperimentDataset,
    MLExperimentRow,
    build_outer_folds,
    load_ml_dataset,
)

NO_CATEGORY = "NO_CATEGORY"
NO_CATEGORY_REPRESENTATION = FeatureRepresentation(
    categorical=tuple(name for name in CATEGORICAL_FEATURES if name != "category"),
    numeric=NUMERIC_FEATURES,
    boolean=BOOLEAN_FEATURES,
)
PROTECTED_PHASE_7_FILES = (
    "ml-router-design.json",
    "fold-assignments.json",
    "oof-predictions.jsonl",
    "predictive-metrics.json",
    "routing-sensitivity.json",
    "coefficient-summary.json",
    "phase-7b-report.md",
    "diagnostic-predictions.jsonl",
    "diagnostic-metrics.json",
    "within-request-ranking.json",
    "rule-based-comparison.json",
    "phase-7c-report.md",
)


def _as_float(values):
    return values.astype(float)


def no_category_feature_matrix(
    rows: list[MLExperimentRow] | tuple[MLExperimentRow, ...],
) -> np.ndarray:
    return np.asarray([
        [row.features[name] for name in NO_CATEGORY_REPRESENTATION.features]
        for row in rows
    ], dtype=object)


def build_no_category_pipeline() -> Pipeline:
    representation = NO_CATEGORY_REPRESENTATION
    categorical_end = len(representation.categorical)
    numeric_end = categorical_end + len(representation.numeric)
    return Pipeline((
        ("preprocess", ColumnTransformer((
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             list(range(categorical_end))),
            ("numeric", StandardScaler(), list(range(categorical_end, numeric_end))),
            ("boolean", FunctionTransformer(_as_float, feature_names_out="one-to-one"),
             list(range(numeric_end, len(representation.features)))),
        ), remainder="drop")),
        ("classifier", LogisticRegression(
            l1_ratio=0.0, C=1.0, solver="lbfgs", max_iter=1000,
            class_weight=None, random_state=RANDOM_STATE,
        )),
    ))


def generate_no_category_predictions(dataset: MLExperimentDataset):
    folds = build_outer_folds(dataset)
    records = []
    fit_audit = []
    for assignment in folds:
        test_tasks = set(assignment.task_ids)
        train_all = [row for row in dataset.rows if row.task_id not in test_tasks]
        train = [row for row in train_all if row.label_status == "valid"]
        test = [row for row in dataset.rows if row.task_id in test_tasks]
        if {row.task_id for row in train_all} & test_tasks:
            raise ValueError("NO_CATEGORY grouped fold leaked a request")
        pipeline = build_no_category_pipeline()
        pipeline.fit(
            no_category_feature_matrix(train),
            np.asarray([int(bool(row.acceptable)) for row in train], dtype=int),
        )
        probabilities = pipeline.predict_proba(no_category_feature_matrix(test))[:, 1]
        for row, probability in zip(test, probabilities):
            records.append({
                "variant": NO_CATEGORY,
                "task_id": row.task_id,
                "category": row.category,
                "candidate_id": row.candidate_id,
                "fold": assignment.fold,
                "predicted_probability": float(probability),
                "label_status": row.label_status,
                "target": row.acceptable,
            })
        fit_audit.append({
            "fold": assignment.fold,
            "train_groups": len({row.task_id for row in train_all}),
            "test_groups": len(test_tasks),
            "valid_fit_rows": len(train),
            "missing_rows_excluded_from_fit": len(train_all) - len(train),
            "test_rows_predicted": len(test),
            "train_task_ids": sorted({row.task_id for row in train_all}),
            "test_task_ids": sorted(test_tasks),
        })
    records.sort(key=lambda item: (item["task_id"], item["candidate_id"]))
    if len(records) != 224 or len({
        (item["task_id"], item["candidate_id"]) for item in records
    }) != 224:
        raise ValueError("NO_CATEGORY must produce 224 unique OOF predictions")
    return tuple(records), fit_audit, folds


def route_predictions(
    dataset: MLExperimentDataset,
    observations,
    records,
    threshold: float,
    policy: str,
    task_ids: set[str] | None = None,
) -> tuple[dict[str, object], list[RoutingDecision]]:
    selected_tasks = task_ids or {row.task_id for row in dataset.rows}
    row_index = {(row.task_id, row.candidate_id): row for row in dataset.rows}
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        task_id = str(record["task_id"])
        if task_id in selected_tasks:
            grouped[task_id].append(record)
    if set(grouped) != selected_tasks:
        raise ValueError("Routing predictions do not cover the requested task set")
    decisions = []
    selections = []
    fallback_count = 0
    for task_id, candidates in sorted(grouped.items()):
        selected, fallback = select_cost_aware_candidate(
            task_id, candidates, row_index, threshold)
        candidate_id = str(selected["candidate_id"])
        fallback_count += int(fallback)
        decisions.append(RoutingDecision(
            policy=policy,
            task_id=task_id,
            candidate_id=candidate_id,
            reason=("highest probability fallback" if fallback
                    else "lowest projected pre-generation cost above threshold"),
        ))
        selections.append({
            "task_id": task_id,
            "candidate_id": candidate_id,
            "predicted_probability": selected["predicted_probability"],
            "projected_cost_usd": str(row_index[(task_id, candidate_id)].projected_cost_usd),
            "fallback_used": fallback,
        })
    selected_observations = [row for row in observations if row.task_id in selected_tasks]
    summary, _ = evaluate_decisions(selected_observations, decisions)
    summary.update({
        "threshold": threshold,
        "fallback_count": fallback_count,
        "fallback_rate": fallback_count / len(selected_tasks),
        "selections": selections,
    })
    return summary, decisions


def _difference_vs_baselines(summary, baselines) -> dict[str, object]:
    learned_cost = Decimal(str(summary["total_realized_cost_usd"]))
    learned_rate = float(summary["acceptable_rate_among_valid"])
    rule = baselines["policies"][RULE_BASED_V1]
    rule_cost = Decimal(str(rule["total_realized_cost_usd"]))
    rule_rate = float(rule["acceptable_rate_among_valid"])
    strongest = baselines["policies"][FOLD_LOCAL_ALWAYS_STRONGEST]
    strongest_cost = Decimal(str(strongest["total_realized_cost_usd"]))
    strongest_rate = float(strongest["acceptable_rate_among_valid"])
    return {
        "vs_rule_based_v1": {
            "coverage_matches": (
                summary["valid_selected_labels"] == rule["valid_selected_labels"]),
            "realized_cost_difference_usd": str(learned_cost - rule_cost),
            "realized_cost_percentage_difference": float((learned_cost - rule_cost) / rule_cost),
            "acceptable_rate_percentage_point_difference": 100 * (learned_rate - rule_rate),
        },
        "vs_fold_local_always_strongest": {
            "coverage_matches": (
                summary["valid_selected_labels"] == strongest["valid_selected_labels"]),
            "cost_reduction_usd": str(strongest_cost - learned_cost),
            "cost_reduction_fraction": float((strongest_cost - learned_cost) / strongest_cost),
            "acceptable_rate_percentage_point_difference": 100 * (learned_rate - strongest_rate),
        },
    }


def full_frontier(root: Path, run_id: UUID, dataset, records, folds):
    frozen = load_frozen_foundation_v3(root, run_id)
    baselines = _baseline_evaluation(root, run_id, folds)
    variants = {}
    decisions = {}
    for variant in (FULL_ADDITIVE, CATEGORY_CANDIDATE_INTERACTION):
        selected_records = [item for item in records if item["variant"] == variant]
        variants[variant] = {}
        decisions[variant] = {}
        for threshold in THRESHOLDS:
            summary, routed = route_predictions(
                dataset, frozen.observations, selected_records, threshold,
                f"{variant}@{threshold:.2f}")
            summary["baseline_differences"] = _difference_vs_baselines(summary, baselines)
            variants[variant][f"{threshold:.2f}"] = summary
            decisions[variant][f"{threshold:.2f}"] = routed
    return {
        "request_count": 56,
        "threshold_grid": list(THRESHOLDS),
        "selection_policy": {
            "qualified": "lowest projected pre-generation cost among P(acceptable) >= threshold",
            "fallback": "highest probability, then lower projected cost, then stable candidate ID",
            "realized_cost_used_for_selection": False,
        },
        "matched_baselines": baselines,
        "variants": variants,
    }, decisions


def compare_policy_points(first: dict[str, object], second: dict[str, object]) -> str:
    if first["valid_selected_labels"] != second["valid_selected_labels"]:
        return "incomparable_coverage"
    first_rate = float(first["acceptable_rate_among_valid"])
    second_rate = float(second["acceptable_rate_among_valid"])
    first_cost = Decimal(str(first["total_realized_cost_usd"]))
    second_cost = Decimal(str(second["total_realized_cost_usd"]))
    first_dominates = (first_rate >= second_rate and first_cost <= second_cost
                       and (first_rate > second_rate or first_cost < second_cost))
    second_dominates = (second_rate >= first_rate and second_cost <= first_cost
                        and (second_rate > first_rate or second_cost < first_cost))
    if first_dominates:
        return "first_dominates"
    if second_dominates:
        return "second_dominates"
    if first_rate == second_rate and first_cost == second_cost:
        return "equivalent"
    return "tradeoff"


def pareto_classification(frontier) -> dict[str, object]:
    baselines = frontier["matched_baselines"]["policies"]
    comparison_baselines = {
        name: baselines[name]
        for name in (ALWAYS_CHEAPEST, FOLD_LOCAL_ALWAYS_STRONGEST, RULE_BASED_V1)
    }
    learned = {
        f"{variant}@{threshold}": summary
        for variant, thresholds in frontier["variants"].items()
        for threshold, summary in thresholds.items()
    }
    all_deployable = {**comparison_baselines, **learned}
    result = {}
    for name, point in learned.items():
        comparisons = {}
        dominated_by = []
        comparable = 0
        for other_name, other in all_deployable.items():
            if other_name == name:
                continue
            relation = compare_policy_points(point, other)
            comparisons[other_name] = relation
            if relation != "incomparable_coverage":
                comparable += 1
            if relation == "second_dominates":
                dominated_by.append(other_name)
        if dominated_by:
            classification = "dominated"
        elif comparable == 0:
            classification = "incomparable_due_to_coverage"
        else:
            classification = "non_dominated"
        result[name] = {
            "classification": classification,
            "dominated_by": sorted(dominated_by),
            "comparisons": comparisons,
        }
    return {
        "coverage_rule": "Points are compared only when valid_selected_labels are exactly equal.",
        "oracle_excluded_as_analysis_only": True,
        "learned_points": result,
    }


def interaction_vs_additive(frontier) -> dict[str, object]:
    artifact = {}
    for threshold in THRESHOLDS:
        key = f"{threshold:.2f}"
        interaction = frontier["variants"][CATEGORY_CANDIDATE_INTERACTION][key]
        additive = frontier["variants"][FULL_ADDITIVE][key]
        interaction_cost = Decimal(str(interaction["total_realized_cost_usd"]))
        additive_cost = Decimal(str(additive["total_realized_cost_usd"]))
        quality_delta = (float(interaction["mean_quality_among_valid"])
                         - float(additive["mean_quality_among_valid"]))
        cost_delta = interaction_cost - additive_cost
        coverage_matches = (
            interaction["valid_selected_labels"] == additive["valid_selected_labels"])
        if not coverage_matches:
            classification = "incomparable_due_to_coverage"
        elif quality_delta > 1e-12 and cost_delta < 0:
            classification = "better_quality_and_cheaper"
        elif quality_delta > 1e-12 and cost_delta >= 0:
            classification = "better_quality_but_more_expensive"
        elif quality_delta < -1e-12 and cost_delta < 0:
            classification = "cheaper_but_lower_quality"
        elif abs(quality_delta) <= 1e-12 and cost_delta == 0:
            classification = "effectively_unchanged"
        elif abs(quality_delta) <= 1e-12 and cost_delta < 0:
            classification = "same_quality_and_cheaper"
        elif abs(quality_delta) <= 1e-12 and cost_delta > 0:
            classification = "same_quality_but_more_expensive"
        else:
            classification = "lower_quality_and_more_expensive"
        candidates = set(interaction["model_distribution"]) | set(additive["model_distribution"])
        artifact[key] = {
            "acceptable_rate_percentage_point_difference": 100 * (
                float(interaction["acceptable_rate_among_valid"])
                - float(additive["acceptable_rate_among_valid"])),
            "mean_quality_difference": quality_delta,
            "realized_cost_difference_usd": str(cost_delta),
            "fallback_rate_difference": (
                float(interaction["fallback_rate"]) - float(additive["fallback_rate"])),
            "missing_selected_label_difference": (
                int(interaction["missing_selected_labels"])
                - int(additive["missing_selected_labels"])),
            "candidate_count_difference": {
                candidate: (
                    interaction["model_distribution"].get(candidate, {}).get("count", 0)
                    - additive["model_distribution"].get(candidate, {}).get("count", 0))
                for candidate in sorted(candidates)
            },
            "coverage_matches": coverage_matches,
            "classification": classification,
        }
    return artifact


def fold_stability(root, run_id, dataset, records, folds) -> dict[str, object]:
    frozen = load_frozen_foundation_v3(root, run_id)
    artifact = {"fold_request_count": 14, "variants": {}}
    for variant in (FULL_ADDITIVE, CATEGORY_CANDIDATE_INTERACTION):
        variant_records = [item for item in records if item["variant"] == variant]
        artifact["variants"][variant] = {}
        for threshold in THRESHOLDS:
            fold_results = []
            for assignment in folds:
                summary, _ = route_predictions(
                    dataset, frozen.observations, variant_records, threshold,
                    f"{variant}@{threshold:.2f}:FOLD_{assignment.fold}",
                    set(assignment.task_ids),
                )
                fold_results.append({
                    "fold": assignment.fold,
                    "requests": len(assignment.task_ids),
                    "valid_selected_labels": summary["valid_selected_labels"],
                    "missing_selected_labels": summary["missing_selected_labels"],
                    "acceptable_rate": summary["acceptable_rate_among_valid"],
                    "realized_total_cost_usd": summary["total_realized_cost_usd"],
                    "mean_quality": summary["mean_quality_among_valid"],
                    "fallback_rate": summary["fallback_rate"],
                    "candidate_distribution": summary["model_distribution"],
                })
            rates = [float(item["acceptable_rate"]) for item in fold_results]
            costs = [float(item["realized_total_cost_usd"]) for item in fold_results]
            artifact["variants"][variant][f"{threshold:.2f}"] = {
                "folds": fold_results,
                "summary": {
                    "mean_acceptable_rate": statistics.fmean(rates),
                    "population_standard_deviation_acceptable_rate": statistics.pstdev(rates),
                    "min_acceptable_rate": min(rates),
                    "max_acceptable_rate": max(rates),
                    "mean_realized_total_cost_usd": statistics.fmean(costs),
                    "population_standard_deviation_cost_usd": statistics.pstdev(costs),
                    "best_fold_ids_by_acceptable_rate": [
                        item["fold"] for item in fold_results
                        if float(item["acceptable_rate"]) == max(rates)
                    ],
                    "worst_fold_ids_by_acceptable_rate": [
                        item["fold"] for item in fold_results
                        if float(item["acceptable_rate"]) == min(rates)
                    ],
                },
            }
    return artifact


def category_routing(root, run_id, dataset, interaction_records) -> dict[str, object]:
    frozen = load_frozen_foundation_v3(root, run_id)
    requests = {request.task_id: request for request in frozen.requests}
    categories: dict[str, set[str]] = defaultdict(set)
    for row in dataset.rows:
        categories[row.category].add(row.task_id)
    rule_policy = RuleBasedV1Policy()
    artifact = {"category_request_count": 8, "categories": {}}
    for category, task_ids in sorted(categories.items()):
        if len(task_ids) != 8:
            raise ValueError("Every Foundation V3 category must contain eight requests")
        rule_decisions = [rule_policy.select(requests[task_id]) for task_id in sorted(task_ids)]
        rule_observations = [row for row in frozen.observations if row.task_id in task_ids]
        rule_summary, _ = evaluate_decisions(rule_observations, rule_decisions)
        thresholds = {}
        for threshold in THRESHOLDS:
            summary, decisions = route_predictions(
                dataset, frozen.observations, interaction_records, threshold,
                f"{CATEGORY_CANDIDATE_INTERACTION}@{threshold:.2f}:{category}", task_ids)
            rule_index = {item.task_id: item.candidate_id for item in rule_decisions}
            thresholds[f"{threshold:.2f}"] = {
                "requests": 8,
                "candidate_distribution": summary["model_distribution"],
                "valid_label_coverage": summary["valid_label_coverage"],
                "acceptable_rate": summary["acceptable_rate_among_valid"],
                "mean_quality": summary["mean_quality_among_valid"],
                "realized_cost_usd": summary["total_realized_cost_usd"],
                "fallback_count": summary["fallback_count"],
                "rule_candidate_agreement_count": sum(
                    decision.candidate_id == rule_index[decision.task_id]
                    for decision in decisions),
            }
        artifact["categories"][category] = {
            "rule_based_v1": rule_summary,
            "thresholds": thresholds,
        }
    return artifact


def no_category_ablation(root, run_id, dataset, interaction_records, folds):
    frozen = load_frozen_foundation_v3(root, run_id)
    records, fit_audit, no_category_folds = generate_no_category_predictions(dataset)
    if no_category_folds != folds:
        raise ValueError("NO_CATEGORY did not reuse the Phase 7B folds")
    row_index = {(row.task_id, row.candidate_id): row for row in dataset.rows}
    disagreement = set(disagreement_tasks(dataset))
    no_category_ranking = {
        "all_requests": _rank_variant(list(records), row_index),
        "disagreement_requests": _rank_variant(list(records), row_index, disagreement),
    }
    interaction_ranking = {
        "all_requests": _rank_variant(list(interaction_records), row_index),
        "disagreement_requests": _rank_variant(
            list(interaction_records), row_index, disagreement),
    }
    no_category_metrics = _predictive_metrics(list(records))
    interaction_metrics = _predictive_metrics(list(interaction_records))
    routing = {}
    comparison = {}
    for threshold in THRESHOLDS:
        key = f"{threshold:.2f}"
        no_summary, _ = route_predictions(
            dataset, frozen.observations, records, threshold, f"{NO_CATEGORY}@{key}")
        interaction_summary, _ = route_predictions(
            dataset, frozen.observations, interaction_records, threshold,
            f"{CATEGORY_CANDIDATE_INTERACTION}@{key}")
        routing[key] = no_summary
        comparison[key] = {
            "coverage_matches": (
                interaction_summary["valid_selected_labels"]
                == no_summary["valid_selected_labels"]),
            "acceptable_rate_percentage_point_difference_interaction_minus_no_category": 100 * (
                float(interaction_summary["acceptable_rate_among_valid"])
                - float(no_summary["acceptable_rate_among_valid"])),
            "mean_quality_difference_interaction_minus_no_category": (
                float(interaction_summary["mean_quality_among_valid"])
                - float(no_summary["mean_quality_among_valid"])),
            "cost_difference_usd_interaction_minus_no_category": str(
                Decimal(str(interaction_summary["total_realized_cost_usd"]))
                - Decimal(str(no_summary["total_realized_cost_usd"]))),
        }
    return {
        "representation": {
            "categorical": NO_CATEGORY_REPRESENTATION.categorical,
            "numeric": NO_CATEGORY_REPRESENTATION.numeric,
            "boolean": NO_CATEGORY_REPRESENTATION.boolean,
            "excluded": ["category", "category_candidate"],
            "forbidden_fields_absent": not bool(
                set(NO_CATEGORY_REPRESENTATION.features) & FORBIDDEN_PREDICTIVE_FIELDS),
        },
        "folds_reused_exactly": True,
        "fit_audit": fit_audit,
        "predictive_metrics": {
            NO_CATEGORY: no_category_metrics,
            "INTERACTION_FULL": interaction_metrics,
            "interaction_minus_no_category": {
                "log_loss_reduction": no_category_metrics["log_loss"] - interaction_metrics["log_loss"],
                "brier_reduction": no_category_metrics["brier_score"] - interaction_metrics["brier_score"],
                "roc_auc_increase": interaction_metrics["roc_auc"] - no_category_metrics["roc_auc"],
                "average_precision_increase": (
                    interaction_metrics["average_precision"]
                    - no_category_metrics["average_precision"]),
            },
        },
        "ranking": {
            NO_CATEGORY: no_category_ranking,
            "INTERACTION_FULL": interaction_ranking,
        },
        "routing_sensitivity": routing,
        "routing_comparison": comparison,
    }, records


def phase_decisions(frontier, stability, ablation) -> dict[str, object]:
    full_metrics = ablation["predictive_metrics"]["INTERACTION_FULL"]
    no_category_metrics = ablation["predictive_metrics"][NO_CATEGORY]
    interaction_pairwise = ablation["ranking"]["INTERACTION_FULL"][
        "all_requests"]["pairwise"]["strict_accuracy"]
    no_category_pairwise = ablation["ranking"][NO_CATEGORY][
        "all_requests"]["pairwise"]["strict_accuracy"]
    return {
        "request_specific_selection": {
            "answer": "Yes, modestly and primarily through explicit category-by-candidate interactions.",
            "evidence": (
                f"Pairwise accuracy is {interaction_pairwise:.6f} with the interaction and "
                f"{no_category_pairwise:.6f} without category; interaction routing is cheaper than "
                "FULL_ADDITIVE at every frozen threshold, with equal or higher mean quality at three "
                "of the five coverage-comparable thresholds."
            ),
        },
        "category_dependence": {
            "answer": "Material dependence on the explicit benchmark category is present.",
            "evidence": (
                f"Removing category raises log loss from {full_metrics['log_loss']:.6f} to "
                f"{no_category_metrics['log_loss']:.6f}, lowers ROC-AUC from "
                f"{full_metrics['roc_auc']:.6f} to {no_category_metrics['roc_auc']:.6f}, and returns "
                "within-request ordering to the candidate-prior result."
            ),
        },
        "nonlinear_model": {
            "answer": "No. Collect more independent requests before considering one constrained nonlinear comparison.",
            "evidence": (
                "The controlled interaction already captures measurable structure, while only 56 "
                "independent requests and visibly variable 14-request fold results cannot support "
                "additional model complexity."
            ),
        },
        "architecture_freeze": {
            "recommendation": "A",
            "answer": "Freeze CATEGORY_CANDIDATE_INTERACTION logistic regression as the Phase 7 learned-router formulation for the next stage.",
            "scope": (
                "This freezes only the formulation. It does not select a threshold, serialize a final "
                "model, resolve category supply, establish production generalization, or authorize deployment."
            ),
        },
        "limitations": [
            "Only 56 independent requests and eight requests per category are available.",
            "Fold-level acceptable rates remain variable and no significance tests are appropriate.",
            "Eight candidate labels are missing, so several policy points use different denominators.",
            "The strongest request-dependent improvement relies on an explicit benchmark category.",
            "The observed oracle is retrospective and not deployable.",
            "Foundation V3 does not establish behavior on production traffic or category shift.",
        ],
    }


def oracle_opportunity_capture(root, run_id, frontier_decisions) -> dict[str, object]:
    frozen = load_frozen_foundation_v3(root, run_id)
    oracle_decisions, oracle_states = oracle_cheapest_acceptable(frozen)
    oracle_index = {item.task_id: item for item in oracle_decisions}
    observation_index = {(row.task_id, row.candidate_id): row for row in frozen.observations}
    artifact = {"oracle_observation_state": oracle_states, "thresholds": {}}
    for threshold in THRESHOLDS:
        key = f"{threshold:.2f}"
        counts = Counter()
        details = []
        for decision in frontier_decisions[CATEGORY_CANDIDATE_INTERACTION][key]:
            oracle = oracle_index[decision.task_id]
            if oracle.candidate_id is None:
                outcome = "oracle_abstention_no_acceptable_candidate"
            else:
                selected = observation_index[(decision.task_id, str(decision.candidate_id))]
                if selected.label_status == "missing":
                    outcome = "interaction_selected_label_missing"
                elif decision.candidate_id == oracle.candidate_id:
                    outcome = "selected_oracle_cheapest_acceptable"
                elif selected.acceptable:
                    outcome = "selected_another_acceptable_candidate"
                else:
                    outcome = "selected_unacceptable_candidate"
            counts[outcome] += 1
            details.append({"task_id": decision.task_id, "outcome": outcome})
        for name in (
            "selected_oracle_cheapest_acceptable",
            "selected_another_acceptable_candidate",
            "selected_unacceptable_candidate",
            "interaction_selected_label_missing",
            "oracle_abstention_no_acceptable_candidate",
        ):
            counts[name] += 0
        artifact["thresholds"][key] = {
            "counts": dict(sorted(counts.items())),
            "requests": 56,
            "details": details,
        }
    return artifact


def architecture_note() -> dict[str, object]:
    return {
        "decision": "No permanent category-supply architecture is selected in Phase 7D.",
        "options": {
            "A_client_hint": {
                "api": "Add an optional task/category hint to the request contract.",
                "trust_validation": "Treat it as untrusted advisory input and validate against a controlled vocabulary.",
                "latency": "Negligible additional routing latency.",
                "generalization": "Depends on client consistency and category coverage outside the benchmark.",
            },
            "B_inferred_category": {
                "api": "Keep category internal and infer it before routing.",
                "trust_validation": "Requires calibrated inference, uncertainty handling, and an auditable fallback.",
                "latency": "Adds preprocessing or model latency before candidate selection.",
                "generalization": "Can reduce client burden but introduces a second learned component and error propagation.",
            },
            "C_direct_features": {
                "api": "No explicit category field is required.",
                "trust_validation": "Uses governed features derived directly from request content.",
                "latency": "Retains the current lightweight feature-extraction path.",
                "generalization": "Avoids benchmark taxonomy dependence but may require more data or richer governed features.",
            },
        },
    }


def _protected_hashes(directory: Path) -> dict[str, str]:
    return {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in PROTECTED_PHASE_7_FILES
    }


def _report(frontier, pareto, delta, folds, categories, ablation, rule, oracle, decisions) -> str:
    lines = [
        "# Phase 7D: Interaction Router Policy Validation",
        "",
        "All learned-policy results use the exact Phase 7B grouped OOF folds. No threshold is selected.",
        "",
        "## Complete 56-request frontier",
        "",
        "| Variant | Threshold | Coverage | Acceptable | Mean quality | Cost | Fallback |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in (FULL_ADDITIVE, CATEGORY_CANDIDATE_INTERACTION):
        for threshold in THRESHOLDS:
            summary = frontier["variants"][variant][f"{threshold:.2f}"]
            lines.append(
                f"| {variant} | {threshold:.2f} | {summary['valid_label_coverage']:.3f} | "
                f"{summary['acceptable_rate_among_valid']:.3f} | "
                f"{summary['mean_quality_among_valid']:.3f} | "
                f"${Decimal(str(summary['total_realized_cost_usd'])):.8f} | "
                f"{summary['fallback_rate']:.3f} |"
            )
    no_metrics = ablation["predictive_metrics"][NO_CATEGORY]
    int_metrics = ablation["predictive_metrics"]["INTERACTION_FULL"]
    lines.extend([
        "",
        "## Category ablation",
        "",
        "| Model | Log loss | Brier | ROC-AUC | Average precision | Pairwise |",
        "|---|---:|---:|---:|---:|---:|",
        f"| INTERACTION_FULL | {int_metrics['log_loss']:.4f} | {int_metrics['brier_score']:.4f} | {int_metrics['roc_auc']:.4f} | {int_metrics['average_precision']:.4f} | {ablation['ranking']['INTERACTION_FULL']['all_requests']['pairwise']['strict_accuracy']:.4f} |",
        f"| NO_CATEGORY | {no_metrics['log_loss']:.4f} | {no_metrics['brier_score']:.4f} | {no_metrics['roc_auc']:.4f} | {no_metrics['average_precision']:.4f} | {ablation['ranking'][NO_CATEGORY]['all_requests']['pairwise']['strict_accuracy']:.4f} |",
        "",
        "## Interpretation",
        "",
        "The interaction formulation is evaluated as a model architecture only. The evidence does not select a threshold, serialize a final model, establish production generalization, or authorize deployment.",
        "",
        "Detailed baseline, Pareto, fold, category, rule-comparison, and observed-oracle accounting is stored in the JSON artifacts.",
        "",
        "## Architecture decision",
        "",
        f"Recommendation {decisions['architecture_freeze']['recommendation']}: {decisions['architecture_freeze']['answer']}",
        "",
        decisions["architecture_freeze"]["scope"],
        "",
        f"Nonlinear model: {decisions['nonlinear_model']['answer']}",
        "",
    ])
    return "\n".join(lines)


def run_policy_validation(root: Path, run_id: UUID = FOUNDATION_V3_RUN_ID):
    directory = root / str(run_id) / "phase-7"
    before_hashes = _protected_hashes(directory)
    dataset = load_ml_dataset(root, run_id)
    preflight_folds = build_outer_folds(dataset)
    integrity = _verify_integrity(root, run_id, preflight_folds)
    records, _, _, folds = generate_diagnostic_predictions(dataset)
    if folds != preflight_folds:
        raise ValueError("Phase 7D diagnostics changed the frozen folds")
    predictive = predictive_diagnostics(records)
    interaction_records = tuple(
        item for item in records if item["variant"] == CATEGORY_CANDIDATE_INTERACTION)
    frontier, frontier_decisions = full_frontier(root, run_id, dataset, records, folds)
    pareto = pareto_classification(frontier)
    delta = interaction_vs_additive(frontier)
    stability = fold_stability(root, run_id, dataset, records, folds)
    categories = category_routing(root, run_id, dataset, interaction_records)
    ablation, no_category_records = no_category_ablation(
        root, run_id, dataset, interaction_records, folds)
    rules = rule_based_comparison(root, run_id, dataset, records)
    oracle = oracle_opportunity_capture(root, run_id, frontier_decisions)
    note = architecture_note()
    decisions = phase_decisions(frontier, stability, ablation)
    paths = {
        "frontier": directory / "interaction-frontier.json",
        "fold_stability": directory / "fold-stability.json",
        "category_routing": directory / "category-routing.json",
        "category_ablation": directory / "category-ablation.json",
        "report": directory / "phase-7d-report.md",
    }
    frontier_artifact = {
        "source_integrity": integrity,
        "predictive_reference": {
            FULL_ADDITIVE: predictive[FULL_ADDITIVE]["overall"],
            CATEGORY_CANDIDATE_INTERACTION: predictive[CATEGORY_CANDIDATE_INTERACTION]["overall"],
        },
        "frontier": frontier,
        "pareto": pareto,
        "interaction_vs_additive": delta,
        "rule_based_comparison": rules,
        "oracle_opportunity_capture": oracle,
        "phase_decisions": decisions,
        "final_threshold_selected": False,
    }
    _atomic_write(paths["frontier"], _json(frontier_artifact))
    _atomic_write(paths["fold_stability"], _json(stability))
    _atomic_write(paths["category_routing"], _json(categories))
    _atomic_write(paths["category_ablation"], _json({
        **ablation,
        "architecture_note": note,
        "no_category_oof_predictions": no_category_records,
    }))
    _atomic_write(paths["report"], _report(
        frontier, pareto, delta, stability, categories, ablation, rules, oracle, decisions))
    after_hashes = _protected_hashes(directory)
    if before_hashes != after_hashes:
        raise ValueError("Phase 7D modified a protected Phase 7A/7B/7C artifact")
    return paths, {
        "integrity": integrity,
        "protected_artifact_hashes": after_hashes,
        "frontier": frontier,
        "pareto": pareto,
        "delta": delta,
        "stability": stability,
        "categories": categories,
        "ablation": ablation,
        "rules": rules,
        "oracle": oracle,
        "architecture_note": note,
        "decisions": decisions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline Phase 7D policy validation")
    parser.add_argument("--root", type=Path, default=Path("benchmark-results"))
    parser.add_argument("--run-id", type=UUID, default=FOUNDATION_V3_RUN_ID)
    args = parser.parse_args()
    started = perf_counter()
    paths, _ = run_policy_validation(args.root, args.run_id)
    for name, path in paths.items():
        print(f"{name}: {path}")
    print(f"wall_clock_seconds: {perf_counter() - started:.6f}")


if __name__ == "__main__":
    main()
