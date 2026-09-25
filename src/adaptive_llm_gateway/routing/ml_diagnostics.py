"""Offline Phase 7C diagnostics for the first supervised routing experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from uuid import UUID

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .analysis import (
    FOUNDATION_V2_SHA256,
    FOUNDATION_V3_PROTOCOL_SHA256,
    FOUNDATION_V3_RUN_ID,
    FOUNDATION_V3_SHA256,
    GEMINI,
    RULE_BASED_V1_RULES,
    RoutingDecision,
    RuleBasedV1Policy,
    evaluate_decisions,
    load_frozen_foundation_v3,
)
from .ml_experiment import (
    DESIGN_SHA256,
    LOGREG_UNWEIGHTED,
    THRESHOLDS,
    _atomic_write,
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
    PREDICTIVE_FEATURES,
    RANDOM_STATE,
    MLExperimentDataset,
    MLExperimentRow,
    build_outer_folds,
    load_ml_dataset,
)

CANDIDATE_ONLY = "CANDIDATE_ONLY"
REQUEST_ONLY = "REQUEST_ONLY"
FULL_ADDITIVE = "FULL_ADDITIVE"
CATEGORY_CANDIDATE_INTERACTION = "CATEGORY_CANDIDATE_INTERACTION"
DIAGNOSTIC_VARIANTS = (
    CANDIDATE_ONLY,
    REQUEST_ONLY,
    FULL_ADDITIVE,
    CATEGORY_CANDIDATE_INTERACTION,
)
INTERACTION_FEATURE = "category_candidate"
REQUEST_NUMERIC_FEATURES = (
    "prompt_characters",
    "approximate_input_tokens",
    "requested_max_output_tokens",
    "constraint_indicator_count",
    "reasoning_indicator_count",
)
REQUEST_BOOLEAN_FEATURES = (
    "contains_code",
    "requests_structured_output",
)


@dataclass(frozen=True)
class FeatureRepresentation:
    categorical: tuple[str, ...]
    numeric: tuple[str, ...] = ()
    boolean: tuple[str, ...] = ()

    @property
    def features(self) -> tuple[str, ...]:
        return self.categorical + self.numeric + self.boolean


REPRESENTATIONS = {
    CANDIDATE_ONLY: FeatureRepresentation(categorical=("candidate_id",)),
    REQUEST_ONLY: FeatureRepresentation(
        categorical=("category",),
        numeric=REQUEST_NUMERIC_FEATURES,
        boolean=REQUEST_BOOLEAN_FEATURES,
    ),
    FULL_ADDITIVE: FeatureRepresentation(
        categorical=CATEGORICAL_FEATURES,
        numeric=NUMERIC_FEATURES,
        boolean=BOOLEAN_FEATURES,
    ),
    CATEGORY_CANDIDATE_INTERACTION: FeatureRepresentation(
        categorical=CATEGORICAL_FEATURES + (INTERACTION_FEATURE,),
        numeric=NUMERIC_FEATURES,
        boolean=BOOLEAN_FEATURES,
    ),
}


def _feature_value(row: MLExperimentRow, name: str):
    if name == INTERACTION_FEATURE:
        return f"{row.category}::{row.candidate_id}"
    return row.features[name]


def diagnostic_feature_matrix(
    rows: list[MLExperimentRow] | tuple[MLExperimentRow, ...],
    variant: str,
) -> np.ndarray:
    representation = REPRESENTATIONS[variant]
    return np.asarray([
        [_feature_value(row, name) for name in representation.features]
        for row in rows
    ], dtype=object)


def _as_float(values):
    return values.astype(float)


def build_diagnostic_pipeline(variant: str) -> Pipeline:
    representation = REPRESENTATIONS[variant]
    categorical_end = len(representation.categorical)
    numeric_end = categorical_end + len(representation.numeric)
    transformers = []
    if representation.categorical:
        transformers.append((
            "categorical",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            list(range(categorical_end)),
        ))
    if representation.numeric:
        transformers.append((
            "numeric", StandardScaler(), list(range(categorical_end, numeric_end))))
    if representation.boolean:
        transformers.append((
            "boolean",
            FunctionTransformer(_as_float, feature_names_out="one-to-one"),
            list(range(numeric_end, len(representation.features))),
        ))
    return Pipeline((
        ("preprocess", ColumnTransformer(tuple(transformers), remainder="drop")),
        ("classifier", LogisticRegression(
            l1_ratio=0.0,
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            class_weight=None,
            random_state=RANDOM_STATE,
        )),
    ))


def generate_diagnostic_predictions(dataset: MLExperimentDataset):
    """Generate grouped OOF predictions for all four predeclared representations."""
    folds = build_outer_folds(dataset)
    task_to_fold = {
        task_id: assignment.fold
        for assignment in folds for task_id in assignment.task_ids
    }
    records: list[dict[str, object]] = []
    fit_audit: dict[str, list[dict[str, object]]] = {}
    coefficients: dict[str, list[dict[str, object]]] = {}
    for variant in DIAGNOSTIC_VARIANTS:
        variant_records: list[dict[str, object]] = []
        variant_audit = []
        variant_coefficients = []
        representation = REPRESENTATIONS[variant]
        for assignment in folds:
            test_tasks = set(assignment.task_ids)
            all_training = [row for row in dataset.rows if row.task_id not in test_tasks]
            training = [row for row in all_training if row.label_status == "valid"]
            testing = [row for row in dataset.rows if row.task_id in test_tasks]
            train_tasks = {row.task_id for row in all_training}
            if train_tasks & test_tasks:
                raise ValueError("Outer grouped fold leaked a request")
            pipeline = build_diagnostic_pipeline(variant)
            pipeline.fit(
                diagnostic_feature_matrix(training, variant),
                np.asarray([int(bool(row.acceptable)) for row in training], dtype=int),
            )
            probabilities = pipeline.predict_proba(
                diagnostic_feature_matrix(testing, variant))[:, 1]
            for row, probability in zip(testing, probabilities):
                variant_records.append({
                    "variant": variant,
                    "task_id": row.task_id,
                    "category": row.category,
                    "candidate_id": row.candidate_id,
                    "fold": assignment.fold,
                    "predicted_probability": float(probability),
                    "label_status": row.label_status,
                    "target": row.acceptable,
                })
            preprocessor = pipeline.named_steps["preprocess"]
            names = preprocessor.get_feature_names_out(representation.features)
            weights = pipeline.named_steps["classifier"].coef_[0]
            ordered = sorted(zip(names.tolist(), weights.tolist()), key=lambda item: item[1])
            interaction_terms = [
                {"feature": name, "coefficient": coefficient}
                for name, coefficient in ordered
                if INTERACTION_FEATURE in name
            ]
            variant_coefficients.append({
                "fold": assignment.fold,
                "largest_negative": [
                    {"feature": name, "coefficient": value}
                    for name, value in ordered[:10]
                ],
                "largest_positive": [
                    {"feature": name, "coefficient": value}
                    for name, value in reversed(ordered[-10:])
                ],
                "category_candidate_interactions": interaction_terms,
            })
            variant_audit.append({
                "fold": assignment.fold,
                "train_groups": len(train_tasks),
                "test_groups": len(test_tasks),
                "valid_fit_rows": len(training),
                "missing_rows_excluded_from_fit": len(all_training) - len(training),
                "test_rows_predicted": len(testing),
                "train_task_ids": sorted(train_tasks),
                "test_task_ids": sorted(test_tasks),
            })
        variant_records.sort(key=lambda item: (item["task_id"], item["candidate_id"]))
        if len(variant_records) != 224 or len({
            (item["task_id"], item["candidate_id"]) for item in variant_records
        }) != 224:
            raise ValueError("Each diagnostic variant must have 224 unique OOF predictions")
        if any(task_to_fold[str(item["task_id"])] != item["fold"] for item in variant_records):
            raise ValueError("Diagnostic OOF fold differs from frozen assignment")
        records.extend(variant_records)
        fit_audit[variant] = variant_audit
        coefficients[variant] = variant_coefficients
    return tuple(records), fit_audit, coefficients, folds


def predictive_diagnostics(records: tuple[dict[str, object], ...]) -> dict[str, object]:
    result = {}
    for variant in DIAGNOSTIC_VARIANTS:
        selected = [item for item in records if item["variant"] == variant]
        result[variant] = {
            "overall": _predictive_metrics(selected),
            "by_fold": {
                str(fold): _predictive_metrics([
                    item for item in selected if item["fold"] == fold])
                for fold in range(4)
            },
        }
    candidate = result[CANDIDATE_ONLY]["overall"]
    full = result[FULL_ADDITIVE]["overall"]
    interaction = result[CATEGORY_CANDIDATE_INTERACTION]["overall"]
    result["absolute_improvement_full_vs_candidate_only"] = {
        "log_loss_reduction": candidate["log_loss"] - full["log_loss"],
        "brier_reduction": candidate["brier_score"] - full["brier_score"],
        "roc_auc_increase": full["roc_auc"] - candidate["roc_auc"],
        "average_precision_increase": (
            full["average_precision"] - candidate["average_precision"]),
    }
    result["absolute_improvement_interaction_vs_full"] = {
        "log_loss_reduction": full["log_loss"] - interaction["log_loss"],
        "brier_reduction": full["brier_score"] - interaction["brier_score"],
        "roc_auc_increase": interaction["roc_auc"] - full["roc_auc"],
        "average_precision_increase": (
            interaction["average_precision"] - full["average_precision"]),
    }
    return result


def disagreement_tasks(dataset: MLExperimentDataset) -> tuple[str, ...]:
    by_task: dict[str, list[MLExperimentRow]] = defaultdict(list)
    for row in dataset.rows:
        by_task[row.task_id].append(row)
    tasks = tuple(sorted(
        task_id for task_id, rows in by_task.items()
        if {row.acceptable for row in rows if row.label_status == "valid"} == {False, True}
    ))
    return tasks


def _rank_variant(
    records: list[dict[str, object]],
    row_index: dict[tuple[str, str], MLExperimentRow],
    task_filter: set[str] | None = None,
) -> dict[str, object]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        task_id = str(record["task_id"])
        if task_filter is None or task_id in task_filter:
            grouped[task_id].append(record)
    top_records = []
    correct = ties = incorrect = pair_count = 0
    no_label_disagreement = 0
    probability_gaps = []
    distribution: Counter[str] = Counter()
    selected_qualities = []
    best_qualities = []
    exact_best_quality = 0
    for task_id, candidates in sorted(grouped.items()):
        ranked = sorted(candidates, key=lambda item: (
            -float(item["predicted_probability"]), str(item["candidate_id"])))
        highest = float(ranked[0]["predicted_probability"])
        top_tie_count = sum(
            math.isclose(float(item["predicted_probability"]), highest, abs_tol=1e-12)
            for item in ranked)
        selected = ranked[0]
        distribution[str(selected["candidate_id"])] += 1
        probability_gaps.append(highest - float(ranked[1]["predicted_probability"]))
        observed = row_index[(task_id, str(selected["candidate_id"]))]
        top_records.append({
            "task_id": task_id,
            "candidate_id": selected["candidate_id"],
            "predicted_probability": highest,
            "top_probability_tie_count": top_tie_count,
            "label_status": observed.label_status,
            "acceptable": observed.acceptable,
            "quality_score": observed.quality_score,
        })
        valid = [item for item in candidates if item["label_status"] == "valid"]
        positives = [item for item in valid if item["target"] is True]
        negatives = [item for item in valid if item["target"] is False]
        if not positives or not negatives:
            no_label_disagreement += 1
        for positive in positives:
            for negative in negatives:
                pair_count += 1
                delta = (float(positive["predicted_probability"])
                         - float(negative["predicted_probability"]))
                if math.isclose(delta, 0.0, abs_tol=1e-12):
                    ties += 1
                elif delta > 0:
                    correct += 1
                else:
                    incorrect += 1
        if observed.label_status == "valid" and observed.quality_score is not None:
            valid_rows = [row_index[(task_id, str(item["candidate_id"]))] for item in valid]
            best = max(float(row.quality_score) for row in valid_rows
                       if row.quality_score is not None)
            selected_quality = float(observed.quality_score)
            selected_qualities.append(selected_quality)
            best_qualities.append(best)
            exact_best_quality += selected_quality == best
    valid_top = [item for item in top_records if item["label_status"] == "valid"]
    acceptable = sum(item["acceptable"] is True for item in valid_top)
    return {
        "request_count": len(grouped),
        "top_1_valid_labels": len(valid_top),
        "top_1_valid_label_coverage": len(valid_top) / len(grouped) if grouped else None,
        "top_1_acceptable": acceptable,
        "top_1_acceptable_rate": acceptable / len(valid_top) if valid_top else None,
        "top_1_tied_requests": sum(item["top_probability_tie_count"] > 1 for item in top_records),
        "candidate_ranking_informative": not all(
            item["top_probability_tie_count"] == len(grouped[item["task_id"]])
            for item in top_records
        ),
        "top_1_distribution": dict(sorted(distribution.items())),
        "mean_best_second_probability_gap": statistics.fmean(probability_gaps),
        "pairwise": {
            "pairs": pair_count,
            "correct": correct,
            "ties": ties,
            "incorrect": incorrect,
            "strict_accuracy": correct / pair_count if pair_count else None,
            "tie_adjusted_accuracy": ((correct + 0.5 * ties) / pair_count
                                      if pair_count else None),
        },
        "requests_without_valid_label_disagreement": no_label_disagreement,
        "best_quality_descriptive": {
            "valid_top_1": len(selected_qualities),
            "exact_best_quality_count": exact_best_quality,
            "exact_best_quality_rate": (exact_best_quality / len(selected_qualities)
                                        if selected_qualities else None),
            "mean_selected_quality": (statistics.fmean(selected_qualities)
                                      if selected_qualities else None),
            "mean_best_available_quality": (statistics.fmean(best_qualities)
                                            if best_qualities else None),
            "mean_quality_regret": (statistics.fmean(
                best - selected for best, selected in zip(best_qualities, selected_qualities))
                if selected_qualities else None),
        },
        "top_1_records": top_records,
    }


def within_request_diagnostics(
    dataset: MLExperimentDataset,
    records: tuple[dict[str, object], ...],
) -> dict[str, object]:
    row_index = {(row.task_id, row.candidate_id): row for row in dataset.rows}
    disagreement = set(disagreement_tasks(dataset))
    artifact: dict[str, object] = {
        "disagreement_request_count": len(disagreement),
        "disagreement_task_ids": sorted(disagreement),
        "no_disagreement_request_count": 56 - len(disagreement),
        "variants": {},
    }
    for variant in DIAGNOSTIC_VARIANTS:
        selected = [item for item in records if item["variant"] == variant]
        artifact["variants"][variant] = {
            "all_requests": _rank_variant(selected, row_index),
            "disagreement_requests": _rank_variant(selected, row_index, disagreement),
        }
    return artifact


def candidate_probability_diagnostics(
    records: tuple[dict[str, object], ...],
) -> dict[str, object]:
    artifact = {}
    for variant in DIAGNOSTIC_VARIANTS:
        candidates: dict[str, list[dict[str, object]]] = defaultdict(list)
        for record in records:
            if record["variant"] == variant:
                candidates[str(record["candidate_id"])].append(record)
        artifact[variant] = {}
        for candidate_id, items in sorted(candidates.items()):
            values = np.asarray([float(item["predicted_probability"]) for item in items])
            valid = [item for item in items if item["label_status"] == "valid"]
            artifact[variant][candidate_id] = {
                "requests": len(items),
                "valid_labels": len(valid),
                "observed_positive_rate": (sum(item["target"] is True for item in valid) / len(valid)),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "p25": float(np.percentile(values, 25)),
                "p75": float(np.percentile(values, 75)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "standard_deviation": float(np.std(values)),
                "range": float(np.max(values) - np.min(values)),
            }
    movement = {}
    for candidate_id in sorted(artifact[FULL_ADDITIVE]):
        candidate = artifact[CANDIDATE_ONLY][candidate_id]
        full = artifact[FULL_ADDITIVE][candidate_id]
        movement[candidate_id] = {
            "candidate_only_standard_deviation": candidate["standard_deviation"],
            "full_additive_standard_deviation": full["standard_deviation"],
            "standard_deviation_increase": (
                full["standard_deviation"] - candidate["standard_deviation"]),
            "candidate_only_range": candidate["range"],
            "full_additive_range": full["range"],
            "range_increase": full["range"] - candidate["range"],
        }
    return {"by_variant_and_candidate": artifact, "full_vs_candidate_only_movement": movement}


def disagreement_routing_diagnostics(
    root: Path,
    run_id: UUID,
    dataset: MLExperimentDataset,
    records: tuple[dict[str, object], ...],
) -> dict[str, object]:
    frozen = load_frozen_foundation_v3(root, run_id)
    tasks = set(disagreement_tasks(dataset))
    observations = [row for row in frozen.observations if row.task_id in tasks]
    row_index = {(row.task_id, row.candidate_id): row for row in dataset.rows}
    artifact = {"request_count": len(tasks), "threshold_grid": list(THRESHOLDS), "variants": {}}
    for variant in (FULL_ADDITIVE, CATEGORY_CANDIDATE_INTERACTION):
        grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
        for record in records:
            if record["variant"] == variant and record["task_id"] in tasks:
                grouped[str(record["task_id"])].append(record)
        thresholds = {}
        for threshold in THRESHOLDS:
            decisions = []
            fallback_count = 0
            for task_id, candidates in sorted(grouped.items()):
                selected, fallback = select_cost_aware_candidate(
                    task_id, candidates, row_index, threshold)
                fallback_count += int(fallback)
                decisions.append(RoutingDecision(
                    policy=f"{variant}@{threshold:.2f}:DISAGREEMENT",
                    task_id=task_id,
                    candidate_id=str(selected["candidate_id"]),
                    reason=("highest probability fallback" if fallback
                            else "lowest projected cost above threshold"),
                ))
            summary, _ = evaluate_decisions(observations, decisions)
            summary["fallback_count"] = fallback_count
            summary["fallback_rate"] = fallback_count / len(tasks)
            thresholds[f"{threshold:.2f}"] = summary
        artifact["variants"][variant] = thresholds
    return artifact


def _rule_branch(request) -> tuple[str, str]:
    feature = request.features
    if feature.category == "coding" or feature.contains_code:
        return "1_coding_or_code", RULE_BASED_V1_RULES[0]
    if feature.category == "reasoning" or feature.reasoning_indicator_count >= 3:
        return "2_reasoning", RULE_BASED_V1_RULES[1]
    if feature.requests_structured_output:
        return "3_structured_output", RULE_BASED_V1_RULES[2]
    if feature.category == "qa":
        return "4_qa", RULE_BASED_V1_RULES[3]
    return "5_default", RULE_BASED_V1_RULES[4]


def _compare_choices(rule_decisions, ml_decisions, observation_index) -> dict[str, object]:
    rule_by_task = {item.task_id: item for item in rule_decisions}
    ml_by_task = {item.task_id: item for item in ml_decisions}
    if set(rule_by_task) != set(ml_by_task):
        raise ValueError("ML and rule comparisons must cover the same requests")
    buckets = Counter()
    agreement = 0
    details = []
    for task_id in sorted(rule_by_task):
        rule_id = str(rule_by_task[task_id].candidate_id)
        ml_id = str(ml_by_task[task_id].candidate_id)
        if rule_id == ml_id:
            agreement += 1
            continue
        rule = observation_index[(task_id, rule_id)]
        ml = observation_index[(task_id, ml_id)]
        if rule.label_status == "missing" or ml.label_status == "missing":
            bucket = "one_or_both_labels_missing"
        elif rule.acceptable and not ml.acceptable:
            bucket = "rule_acceptable_ml_unacceptable"
        elif ml.acceptable and not rule.acceptable:
            bucket = "ml_acceptable_rule_unacceptable"
        elif rule.acceptable and ml.acceptable:
            bucket = "both_acceptable"
        else:
            bucket = "neither_acceptable"
        buckets[bucket] += 1
        details.append({
            "task_id": task_id,
            "rule_candidate_id": rule_id,
            "ml_candidate_id": ml_id,
            "outcome_bucket": bucket,
        })
    for name in (
        "rule_acceptable_ml_unacceptable",
        "ml_acceptable_rule_unacceptable",
        "both_acceptable",
        "neither_acceptable",
        "one_or_both_labels_missing",
    ):
        buckets[name] += 0
    return {
        "requests": len(rule_by_task),
        "agreement_count": agreement,
        "disagreement_count": len(rule_by_task) - agreement,
        "disagreement_outcomes": dict(sorted(buckets.items())),
        "disagreements": details,
    }


def rule_based_comparison(
    root: Path,
    run_id: UUID,
    dataset: MLExperimentDataset,
    records: tuple[dict[str, object], ...],
) -> dict[str, object]:
    frozen = load_frozen_foundation_v3(root, run_id)
    policy = RuleBasedV1Policy()
    rule_decisions = [policy.select(request) for request in frozen.requests]
    decisions_by_task = {item.task_id: item for item in rule_decisions}
    observation_index = {(row.task_id, row.candidate_id): row for row in frozen.observations}
    request_by_task = {request.task_id: request for request in frozen.requests}
    branches: dict[str, list[str]] = defaultdict(list)
    branch_rules = {}
    for request in frozen.requests:
        branch, rule = _rule_branch(request)
        branches[branch].append(request.task_id)
        branch_rules[branch] = rule
    branch_artifact = {}
    for branch, task_ids in sorted(branches.items()):
        task_set = set(task_ids)
        observations = [row for row in frozen.observations if row.task_id in task_set]
        decisions = [decisions_by_task[task_id] for task_id in sorted(task_ids)]
        summary, _ = evaluate_decisions(observations, decisions)
        branch_artifact[branch] = {
            "rule": branch_rules[branch],
            "request_count": len(task_ids),
            "selected_candidate": decisions[0].candidate_id,
            "summary": summary,
        }
    row_index = {(row.task_id, row.candidate_id): row for row in dataset.rows}
    comparisons = {}
    for variant in (FULL_ADDITIVE, CATEGORY_CANDIDATE_INTERACTION):
        grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
        for record in records:
            if record["variant"] == variant:
                grouped[str(record["task_id"])].append(record)
        top_decisions = []
        for task_id, candidates in sorted(grouped.items()):
            selected = min(candidates, key=lambda item: (
                -float(item["predicted_probability"]), str(item["candidate_id"])))
            top_decisions.append(RoutingDecision(
                policy=f"{variant}:TOP_1", task_id=task_id,
                candidate_id=str(selected["candidate_id"]), reason="highest OOF probability"))
        variant_comparison = {
            "top_ranked": _compare_choices(rule_decisions, top_decisions, observation_index),
            "threshold_selected": {},
        }
        for threshold in THRESHOLDS:
            threshold_decisions = []
            for task_id, candidates in sorted(grouped.items()):
                selected, fallback = select_cost_aware_candidate(
                    task_id, candidates, row_index, threshold)
                threshold_decisions.append(RoutingDecision(
                    policy=f"{variant}@{threshold:.2f}", task_id=task_id,
                    candidate_id=str(selected["candidate_id"]),
                    reason="fallback" if fallback else "qualified projected-cost choice"))
            variant_comparison["threshold_selected"][f"{threshold:.2f}"] = _compare_choices(
                rule_decisions, threshold_decisions, observation_index)
        comparisons[variant] = variant_comparison
    return {
        "frozen_rules": list(RULE_BASED_V1_RULES),
        "branch_analysis": branch_artifact,
        "ml_vs_rule": comparisons,
        "request_count": len(request_by_task),
    }


def gemini_diagnostics(
    dataset: MLExperimentDataset,
    records: tuple[dict[str, object], ...],
) -> dict[str, object]:
    gemini_rows = [row for row in dataset.rows if row.candidate_id == GEMINI]
    valid = [row for row in gemini_rows if row.label_status == "valid"]
    artifact: dict[str, object] = {
        "valid_labels": len(valid),
        "positive_labels": sum(row.acceptable is True for row in valid),
        "observed_positive_rate": sum(row.acceptable is True for row in valid) / len(valid),
        "by_variant": {},
    }
    for variant in DIAGNOSTIC_VARIANTS:
        selected = [item for item in records if item["variant"] == variant]
        by_task: dict[str, list[dict[str, object]]] = defaultdict(list)
        for item in selected:
            by_task[str(item["task_id"])].append(item)
        ranks = Counter()
        top_count = 0
        tied_for_top = 0
        gemini_probabilities = []
        for candidates in by_task.values():
            ranked = sorted(candidates, key=lambda item: (
                -float(item["predicted_probability"]), str(item["candidate_id"])))
            rank = next(index for index, item in enumerate(ranked, 1)
                        if item["candidate_id"] == GEMINI)
            ranks[str(rank)] += 1
            gemini = next(item for item in candidates if item["candidate_id"] == GEMINI)
            probability = float(gemini["predicted_probability"])
            gemini_probabilities.append(probability)
            top_probability = max(float(item["predicted_probability"]) for item in candidates)
            if rank == 1:
                top_count += 1
            if math.isclose(probability, top_probability, abs_tol=1e-12):
                tied_for_top += 1
        artifact["by_variant"][variant] = {
            "rank_distribution_stable_tiebreak": dict(sorted(ranks.items())),
            "predicted_top_1_count": top_count,
            "tied_for_top_probability_count": tied_for_top,
            "threshold_exceed_counts": {
                f"{threshold:.2f}": sum(value >= threshold for value in gemini_probabilities)
                for threshold in THRESHOLDS
            },
            "probability_mean": statistics.fmean(gemini_probabilities),
            "probability_min": min(gemini_probabilities),
            "probability_max": max(gemini_probabilities),
        }
    cheapest_count = 0
    by_task: dict[str, list[MLExperimentRow]] = defaultdict(list)
    for row in dataset.rows:
        by_task[row.task_id].append(row)
    for rows in by_task.values():
        acceptable = [row for row in rows if row.label_status == "valid" and row.acceptable]
        if acceptable:
            cheapest = min(acceptable, key=lambda row: (
                row.realized_cost_usd if row.realized_cost_usd is not None else math.inf,
                row.candidate_id,
            ))
            cheapest_count += cheapest.candidate_id == GEMINI
    artifact["cheapest_observed_acceptable_count"] = cheapest_count
    return artifact


def interaction_coefficient_summary(coefficients: dict[str, object]) -> dict[str, object]:
    folds = coefficients[CATEGORY_CANDIDATE_INTERACTION]
    by_feature: dict[str, list[float]] = defaultdict(list)
    for fold in folds:
        for item in fold["category_candidate_interactions"]:
            by_feature[item["feature"]].append(float(item["coefficient"]))
    means = [
        {"feature": feature, "mean_coefficient": statistics.fmean(values),
         "fold_coefficients": values}
        for feature, values in sorted(by_feature.items())
    ]
    ordered = sorted(means, key=lambda item: item["mean_coefficient"])
    return {
        "strongest_negative": ordered[:10],
        "strongest_positive": list(reversed(ordered[-10:])),
        "note": "Fold-local standardized coefficients are descriptive, not causal.",
    }


def decision_framework(metrics, ranking, routing) -> dict[str, object]:
    improvement = metrics["absolute_improvement_full_vs_candidate_only"]
    interaction = metrics["absolute_improvement_interaction_vs_full"]
    full_rank = ranking["variants"][FULL_ADDITIVE]["disagreement_requests"]
    candidate_rank = ranking["variants"][CANDIDATE_ONLY]["disagreement_requests"]
    interaction_rank = ranking["variants"][CATEGORY_CANDIDATE_INTERACTION][
        "disagreement_requests"]
    return {
        "Q1": {
            "answer": "Yes for row-level prediction, but not for candidate ordering in the additive formulation.",
            "evidence": (
                f"FULL_ADDITIVE reduced OOF log loss by {improvement['log_loss_reduction']:.6f} "
                f"and Brier by {improvement['brier_reduction']:.6f}; however, it preserved the "
                "candidate-only within-request ordering on this dataset."
            ),
        },
        "Q2": {
            "answer": "No clear request-dependent ranking benefit beyond global candidate priors.",
            "evidence": (
                f"FULL_ADDITIVE and CANDIDATE_ONLY both achieved disagreement pairwise accuracy "
                f"{full_rank['pairwise']['strict_accuracy']:.6f} versus "
                f"{candidate_rank['pairwise']['strict_accuracy']:.6f}, and both selected Sonnet "
                "as top-1 on all 39 disagreement requests."
            ),
        },
        "Q3": {
            "answer": "Yes, modestly; the interaction helps prediction and pairwise ordering, with mixed threshold effects.",
            "evidence": (
                f"The interaction reduced log loss by {interaction['log_loss_reduction']:.6f}, "
                f"increased ROC-AUC by {interaction['roc_auc_increase']:.6f}, and raised "
                f"disagreement pairwise accuracy from {full_rank['pairwise']['strict_accuracy']:.6f} "
                f"to {interaction_rank['pairwise']['strict_accuracy']:.6f}. It did not improve "
                "disagreement top-1 acceptable rate and changed only one top-1 selection."
            ),
        },
        "Q4": {
            "answer": "RULE_BASED_V1 encodes category-specific candidate choices and avoids paying for Sonnet where cheaper candidates are often sufficient.",
            "evidence": (
                "The additive model applies request effects equally to every candidate, so they change "
                "absolute probabilities without changing candidate order. The frozen rule explicitly "
                "routes coding/reasoning to Sonnet, structured/default work to Luna, and QA to Nemotron."
            ),
        },
        "Q5": {
            "choice": "B",
            "answer": "Keep logistic regression and retain the predeclared category-by-candidate interactions for the next controlled experiment.",
            "evidence": (
                "This is the smallest tested change that improved held-out proper losses, ranking, and "
                "several fixed-threshold disagreement results. More data is still needed before any "
                "production claim; the evidence does not yet justify a nonlinear model."
            ),
        },
    }


def _report(metrics, ranking, probabilities, routing, rules, gemini, interactions, decisions) -> str:
    lines = [
        "# Phase 7C: Learned Router Signal Diagnostics",
        "",
        "All results use the frozen Phase 7B grouped folds and out-of-fold probabilities.",
        "",
        "## Model comparison",
        "",
        "| Variant | Log loss | Brier | ROC-AUC | Avg precision | Top-1 acceptable | Pairwise strict accuracy | Disagreement top-1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in DIAGNOSTIC_VARIANTS:
        metric = metrics[variant]["overall"]
        rank = ranking["variants"][variant]["all_requests"]
        disagree = ranking["variants"][variant]["disagreement_requests"]
        top_rate = (f"{rank['top_1_acceptable_rate']:.4f}"
                    if rank["candidate_ranking_informative"] else "N/A (all tied)")
        pairwise = (f"{rank['pairwise']['strict_accuracy']:.4f}"
                    if rank["candidate_ranking_informative"] else "N/A (all tied)")
        disagreement_rate = (f"{disagree['top_1_acceptable_rate']:.4f}"
                             if disagree["candidate_ranking_informative"]
                             else "N/A (all tied)")
        lines.append(
            f"| {variant} | {metric['log_loss']:.4f} | {metric['brier_score']:.4f} | "
            f"{metric['roc_auc']:.4f} | {metric['average_precision']:.4f} | "
            f"{top_rate} | {pairwise} | {disagreement_rate} |"
        )
    improvement = metrics["absolute_improvement_full_vs_candidate_only"]
    interaction = metrics["absolute_improvement_interaction_vs_full"]
    lines.extend([
        "",
        "## Prior versus request signal",
        "",
        f"FULL_ADDITIVE versus CANDIDATE_ONLY reduces log loss by {improvement['log_loss_reduction']:.4f} and Brier by {improvement['brier_reduction']:.4f}, while increasing ROC-AUC by {improvement['roc_auc_increase']:.4f}.",
        f"The controlled category-by-candidate model changes log loss by a reduction of {interaction['log_loss_reduction']:.4f} and ROC-AUC by {interaction['roc_auc_increase']:+.4f} relative to FULL_ADDITIVE.",
        f"There are {ranking['disagreement_request_count']} disagreement requests and {ranking['no_disagreement_request_count']} requests without valid-label disagreement.",
        "",
        "## Interpretation boundary",
        "",
        "Probability variation is interpreted together with within-request ranking. These diagnostics do not select a production threshold or establish generalization beyond Foundation V3.",
        "",
        f"Gemini is the cheapest observed acceptable candidate on {gemini['cheapest_observed_acceptable_count']} requests.",
        "",
        "Detailed routing sensitivity, probability distributions, rule comparisons, and coefficients are in the JSON artifacts.",
        "",
        "## Decision framework",
        "",
    ])
    for question in ("Q1", "Q2", "Q3", "Q4", "Q5"):
        lines.extend([
            f"### {question}",
            "",
            decisions[question]["answer"],
            "",
            decisions[question]["evidence"],
            "",
        ])
    return "\n".join(lines)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_integrity(root: Path, run_id: UUID, folds) -> dict[str, object]:
    project = root.parent
    paths = {
        "foundation_v2": project / "benchmarks/datasets/foundation-v2.json",
        "foundation_v3": project / "benchmarks/datasets/foundation-v3.json",
        "protocol": project / "benchmarks/protocols/foundation-v3.json",
        "phase_7a_design": root / str(run_id) / "phase-7/ml-router-design.json",
    }
    expected = {
        "foundation_v2": FOUNDATION_V2_SHA256,
        "foundation_v3": FOUNDATION_V3_SHA256,
        "protocol": FOUNDATION_V3_PROTOCOL_SHA256,
        "phase_7a_design": DESIGN_SHA256,
    }
    actual = {name: _sha256(path) for name, path in paths.items()}
    if actual != expected:
        raise ValueError(f"Frozen source hash mismatch: {actual}")
    fold_path = root / str(run_id) / "phase-7/fold-assignments.json"
    stored = json.loads(fold_path.read_text(encoding="utf-8"))
    generated = [assignment.model_dump(mode="json") for assignment in folds]
    if stored["folds"] != generated:
        raise ValueError("Generated folds differ from Phase 7B fold assignments")
    return {
        "hashes": actual,
        "phase_7b_folds_reused_exactly": True,
        "phase_7b_fold_artifact_sha256": _sha256(fold_path),
    }


def _verify_full_additive_matches_phase_7b(root: Path, run_id: UUID, records) -> None:
    path = root / str(run_id) / "phase-7/oof-predictions.jsonl"
    existing = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    expected = {
        (item["task_id"], item["candidate_id"]): float(item["predicted_probability"])
        for item in existing if item["variant"] == LOGREG_UNWEIGHTED
    }
    current = {
        (item["task_id"], item["candidate_id"]): float(item["predicted_probability"])
        for item in records if item["variant"] == FULL_ADDITIVE
    }
    if expected.keys() != current.keys() or any(
        expected[key] != round(current[key], 12) for key in expected
    ):
        raise ValueError("FULL_ADDITIVE does not exactly reproduce Phase 7B unweighted OOF output")


def run_diagnostics(root: Path, run_id: UUID = FOUNDATION_V3_RUN_ID):
    dataset = load_ml_dataset(root, run_id)
    preflight_folds = build_outer_folds(dataset)
    integrity = _verify_integrity(root, run_id, preflight_folds)
    records, fit_audit, coefficients, folds = generate_diagnostic_predictions(dataset)
    if folds != preflight_folds:
        raise ValueError("Diagnostic training changed the preflight fold assignments")
    _verify_full_additive_matches_phase_7b(root, run_id, records)
    predictive = predictive_diagnostics(records)
    ranking = within_request_diagnostics(dataset, records)
    if ranking["disagreement_request_count"] != 39:
        raise ValueError("Frozen data no longer contains the expected 39 disagreement requests")
    probabilities = candidate_probability_diagnostics(records)
    routing = disagreement_routing_diagnostics(root, run_id, dataset, records)
    rules = rule_based_comparison(root, run_id, dataset, records)
    gemini = gemini_diagnostics(dataset, records)
    interactions = interaction_coefficient_summary(coefficients)
    decisions = decision_framework(predictive, ranking, routing)
    output = root / str(run_id) / "phase-7"
    paths = {
        "predictions": output / "diagnostic-predictions.jsonl",
        "metrics": output / "diagnostic-metrics.json",
        "ranking": output / "within-request-ranking.json",
        "rules": output / "rule-based-comparison.json",
        "report": output / "phase-7c-report.md",
    }
    metric_artifact = {
        "source_integrity": integrity,
        "feature_representations": {
            variant: {
                "categorical": representation.categorical,
                "numeric": representation.numeric,
                "boolean": representation.boolean,
                "features": representation.features,
            }
            for variant, representation in REPRESENTATIONS.items()
        },
        "forbidden_predictive_fields": sorted(FORBIDDEN_PREDICTIVE_FIELDS),
        "fit_audit": fit_audit,
        "predictive_metrics": predictive,
        "candidate_probability_diagnostics": probabilities,
        "disagreement_routing_sensitivity": routing,
        "coefficients": {
            FULL_ADDITIVE: coefficients[FULL_ADDITIVE],
            CATEGORY_CANDIDATE_INTERACTION: coefficients[CATEGORY_CANDIDATE_INTERACTION],
            "interaction_summary": interactions,
        },
        "gemini": gemini,
        "decision_framework": decisions,
        "final_threshold_selected": False,
    }
    _atomic_write(paths["predictions"], "".join(
        json.dumps(_normalized(record), sort_keys=True, separators=(",", ":")) + "\n"
        for record in sorted(records, key=lambda item: (
            str(item["variant"]), str(item["task_id"]), str(item["candidate_id"])))
    ))
    _atomic_write(paths["metrics"], _json(metric_artifact))
    _atomic_write(paths["ranking"], _json(ranking))
    _atomic_write(paths["rules"], _json(rules))
    _atomic_write(paths["report"], _report(
        predictive, ranking, probabilities, routing, rules, gemini, interactions, decisions))
    return paths, {
        "integrity": integrity,
        "predictive": predictive,
        "ranking": ranking,
        "probabilities": probabilities,
        "routing": routing,
        "rules": rules,
        "gemini": gemini,
        "interactions": interactions,
        "decisions": decisions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline Phase 7C signal diagnostics")
    parser.add_argument("--root", type=Path, default=Path("benchmark-results"))
    parser.add_argument("--run-id", type=UUID, default=FOUNDATION_V3_RUN_ID)
    args = parser.parse_args()
    started = perf_counter()
    paths, _ = run_diagnostics(args.root, args.run_id)
    for name, path in paths.items():
        print(f"{name}: {path}")
    print(f"wall_clock_seconds: {perf_counter() - started:.6f}")


if __name__ == "__main__":
    main()
