"""First grouped, offline logistic-regression routing experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from time import perf_counter
from uuid import UUID, uuid4

import numpy as np
import sklearn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .analysis import (
    ALWAYS_CHEAPEST,
    ALWAYS_STRONGEST,
    ORACLE_CHEAPEST_ACCEPTABLE,
    RANDOM_SEEDED,
    RULE_BASED_V1,
    AlwaysCheapestPolicy,
    FixedCandidatePolicy,
    RandomSeededPolicy,
    RoutingDecision,
    RuleBasedV1Policy,
    evaluate_decisions,
    load_frozen_foundation_v3,
    oracle_cheapest_acceptable,
    resolve_strongest,
    FOUNDATION_V2_SHA256,
    FOUNDATION_V3_PROTOCOL_SHA256,
    FOUNDATION_V3_RUN_ID,
    FOUNDATION_V3_SHA256,
)
from .ml_features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    BOOLEAN_FEATURES,
    FORBIDDEN_PREDICTIVE_FIELDS,
    PREDICTIVE_FEATURES,
    RANDOM_STATE,
    MLExperimentDataset,
    MLExperimentRow,
    build_outer_folds,
    build_pipeline,
    feature_matrix,
    load_ml_dataset,
)

LOGREG_UNWEIGHTED = "LOGREG_UNWEIGHTED"
LOGREG_BALANCED = "LOGREG_BALANCED"
FOLD_LOCAL_ALWAYS_STRONGEST = "FOLD_LOCAL_ALWAYS_STRONGEST"
VARIANTS = {
    LOGREG_UNWEIGHTED: None,
    LOGREG_BALANCED: "balanced",
}
THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
CALIBRATION_EDGES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
DESIGN_SHA256 = "abb18acee90efd773a55eedd0f75ae69cb487b8b2f667c8d1aae336b0e8e434b"


def _atomic_write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _normalized(value):
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return round(value, 12)
    if isinstance(value, dict):
        return {key: _normalized(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_normalized(item) for item in value]
    return value


def _json(value) -> str:
    return json.dumps(_normalized(value), indent=2, sort_keys=True) + "\n"


def _predictive_metrics(records: list[dict[str, object]]) -> dict[str, object]:
    valid = [record for record in records if record["label_status"] == "valid"]
    y = np.asarray([int(bool(record["target"])) for record in valid], dtype=int)
    probability = np.asarray([float(record["predicted_probability"]) for record in valid])
    predicted = (probability >= 0.5).astype(int)
    result: dict[str, object] = {
        "rows": len(valid),
        "positive": int(y.sum()),
        "negative": int(len(y) - y.sum()),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, probability)),
        "accuracy_at_0_5": float(accuracy_score(y, predicted)),
        "precision_at_0_5": float(precision_score(y, predicted, zero_division=0)),
        "recall_at_0_5": float(recall_score(y, predicted, zero_division=0)),
        "f1_at_0_5": float(f1_score(y, predicted, zero_division=0)),
    }
    if len(set(y.tolist())) == 2:
        result["roc_auc"] = float(roc_auc_score(y, probability))
        result["average_precision"] = float(average_precision_score(y, probability))
        result["undefined_metrics"] = []
    else:
        result["roc_auc"] = None
        result["average_precision"] = None
        result["undefined_metrics"] = [
            "roc_auc and average_precision require both target classes in the fold"]
    return result


def _calibration(records: list[dict[str, object]]) -> dict[str, object]:
    valid = [record for record in records if record["label_status"] == "valid"]
    bins = []
    for index, (lower, upper) in enumerate(zip(CALIBRATION_EDGES, CALIBRATION_EDGES[1:])):
        members = [record for record in valid if (
            lower <= float(record["predicted_probability"]) <= upper
            if index == len(CALIBRATION_EDGES) - 2
            else lower <= float(record["predicted_probability"]) < upper
        )]
        bins.append({
            "lower": lower,
            "upper": upper,
            "count": len(members),
            "mean_predicted_probability": (
                statistics.fmean(float(item["predicted_probability"]) for item in members)
                if members else None),
            "observed_positive_rate": (
                statistics.fmean(bool(item["target"]) for item in members)
                if members else None),
        })
    return {
        "bins": bins,
        "calibration_model_fitted": False,
        "slope_intercept": None,
        "note": "Descriptive bins only; sample size does not justify fitting a calibration model.",
    }


def generate_oof_predictions(
    dataset: MLExperimentDataset,
) -> tuple[tuple[dict[str, object], ...], dict[str, object], dict[str, object], tuple]:
    folds = build_outer_folds(dataset)
    task_to_fold = {
        task_id: assignment.fold
        for assignment in folds for task_id in assignment.task_ids
    }
    all_records = []
    coefficient_artifact: dict[str, object] = {}
    fit_audit: dict[str, object] = {}
    for variant, class_weight in VARIANTS.items():
        variant_records = []
        coefficient_by_fold = []
        variant_fit_audit = []
        for assignment in folds:
            test_tasks = set(assignment.task_ids)
            train_rows_all = [row for row in dataset.rows if row.task_id not in test_tasks]
            train_rows = [row for row in train_rows_all if row.label_status == "valid"]
            test_rows = [row for row in dataset.rows if row.task_id in test_tasks]
            train_tasks = {row.task_id for row in train_rows_all}
            if train_tasks & test_tasks:
                raise ValueError("Outer grouped fold leaked a task between train and test")
            pipeline = build_pipeline(class_weight)
            pipeline.fit(
                feature_matrix(train_rows),
                np.asarray([int(bool(row.acceptable)) for row in train_rows], dtype=int),
            )
            probabilities = pipeline.predict_proba(feature_matrix(test_rows))[:, 1]
            for row, probability in zip(test_rows, probabilities):
                variant_records.append({
                    "variant": variant,
                    "task_id": row.task_id,
                    "candidate_id": row.candidate_id,
                    "fold": assignment.fold,
                    "predicted_probability": float(probability),
                    "label_status": row.label_status,
                    "target": row.acceptable,
                })
            preprocessor = pipeline.named_steps["preprocess"]
            names = preprocessor.get_feature_names_out(PREDICTIVE_FEATURES)
            coefficients = pipeline.named_steps["classifier"].coef_[0]
            ordered = sorted(zip(names.tolist(), coefficients.tolist()), key=lambda item: item[1])
            coefficient_by_fold.append({
                "fold": assignment.fold,
                "largest_negative": [
                    {"feature": name, "coefficient": value} for name, value in ordered[:10]],
                "largest_positive": [
                    {"feature": name, "coefficient": value} for name, value in reversed(ordered[-10:])],
            })
            variant_fit_audit.append({
                "fold": assignment.fold,
                "train_groups": len(train_tasks),
                "test_groups": len(test_tasks),
                "valid_fit_rows": len(train_rows),
                "missing_rows_excluded_from_fit": len(train_rows_all) - len(train_rows),
                "test_rows_predicted": len(test_rows),
                "train_task_ids": sorted(train_tasks),
                "test_task_ids": sorted(test_tasks),
                "preprocessor_instance_is_fold_local": True,
            })
        variant_records.sort(key=lambda item: (item["task_id"], item["candidate_id"]))
        if len(variant_records) != 224 or len({
            (item["task_id"], item["candidate_id"]) for item in variant_records}) != 224:
            raise ValueError("Each variant must produce exactly 224 unique OOF probabilities")
        if any(task_to_fold[item["task_id"]] != item["fold"] for item in variant_records):
            raise ValueError("OOF prediction fold does not match the frozen task assignment")
        all_records.extend(variant_records)
        coefficient_artifact[variant] = coefficient_by_fold
        fit_audit[variant] = variant_fit_audit
    return tuple(all_records), coefficient_artifact, fit_audit, folds


def predictive_evaluation(records: tuple[dict[str, object], ...]) -> dict[str, object]:
    artifact = {}
    for variant in VARIANTS:
        selected = [record for record in records if record["variant"] == variant]
        artifact[variant] = {
            "overall": _predictive_metrics(selected),
            "by_fold": {
                str(fold): _predictive_metrics([
                    record for record in selected if record["fold"] == fold])
                for fold in range(4)
            },
            "calibration": _calibration(selected),
        }
    return artifact


def _baseline_evaluation(root: Path, run_id: UUID, folds) -> dict[str, object]:
    frozen = load_frozen_foundation_v3(root, run_id)
    requests = {request.task_id: request for request in frozen.requests}
    baselines = {}
    full_strongest, strongest_details = resolve_strongest(frozen.observations)
    policies = (
        AlwaysCheapestPolicy(),
        FixedCandidatePolicy(full_strongest),
        RandomSeededPolicy(),
        RuleBasedV1Policy(),
    )
    for policy in policies:
        summary, _ = evaluate_decisions(
            frozen.observations, [policy.select(request) for request in frozen.requests])
        baselines[policy.name] = summary
    oracle, oracle_state = oracle_cheapest_acceptable(frozen)
    oracle_summary, _ = evaluate_decisions(frozen.observations, oracle)
    oracle_summary["observation_state"] = oracle_state
    baselines[ORACLE_CHEAPEST_ACCEPTABLE] = oracle_summary

    fold_local_decisions, resolved = fold_local_strongest_decisions(frozen, folds)
    fold_summary, _ = evaluate_decisions(frozen.observations, fold_local_decisions)
    baselines[FOLD_LOCAL_ALWAYS_STRONGEST] = fold_summary
    return {
        "same_request_set": True,
        "request_count": len(requests),
        "full_data_strongest_candidate_descriptive_only": full_strongest,
        "fold_local_strongest_resolution": resolved,
        "policies": baselines,
    }


def fold_local_strongest_decisions(frozen, folds) -> tuple[list[RoutingDecision], dict[str, object]]:
    """Resolve each strongest baseline from outer-training labels only."""
    fold_local_decisions = []
    resolved = {}
    for assignment in folds:
        test_tasks = set(assignment.task_ids)
        strongest, details = resolve_strongest(
            row for row in frozen.observations if row.task_id not in test_tasks)
        resolved[str(assignment.fold)] = {
            "candidate_id": strongest,
            "training_only_candidate_metrics": details,
        }
        for task_id in assignment.task_ids:
            fold_local_decisions.append(RoutingDecision(
                policy=FOLD_LOCAL_ALWAYS_STRONGEST,
                task_id=task_id,
                candidate_id=strongest,
                reason="strongest candidate resolved from outer-training labels only",
            ))
    return fold_local_decisions, resolved


def routing_sensitivity(
    root: Path,
    run_id: UUID,
    dataset: MLExperimentDataset,
    records: tuple[dict[str, object], ...],
    baselines: dict[str, object],
) -> dict[str, object]:
    frozen = load_frozen_foundation_v3(root, run_id)
    observation_index = {(row.task_id, row.candidate_id): row for row in frozen.observations}
    row_index = {(row.task_id, row.candidate_id): row for row in dataset.rows}
    reference_cost = Decimal(str(
        baselines["policies"][FOLD_LOCAL_ALWAYS_STRONGEST]["total_realized_cost_usd"]))
    artifact = {
        "threshold_grid": list(THRESHOLDS),
        "final_threshold_selected": False,
        "variants": {},
    }
    for variant in VARIANTS:
        predictions = [record for record in records if record["variant"] == variant]
        by_task: dict[str, list[dict[str, object]]] = defaultdict(list)
        for record in predictions:
            by_task[str(record["task_id"])].append(record)
        threshold_results = {}
        for threshold in THRESHOLDS:
            decisions = []
            fallback_by_task = {}
            selection_records = []
            for task_id, candidates in sorted(by_task.items()):
                selected, fallback = select_cost_aware_candidate(
                    task_id, candidates, row_index, threshold)
                candidate_id = str(selected["candidate_id"])
                fallback_by_task[task_id] = fallback
                decisions.append(RoutingDecision(
                    policy=f"{variant}@{threshold:.2f}",
                    task_id=task_id,
                    candidate_id=candidate_id,
                    reason=("lowest projected cost above threshold" if not fallback
                            else "highest predicted probability fallback"),
                ))
                observed = observation_index[(task_id, candidate_id)]
                selection_records.append({
                    "task_id": task_id,
                    "candidate_id": candidate_id,
                    "predicted_probability": selected["predicted_probability"],
                    "projected_cost_usd": str(row_index[(task_id, candidate_id)].projected_cost_usd),
                    "fallback_used": fallback,
                    "selected_label_status": observed.label_status,
                })
            summary, _ = evaluate_decisions(frozen.observations, decisions)
            fallback_count = sum(fallback_by_task.values())
            selected_cost = Decimal(str(summary["total_realized_cost_usd"]))
            summary.update({
                "threshold": threshold,
                "fallback_count": fallback_count,
                "fallback_rate": fallback_count / 56,
                "cost_reduction_vs_fold_local_strongest": float(
                    (reference_cost - selected_cost) / reference_cost),
                "selections": selection_records,
            })
            threshold_results[f"{threshold:.2f}"] = summary
        artifact["variants"][variant] = threshold_results
    return artifact


def select_cost_aware_candidate(
    task_id: str,
    candidates: list[dict[str, object]],
    row_index: dict[tuple[str, str], MLExperimentRow],
    threshold: float,
) -> tuple[dict[str, object], bool]:
    """Select using OOF probability and projected cost, never realized outcomes."""
    qualified = [item for item in candidates
                 if float(item["predicted_probability"]) >= threshold]
    if qualified:
        return min(qualified, key=lambda item: (
            row_index[(task_id, str(item["candidate_id"]))].projected_cost_usd,
            str(item["candidate_id"]),
        )), False
    return min(candidates, key=lambda item: (
        -float(item["predicted_probability"]),
        row_index[(task_id, str(item["candidate_id"]))].projected_cost_usd,
        str(item["candidate_id"]),
    )), True


def _report(predictive, sensitivity, baselines) -> str:
    lines = [
        "# Phase 7B: First Supervised ML Router Experiment",
        "",
        "Grouped out-of-fold analysis over 56 Foundation V3 requests. No final threshold or model variant is selected.",
        "",
        "## Predictive metrics",
        "",
        "| Variant | Log loss | Brier | ROC-AUC | Average precision | Accuracy | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        metric = predictive[variant]["overall"]
        lines.append(
            f"| {variant} | {metric['log_loss']:.4f} | {metric['brier_score']:.4f} | "
            f"{metric['roc_auc']:.4f} | {metric['average_precision']:.4f} | "
            f"{metric['accuracy_at_0_5']:.4f} | {metric['precision_at_0_5']:.4f} | "
            f"{metric['recall_at_0_5']:.4f} | {metric['f1_at_0_5']:.4f} |"
        )
    for variant in VARIANTS:
        lines.extend([
            "",
            f"## {variant} routing sensitivity",
            "",
            "| Threshold | Coverage | Acceptable rate | Mean quality | Total cost | Cost reduction vs fold-local strongest | Fallback rate | Missing selected |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for threshold in THRESHOLDS:
            summary = sensitivity["variants"][variant][f"{threshold:.2f}"]
            lines.append(
                f"| {threshold:.2f} | {summary['valid_label_coverage']:.3f} | "
                f"{summary['acceptable_rate_among_valid']:.3f} | "
                f"{summary['mean_quality_among_valid']:.3f} | "
                f"${Decimal(str(summary['total_realized_cost_usd'])):.8f} | "
                f"{summary['cost_reduction_vs_fold_local_strongest']:.2%} | "
                f"{summary['fallback_rate']:.2%} | {summary['missing_selected_labels']} |"
            )
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "These are sensitivity results on a small frozen benchmark. They do not select a production threshold, serialize a final model, or establish generalization beyond Foundation V3.",
        "",
        f"Fold-local strongest reference cost: ${Decimal(str(baselines['policies'][FOLD_LOCAL_ALWAYS_STRONGEST]['total_realized_cost_usd'])):.8f}.",
        "",
    ])
    return "\n".join(lines)


def run_experiment(root: Path, run_id: UUID = FOUNDATION_V3_RUN_ID) -> tuple[dict[str, Path], dict[str, object]]:
    design_path = root / str(run_id) / "phase-7" / "ml-router-design.json"
    if hashlib.sha256(design_path.read_bytes()).hexdigest() != DESIGN_SHA256:
        raise ValueError("Phase 7A design artifact hash mismatch")
    dataset = load_ml_dataset(root, run_id)
    records, coefficients, fit_audit, folds = generate_oof_predictions(dataset)
    predictive = predictive_evaluation(records)
    baselines = _baseline_evaluation(root, run_id, folds)
    sensitivity = routing_sensitivity(root, run_id, dataset, records, baselines)
    output = root / str(run_id) / "phase-7"
    paths = {
        "folds": output / "fold-assignments.json",
        "oof": output / "oof-predictions.jsonl",
        "predictive": output / "predictive-metrics.json",
        "routing": output / "routing-sensitivity.json",
        "coefficients": output / "coefficient-summary.json",
        "report": output / "phase-7b-report.md",
    }
    fold_artifact = {
        "random_state": RANDOM_STATE,
        "grouping_key": "task_id",
        "predictive_use_of_task_id": False,
        "folds": [assignment.model_dump(mode="json") for assignment in folds],
    }
    predictive_artifact = {
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "architecture": platform.machine(),
        },
        "source": {
            "run_id": str(run_id),
            "foundation_v2_sha256": FOUNDATION_V2_SHA256,
            "foundation_v3_sha256": FOUNDATION_V3_SHA256,
            "foundation_v3_protocol_sha256": FOUNDATION_V3_PROTOCOL_SHA256,
            "phase_7a_design_sha256": DESIGN_SHA256,
            "rows": len(dataset.rows),
            "valid_targets": sum(row.label_status == "valid" for row in dataset.rows),
            "missing_targets": sum(row.label_status == "missing" for row in dataset.rows),
        },
        "feature_governance": {
            "categorical": CATEGORICAL_FEATURES,
            "numeric": NUMERIC_FEATURES,
            "boolean": BOOLEAN_FEATURES,
            "forbidden": sorted(FORBIDDEN_PREDICTIVE_FIELDS),
        },
        "variants": {
            LOGREG_UNWEIGHTED: {"class_weight": None, "l1_ratio": 0.0, "C": 1.0},
            LOGREG_BALANCED: {"class_weight": "balanced", "l1_ratio": 0.0, "C": 1.0},
        },
        "fit_audit": fit_audit,
        "metrics": predictive,
    }
    _atomic_write(paths["folds"], _json(fold_artifact))
    _atomic_write(paths["oof"], "".join(
        json.dumps(_normalized(record), sort_keys=True, separators=(",", ":")) + "\n"
        for record in sorted(records, key=lambda item: (
            str(item["variant"]), str(item["task_id"]), str(item["candidate_id"])))
    ))
    _atomic_write(paths["predictive"], _json(predictive_artifact))
    _atomic_write(paths["routing"], _json({
        "matched_baselines": baselines,
        "learned_router_sensitivity": sensitivity,
    }))
    _atomic_write(paths["coefficients"], _json({
        "interpretation": "Fold-local standardized coefficients are descriptive, not causal.",
        "fold_coefficients": coefficients,
    }))
    _atomic_write(paths["report"], _report(predictive, sensitivity, baselines))
    return paths, {
        "predictive": predictive,
        "sensitivity": sensitivity,
        "baselines": baselines,
        "folds": fold_artifact,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local grouped Phase 7B experiment")
    parser.add_argument("--root", type=Path, default=Path("benchmark-results"))
    parser.add_argument("--run-id", type=UUID, default=FOUNDATION_V3_RUN_ID)
    args = parser.parse_args()
    started = perf_counter()
    paths, _ = run_experiment(args.root, args.run_id)
    for name, path in paths.items():
        print(f"{name}: {path}")
    print(f"wall_clock_seconds: {perf_counter() - started:.6f}")


if __name__ == "__main__":
    main()
