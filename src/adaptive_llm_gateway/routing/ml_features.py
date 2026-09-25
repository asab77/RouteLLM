"""Governed feature rows and deterministic grouped folds for ML experiments."""
from __future__ import annotations

from collections import Counter, defaultdict
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Literal
from uuid import UUID

import numpy as np
from pydantic import Field, model_validator
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from adaptive_llm_gateway.benchmarks.routing_export import RoutingDatasetRow
from adaptive_llm_gateway.models.schemas import DomainModel
from .analysis import FOUNDATION_V3_RUN_ID, load_frozen_foundation_v3

CATEGORICAL_FEATURES = (
    "category",
    "candidate_id",
    "upstream_provider_pin",
    "reasoning_effort",
)
NUMERIC_FEATURES = (
    "prompt_characters",
    "approximate_input_tokens",
    "requested_max_output_tokens",
    "constraint_indicator_count",
    "reasoning_indicator_count",
    "configured_input_cost_per_1m_tokens",
    "configured_output_cost_per_1m_tokens",
    "context_window",
    "effective_max_output_tokens",
)
BOOLEAN_FEATURES = (
    "contains_code",
    "requests_structured_output",
    "supports_temperature",
)
PREDICTIVE_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES + BOOLEAN_FEATURES

FORBIDDEN_PREDICTIVE_FIELDS = frozenset({
    "acceptable", "quality_score", "analysis_difficulty", "provider_outcome",
    "provider_error_category", "latency_ms", "candidate_cost_usd", "input_tokens",
    "output_tokens", "reasoning_tokens", "evaluator_name", "evaluator_version",
    "evaluation_status", "evaluation_call_made", "evaluation_input_tokens",
    "evaluation_output_tokens", "evaluation_latency_ms", "evaluation_cost_usd",
    "task_id", "request_id", "benchmark_result_id", "raw_result_path",
    "evaluation_path", "missing_label_reason", "generated_at", "ground_truth",
    "response_text", "run_id", "dataset_sha256",
})

OUTER_FOLDS = 4
RANDOM_STATE = 20260924


class MLExperimentRow(DomainModel):
    """Governed predictors plus separately held retrospective fields."""

    task_id: str
    candidate_id: str
    category: str
    features: dict[str, str | float | bool]
    label_status: Literal["valid", "missing"]
    acceptable: bool | None
    quality_score: float | None = Field(default=None, ge=0, le=1)
    analysis_difficulty: Literal["easy", "medium", "hard"] | None
    realized_cost_usd: Decimal | None = Field(default=None, ge=0)
    latency_ms: float = Field(ge=0)

    @model_validator(mode="after")
    def governed_feature_boundary(self):
        if set(self.features) != set(PREDICTIVE_FEATURES):
            raise ValueError("Predictive row must contain exactly the governed feature set")
        if set(self.features) & FORBIDDEN_PREDICTIVE_FIELDS:
            raise ValueError("Predictive row contains a forbidden leakage field")
        if self.features["candidate_id"] != self.candidate_id:
            raise ValueError("Candidate predictor must match the row candidate")
        if self.features["category"] != self.category:
            raise ValueError("Category predictor must match the request category")
        if self.label_status == "valid" and self.acceptable is None:
            raise ValueError("Valid ML rows require a target")
        if self.label_status == "missing" and self.acceptable is not None:
            raise ValueError("Missing targets must remain missing")
        return self

    @property
    def projected_cost_usd(self) -> Decimal:
        with localcontext() as context:
            context.prec = 50
            return (
                Decimal(str(self.features["approximate_input_tokens"]))
                * Decimal(str(self.features["configured_input_cost_per_1m_tokens"]))
                + Decimal(str(self.features["effective_max_output_tokens"]))
                * Decimal(str(self.features["configured_output_cost_per_1m_tokens"]))
            ) / Decimal(1_000_000)
class MLExperimentDataset(DomainModel):
    run_id: UUID
    rows: tuple[MLExperimentRow, ...]


class FoldAssignment(DomainModel):
    fold: int = Field(ge=0, lt=OUTER_FOLDS)
    task_ids: tuple[str, ...]
    category_counts: dict[str, int]
    positive_labels: int = Field(ge=0)
    negative_labels: int = Field(ge=0)
    missing_labels: int = Field(ge=0)


def _feature_values(row: RoutingDatasetRow) -> dict[str, str | float | bool]:
    capabilities = row.candidate.capabilities
    return {
        "category": row.request_features.category or "unknown",
        "candidate_id": row.candidate.internal_id,
        "upstream_provider_pin": row.candidate.upstream_provider_pin,
        "reasoning_effort": row.candidate.reasoning_effort or "none",
        "prompt_characters": float(row.request_features.prompt_characters),
        "approximate_input_tokens": float(row.request_features.approximate_input_tokens),
        "requested_max_output_tokens": float(row.request_features.max_output_tokens),
        "constraint_indicator_count": float(row.request_features.constraint_indicator_count),
        "reasoning_indicator_count": float(row.request_features.reasoning_indicator_count),
        "configured_input_cost_per_1m_tokens": float(row.candidate.input_cost_per_1m_tokens),
        "configured_output_cost_per_1m_tokens": float(row.candidate.output_cost_per_1m_tokens),
        "context_window": float(row.candidate.context_window),
        "effective_max_output_tokens": float(row.effective_max_output_tokens),
        "contains_code": row.request_features.contains_code,
        "requests_structured_output": row.request_features.requests_structured_output,
        "supports_temperature": bool(capabilities.get("supports_temperature", False)),
    }


def load_ml_dataset(root: Path, run_id: UUID = FOUNDATION_V3_RUN_ID) -> MLExperimentDataset:
    frozen = load_frozen_foundation_v3(root, run_id)
    path = root / str(run_id) / "routing-dataset.jsonl"
    exported = tuple(
        RoutingDatasetRow.model_validate_json(line)
        for line in path.read_bytes().splitlines() if line
    )
    rows = tuple(MLExperimentRow(
        task_id=row.task_id,
        candidate_id=row.candidate.internal_id,
        category=row.request_features.category or "unknown",
        features=_feature_values(row),
        label_status=row.label_status,
        acceptable=row.acceptable,
        quality_score=row.quality_score,
        analysis_difficulty=row.analysis_difficulty,
        realized_cost_usd=row.candidate_cost_usd,
        latency_ms=row.latency_ms,
    ) for row in sorted(exported, key=lambda item: (item.task_id, item.candidate.internal_id)))
    if len(rows) != 224 or len({(row.task_id, row.candidate_id) for row in rows}) != 224:
        raise ValueError("ML dataset must preserve all 224 unique candidate rows")
    if {row.task_id for row in rows} != {request.task_id for request in frozen.requests}:
        raise ValueError("ML dataset request groups differ from the frozen routing dataset")
    if Counter(row.label_status for row in rows) != Counter({"valid": 216, "missing": 8}):
        raise ValueError("ML dataset target availability differs from the frozen dataset")
    return MLExperimentDataset(run_id=run_id, rows=rows)


def build_outer_folds(dataset: MLExperimentDataset) -> tuple[FoldAssignment, ...]:
    by_task: dict[str, list[MLExperimentRow]] = defaultdict(list)
    for row in dataset.rows:
        by_task[row.task_id].append(row)
    profiles = {}
    for task_id, rows in by_task.items():
        if len(rows) != 4 or len({row.category for row in rows}) != 1:
            raise ValueError("Every request group must contain four rows from one category")
        profiles[task_id] = {
            "category": rows[0].category,
            "positive": sum(row.acceptable is True for row in rows),
            "negative": sum(row.acceptable is False for row in rows),
            "missing": sum(row.label_status == "missing" for row in rows),
        }

    fold_tasks: list[list[str]] = [[] for _ in range(OUTER_FOLDS)]
    fold_positive = [0] * OUTER_FOLDS
    fold_missing = [0] * OUTER_FOLDS
    categories = sorted({profile["category"] for profile in profiles.values()})
    for category in categories:
        tasks = sorted(
            (task_id for task_id, profile in profiles.items() if profile["category"] == category),
            key=lambda task_id: (
                profiles[task_id]["positive"], profiles[task_id]["missing"], task_id),
        )
        if len(tasks) != 8:
            raise ValueError("Category-balanced four-fold evaluation requires eight tasks per category")
        pairs = [(tasks[index], tasks[-1 - index]) for index in range(4)]
        pairs.sort(key=lambda pair: (
            -sum(profiles[item]["missing"] for item in pair),
            -sum(profiles[item]["positive"] for item in pair),
            pair,
        ))
        available = set(range(OUTER_FOLDS))
        for pair in pairs:
            fold = min(available, key=lambda item: (
                fold_missing[item], fold_positive[item], item))
            available.remove(fold)
            fold_tasks[fold].extend(pair)
            fold_positive[fold] += sum(profiles[item]["positive"] for item in pair)
            fold_missing[fold] += sum(profiles[item]["missing"] for item in pair)

    assignments = []
    all_test_tasks = []
    for fold, task_ids in enumerate(fold_tasks):
        task_ids = sorted(task_ids)
        all_test_tasks.extend(task_ids)
        category_counts = Counter(profiles[task_id]["category"] for task_id in task_ids)
        assignment = FoldAssignment(
            fold=fold,
            task_ids=tuple(task_ids),
            category_counts=dict(sorted(category_counts.items())),
            positive_labels=sum(profiles[item]["positive"] for item in task_ids),
            negative_labels=sum(profiles[item]["negative"] for item in task_ids),
            missing_labels=sum(profiles[item]["missing"] for item in task_ids),
        )
        if len(task_ids) != 14 or set(category_counts.values()) != {2}:
            raise ValueError("Each outer fold must contain 14 tasks and two from every category")
        assignments.append(assignment)
    if len(all_test_tasks) != 56 or len(set(all_test_tasks)) != 56:
        raise ValueError("Every request group must appear in exactly one outer fold")
    return tuple(assignments)


def feature_matrix(rows: tuple[MLExperimentRow, ...] | list[MLExperimentRow]) -> np.ndarray:
    return np.asarray([
        [row.features[name] for name in PREDICTIVE_FEATURES]
        for row in rows
    ], dtype=object)


def _as_float(values):
    return values.astype(float)


def build_pipeline(class_weight: str | None) -> Pipeline:
    categorical_indices = list(range(0, len(CATEGORICAL_FEATURES)))
    numeric_start = len(CATEGORICAL_FEATURES)
    numeric_indices = list(range(numeric_start, numeric_start + len(NUMERIC_FEATURES)))
    boolean_start = numeric_start + len(NUMERIC_FEATURES)
    boolean_indices = list(range(boolean_start, len(PREDICTIVE_FEATURES)))
    preprocessor = ColumnTransformer(
        transformers=(
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             categorical_indices),
            ("numeric", StandardScaler(), numeric_indices),
            ("boolean", FunctionTransformer(
                _as_float, feature_names_out="one-to-one"), boolean_indices),
        ),
        remainder="drop",
    )
    return Pipeline((
        ("preprocess", preprocessor),
        ("classifier", LogisticRegression(
            l1_ratio=0.0, C=1.0, solver="lbfgs", max_iter=1000,
            class_weight=class_weight, random_state=RANDOM_STATE,
        )),
    ))
