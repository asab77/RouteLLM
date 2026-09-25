"""Canonical pre-generation inputs and shared preprocessing for quality prediction."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Iterable

import numpy as np
from pydantic import Field
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from adaptive_llm_gateway.errors import MissingRoutingCategoryError
from adaptive_llm_gateway.models import ModelConfig, ReasoningEffort
from adaptive_llm_gateway.models.schemas import DomainModel, Identifier

from .features import ROUTING_CATEGORY_TAXONOMY_VERSION, RoutingCategory, RoutingRequestFeatures

CANONICAL_QUALITY_FEATURE_SCHEMA_VERSION = "1.0.0"
QUALITY_PREPROCESSING_ID = "interaction-no-provider-pin-column-transformer-v1"
PREDICTOR_FORMULATION_ID = "INTERACTION_NO_PROVIDER_PIN"
PREDICTOR_FORMULATION_VERSION = "1.0.0"
QUALITY_RANDOM_STATE = 20260924
INTERACTION_FEATURE = "category_candidate"

CANONICAL_CATEGORICAL_FEATURES = (
    "category",
    "candidate_id",
    "reasoning_effort",
    INTERACTION_FEATURE,
)
CANONICAL_NUMERIC_FEATURES = (
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
CANONICAL_BOOLEAN_FEATURES = (
    "contains_code",
    "requests_structured_output",
    "supports_temperature",
)
CANONICAL_PREDICTIVE_FEATURES = (
    CANONICAL_CATEGORICAL_FEATURES
    + CANONICAL_NUMERIC_FEATURES
    + CANONICAL_BOOLEAN_FEATURES
)

FORBIDDEN_CANONICAL_FEATURES = frozenset({
    "upstream_provider_pin", "generated_response", "input_tokens", "output_tokens",
    "reasoning_tokens", "latency_ms", "realized_cost_usd", "candidate_cost_usd",
    "evaluator_outcome", "quality_score", "acceptable", "analysis_difficulty",
    "ground_truth", "provider_outcome", "provider_error_category", "response_text",
    "category_provenance", "provider_payload",
})


class CanonicalQualityFeatures(DomainModel):
    """Versioned predictor input shared by training and production serving."""

    schema_version: str = CANONICAL_QUALITY_FEATURE_SCHEMA_VERSION
    category: RoutingCategory
    candidate_id: Identifier
    reasoning_effort: ReasoningEffort
    prompt_characters: float = Field(ge=0, allow_inf_nan=False)
    approximate_input_tokens: float = Field(ge=0, allow_inf_nan=False)
    requested_max_output_tokens: float = Field(gt=0, allow_inf_nan=False)
    constraint_indicator_count: float = Field(ge=0, allow_inf_nan=False)
    reasoning_indicator_count: float = Field(ge=0, allow_inf_nan=False)
    configured_input_cost_per_1m_tokens: float = Field(ge=0, allow_inf_nan=False)
    configured_output_cost_per_1m_tokens: float = Field(ge=0, allow_inf_nan=False)
    context_window: float = Field(gt=0, allow_inf_nan=False)
    effective_max_output_tokens: float = Field(gt=0, allow_inf_nan=False)
    contains_code: bool = Field(strict=True)
    requests_structured_output: bool = Field(strict=True)
    supports_temperature: bool = Field(strict=True)

    @property
    def category_candidate(self) -> str:
        return f"{self.category.value}::{self.candidate_id}"


def canonicalize_category(value: RoutingCategory | str) -> RoutingCategory:
    """Map frozen experimental categories into the production taxonomy."""
    normalized = value.value if isinstance(value, RoutingCategory) else value.strip()
    if normalized == "json":
        normalized = RoutingCategory.STRUCTURED_JSON.value
    return RoutingCategory(normalized)


def canonicalize_reasoning_effort(value: ReasoningEffort | str | None) -> ReasoningEffort:
    """Represent omitted reasoning exactly as the Phase 7 string value `none`."""
    if value is None:
        return ReasoningEffort.NONE
    return value if isinstance(value, ReasoningEffort) else ReasoningEffort(value)


def resolve_effective_output_allowance(
    request_features: RoutingRequestFeatures,
    candidate: ModelConfig,
) -> int:
    """Resolve configured pre-generation allowance without candidate identity checks."""
    if request_features.category is None:
        raise MissingRoutingCategoryError(
            "quality prediction requires an explicit routing category"
        )
    category = canonicalize_category(request_features.category).value
    matching = tuple(
        item.max_output_tokens
        for item in candidate.output_token_policy.category_overrides
        if canonicalize_category(item.category).value == category
    )
    if len(matching) > 1:
        raise ValueError("candidate output-token policy has duplicate canonical categories")
    return matching[0] if matching else request_features.max_output_tokens


def canonical_from_training_row(row: Any) -> CanonicalQualityFeatures:
    """Adapt one governed Foundation experiment row without outcome leakage."""
    values = row.features
    return CanonicalQualityFeatures(
        category=canonicalize_category(values["category"]),
        candidate_id=row.candidate_id,
        reasoning_effort=canonicalize_reasoning_effort(values["reasoning_effort"]),
        prompt_characters=float(values["prompt_characters"]),
        approximate_input_tokens=float(values["approximate_input_tokens"]),
        requested_max_output_tokens=float(values["requested_max_output_tokens"]),
        constraint_indicator_count=float(values["constraint_indicator_count"]),
        reasoning_indicator_count=float(values["reasoning_indicator_count"]),
        configured_input_cost_per_1m_tokens=float(
            values["configured_input_cost_per_1m_tokens"]
        ),
        configured_output_cost_per_1m_tokens=float(
            values["configured_output_cost_per_1m_tokens"]
        ),
        context_window=float(values["context_window"]),
        effective_max_output_tokens=float(values["effective_max_output_tokens"]),
        contains_code=bool(values["contains_code"]),
        requests_structured_output=bool(values["requests_structured_output"]),
        supports_temperature=bool(values["supports_temperature"]),
    )


def canonical_from_production(
    request_features: RoutingRequestFeatures,
    candidate: ModelConfig,
) -> CanonicalQualityFeatures:
    """Adapt production request and candidate configuration to the same contract."""
    if request_features.category is None:
        raise MissingRoutingCategoryError(
            "quality prediction requires an explicit routing category"
        )
    return CanonicalQualityFeatures(
        category=canonicalize_category(request_features.category),
        candidate_id=candidate.model_id,
        reasoning_effort=canonicalize_reasoning_effort(candidate.reasoning_effort),
        prompt_characters=float(request_features.prompt_characters),
        approximate_input_tokens=float(request_features.approximate_input_tokens),
        requested_max_output_tokens=float(request_features.max_output_tokens),
        constraint_indicator_count=float(request_features.constraint_indicator_count),
        reasoning_indicator_count=float(request_features.reasoning_indicator_count),
        configured_input_cost_per_1m_tokens=float(candidate.input_cost_per_1m_tokens),
        configured_output_cost_per_1m_tokens=float(candidate.output_cost_per_1m_tokens),
        context_window=float(candidate.context_window),
        effective_max_output_tokens=float(
            resolve_effective_output_allowance(request_features, candidate)
        ),
        contains_code=request_features.contains_code,
        requests_structured_output=request_features.requests_structured_output,
        supports_temperature=candidate.capabilities.supports_temperature,
    )


def canonical_feature_matrix(rows: Iterable[CanonicalQualityFeatures]) -> np.ndarray:
    """Create the sole ordered matrix representation consumed by sklearn."""
    matrix = []
    for row in rows:
        values = row.model_dump(exclude={"schema_version"}, mode="python")
        values["category"] = row.category.value
        values["reasoning_effort"] = row.reasoning_effort.value
        values[INTERACTION_FEATURE] = row.category_candidate
        matrix.append([values[name] for name in CANONICAL_PREDICTIVE_FEATURES])
    return np.asarray(matrix, dtype=object)


def _as_float(values: np.ndarray) -> np.ndarray:
    return values.astype(float)


def build_quality_pipeline() -> Pipeline:
    """Build the exact approved no-provider-pin preprocessing and classifier."""
    categorical_end = len(CANONICAL_CATEGORICAL_FEATURES)
    numeric_end = categorical_end + len(CANONICAL_NUMERIC_FEATURES)
    preprocess = ColumnTransformer((
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            list(range(categorical_end)),
        ),
        (
            "numeric",
            StandardScaler(),
            list(range(categorical_end, numeric_end)),
        ),
        (
            "boolean",
            FunctionTransformer(_as_float, feature_names_out="one-to-one"),
            list(range(numeric_end, len(CANONICAL_PREDICTIVE_FEATURES))),
        ),
    ), remainder="drop")
    return Pipeline((
        ("preprocess", preprocess),
        ("classifier", LogisticRegression(
            l1_ratio=0.0,
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            class_weight=None,
            random_state=QUALITY_RANDOM_STATE,
        )),
    ))


class CanonicalCompatibilityStatus(StrEnum):
    EXACT_MATCH = "EXACT_MATCH"
    CANONICALIZED_MATCH = "CANONICALIZED_MATCH"
    UNRESOLVED = "UNRESOLVED"


class CanonicalSkewAuditEntry(DomainModel):
    feature_name: str
    status: CanonicalCompatibilityStatus
    resolution: str


CANONICAL_TRAINING_SERVING_AUDIT = (
    CanonicalSkewAuditEntry(feature_name="category", status=CanonicalCompatibilityStatus.CANONICALIZED_MATCH,
        resolution="Training json maps to structured_json; production categories are already canonical."),
    CanonicalSkewAuditEntry(feature_name="candidate_id", status=CanonicalCompatibilityStatus.EXACT_MATCH,
        resolution="Training internal ID and production ModelConfig.model_id are identical."),
    CanonicalSkewAuditEntry(feature_name="reasoning_effort", status=CanonicalCompatibilityStatus.CANONICALIZED_MATCH,
        resolution="Training strings and production enum/None normalize to ReasoningEffort."),
    CanonicalSkewAuditEntry(feature_name=INTERACTION_FEATURE, status=CanonicalCompatibilityStatus.CANONICALIZED_MATCH,
        resolution="One shared vectorizer derives canonical category::candidate identity."),
    CanonicalSkewAuditEntry(feature_name="prompt_characters", status=CanonicalCompatibilityStatus.EXACT_MATCH,
        resolution="Both adapters use the governed user-prompt character count."),
    CanonicalSkewAuditEntry(feature_name="approximate_input_tokens", status=CanonicalCompatibilityStatus.EXACT_MATCH,
        resolution="Both adapters use the shared whitespace estimate."),
    CanonicalSkewAuditEntry(feature_name="requested_max_output_tokens", status=CanonicalCompatibilityStatus.EXACT_MATCH,
        resolution="Both adapters use the explicit request allowance."),
    CanonicalSkewAuditEntry(feature_name="constraint_indicator_count", status=CanonicalCompatibilityStatus.EXACT_MATCH,
        resolution="Both adapters use the shared frozen vocabulary count."),
    CanonicalSkewAuditEntry(feature_name="reasoning_indicator_count", status=CanonicalCompatibilityStatus.EXACT_MATCH,
        resolution="Both adapters use the shared frozen vocabulary count."),
    CanonicalSkewAuditEntry(feature_name="configured_input_cost_per_1m_tokens", status=CanonicalCompatibilityStatus.EXACT_MATCH,
        resolution="Both adapters use configured candidate input price."),
    CanonicalSkewAuditEntry(feature_name="configured_output_cost_per_1m_tokens", status=CanonicalCompatibilityStatus.EXACT_MATCH,
        resolution="Both adapters use configured candidate output price."),
    CanonicalSkewAuditEntry(feature_name="context_window", status=CanonicalCompatibilityStatus.EXACT_MATCH,
        resolution="Both adapters use configured candidate context window."),
    CanonicalSkewAuditEntry(feature_name="effective_max_output_tokens", status=CanonicalCompatibilityStatus.CANONICALIZED_MATCH,
        resolution="Typed category allowance policy reproduces frozen configured overrides."),
    CanonicalSkewAuditEntry(feature_name="contains_code", status=CanonicalCompatibilityStatus.EXACT_MATCH,
        resolution="Both adapters use the governed request indicator."),
    CanonicalSkewAuditEntry(feature_name="requests_structured_output", status=CanonicalCompatibilityStatus.CANONICALIZED_MATCH,
        resolution="Benchmark configuration and explicit production requirement map to one boolean."),
    CanonicalSkewAuditEntry(feature_name="supports_temperature", status=CanonicalCompatibilityStatus.EXACT_MATCH,
        resolution="Both adapters use typed candidate capability configuration."),
)


def validate_canonical_skew_audit() -> dict[str, int]:
    names = [item.feature_name for item in CANONICAL_TRAINING_SERVING_AUDIT]
    if len(names) != len(set(names)) or set(names) != set(CANONICAL_PREDICTIVE_FEATURES):
        raise ValueError("canonical skew audit must cover every predictive feature once")
    counts = {
        status.value: sum(item.status is status for item in CANONICAL_TRAINING_SERVING_AUDIT)
        for status in CanonicalCompatibilityStatus
    }
    if counts[CanonicalCompatibilityStatus.UNRESOLVED.value]:
        raise ValueError("canonical training/serving skew audit remains unresolved")
    return counts


__all__ = [
    "CANONICAL_BOOLEAN_FEATURES", "CANONICAL_CATEGORICAL_FEATURES",
    "CANONICAL_NUMERIC_FEATURES", "CANONICAL_PREDICTIVE_FEATURES",
    "CANONICAL_QUALITY_FEATURE_SCHEMA_VERSION", "CANONICAL_TRAINING_SERVING_AUDIT",
    "CanonicalCompatibilityStatus", "CanonicalQualityFeatures", "CanonicalSkewAuditEntry",
    "FORBIDDEN_CANONICAL_FEATURES", "INTERACTION_FEATURE", "PREDICTOR_FORMULATION_ID",
    "PREDICTOR_FORMULATION_VERSION", "QUALITY_PREPROCESSING_ID",
    "QUALITY_RANDOM_STATE",
    "ROUTING_CATEGORY_TAXONOMY_VERSION", "build_quality_pipeline",
    "canonical_feature_matrix", "canonical_from_production", "canonical_from_training_row",
    "canonicalize_category", "canonicalize_reasoning_effort",
    "resolve_effective_output_allowance", "validate_canonical_skew_audit",
]
