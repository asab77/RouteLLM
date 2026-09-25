"""Governed production request features available before model execution."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any

from pydantic import Field, model_validator

from adaptive_llm_gateway.models import InferenceRequest
from adaptive_llm_gateway.models.schemas import DomainModel

ROUTING_CATEGORY_TAXONOMY_VERSION = "phase7-v1"


class RoutingCategory(StrEnum):
    """Current versionable router taxonomy inherited from Phase 7."""

    CLASSIFICATION = "classification"
    CODING = "coding"
    EXTRACTION = "extraction"
    STRUCTURED_JSON = "structured_json"
    QA = "qa"
    REASONING = "reasoning"
    SUMMARIZATION = "summarization"


class CategoryProvenance(StrEnum):
    CLIENT_HINT = "client_hint"
    INFERRED = "inferred"
    ABSENT = "absent"


class FeatureSource(StrEnum):
    REQUEST_CONTENT = "request_content"
    REQUEST_CONFIGURATION = "request_configuration"
    CLIENT_HINT = "client_hint"
    DERIVED_METADATA = "derived_metadata"


class LeakageStatus(StrEnum):
    PRE_GENERATION_ALLOWED = "pre_generation_allowed"
    POST_GENERATION_FORBIDDEN = "post_generation_forbidden"


class FeatureCompatibilityStatus(StrEnum):
    EXACT_MATCH = "EXACT_MATCH"
    COMPATIBLE_WITH_DOCUMENTED_DIFFERENCE = "COMPATIBLE_WITH_DOCUMENTED_DIFFERENCE"
    NOT_AVAILABLE_IN_PRODUCTION = "NOT_AVAILABLE_IN_PRODUCTION"
    EXPERIMENT_ONLY = "EXPERIMENT_ONLY"


class RoutingRequestFeatures(DomainModel):
    """Deterministic request-visible values supplied to a future predictor."""

    category: RoutingCategory | None = None
    category_provenance: CategoryProvenance = CategoryProvenance.ABSENT
    prompt_characters: int = Field(ge=0, strict=True)
    system_prompt_characters: int = Field(ge=0, strict=True)
    message_count: int = Field(ge=1, le=2, strict=True)
    system_prompt_present: bool = Field(strict=True)
    approximate_input_tokens: int = Field(ge=0, strict=True)
    contains_code: bool = Field(strict=True)
    requests_structured_output: bool = Field(strict=True)
    max_output_tokens: int = Field(gt=0, strict=True)
    constraint_indicator_count: int = Field(ge=0, strict=True)
    reasoning_indicator_count: int = Field(ge=0, strict=True)

    @model_validator(mode="before")
    @classmethod
    def supply_compatible_metadata_defaults(cls, value: Any) -> Any:
        """Preserve Phase 8A construction while making new metadata explicit."""
        if not isinstance(value, dict):
            return value
        data = dict(value)
        category = data.get("category")
        if isinstance(category, str):
            category = category.strip()
            data["category"] = category
        data.setdefault(
            "category_provenance",
            CategoryProvenance.CLIENT_HINT if category is not None else CategoryProvenance.ABSENT,
        )
        system_characters = data.get("system_prompt_characters", 0)
        data.setdefault("system_prompt_present", bool(system_characters))
        data.setdefault("message_count", 2 if system_characters else 1)
        return data

    @model_validator(mode="after")
    def metadata_is_consistent(self) -> "RoutingRequestFeatures":
        if (self.category is None) != (self.category_provenance is CategoryProvenance.ABSENT):
            raise ValueError("absent category and category provenance must agree")
        if self.category is not None and self.category_provenance is CategoryProvenance.ABSENT:
            raise ValueError("a category requires non-absent provenance")
        if self.system_prompt_present != (self.system_prompt_characters > 0):
            raise ValueError("system prompt presence must match its character count")
        expected_messages = 2 if self.system_prompt_present else 1
        if self.message_count != expected_messages:
            raise ValueError("message count must represent user plus optional system prompt")
        return self


class FeatureGovernanceEntry(DomainModel):
    name: str
    value_type: str
    source: FeatureSource
    extraction_semantics: str
    required: bool
    client_controlled: bool
    directly_derived: bool
    leakage_status: LeakageStatus


class TrainingServingAuditEntry(DomainModel):
    feature_name: str
    status: FeatureCompatibilityStatus
    production_semantics: str
    difference: str | None = None


_CONSTRAINTS = re.compile(
    r"\b(?:must|only|exactly|exclude|include|return|output|do not|at most|no more than)\b",
    re.I,
)
_REASONING = re.compile(
    r"\b(?:infer|determine|deduce|calculate|after|before|unless|if|therefore)\b",
    re.I,
)
_CODE = re.compile(
    r"\x60{3}|\bpython code\b|\bdef\s+\w+\s*\(|\bdefin(?:e|ing)\s+\w+\s*\(|"
    r"\bfunction\s+\w+\s*\(|\bclass\s+\w+",
    re.I,
)


def request_text(request: InferenceRequest) -> str:
    """Join the optional system prompt and required user prompt as Phase 7 did."""
    return " ".join(part for part in (request.system_prompt, request.prompt) if part)


def approximate_input_tokens(request: InferenceRequest) -> int:
    """Return the provider-independent whitespace-token estimate used before routing."""
    return len(request_text(request).split())


def text_contains_code(text: str) -> bool:
    return bool(_CODE.search(text))


def constraint_indicator_count(text: str) -> int:
    return len(_CONSTRAINTS.findall(text))


def reasoning_indicator_count(text: str) -> int:
    return len(_REASONING.findall(text))


class ProductionRequestFeatureExtractor:
    """Pure, deterministic conversion from an inference request to routing features."""

    def extract(
        self,
        request: InferenceRequest,
        *,
        category_hint: RoutingCategory | str | None = None,
        structured_output_required: bool = False,
    ) -> RoutingRequestFeatures:
        if not isinstance(request, InferenceRequest):
            raise TypeError("request must be an InferenceRequest")
        if not isinstance(structured_output_required, bool):
            raise TypeError("structured_output_required must be a boolean")
        normalized_hint = category_hint.strip() if isinstance(category_hint, str) else category_hint
        category = None if normalized_hint is None else RoutingCategory(normalized_hint)
        provenance = (
            CategoryProvenance.ABSENT
            if category is None
            else CategoryProvenance.CLIENT_HINT
        )
        text = request_text(request)
        system_present = request.system_prompt is not None
        return RoutingRequestFeatures(
            category=category,
            category_provenance=provenance,
            prompt_characters=len(request.prompt),
            system_prompt_characters=len(request.system_prompt or ""),
            message_count=2 if system_present else 1,
            system_prompt_present=system_present,
            approximate_input_tokens=approximate_input_tokens(request),
            contains_code=text_contains_code(text),
            requests_structured_output=structured_output_required,
            max_output_tokens=request.max_output_tokens,
            constraint_indicator_count=constraint_indicator_count(text),
            reasoning_indicator_count=reasoning_indicator_count(text),
        )


PRODUCTION_FEATURE_GOVERNANCE = (
    FeatureGovernanceEntry(name="category", value_type="RoutingCategory | None",
        source=FeatureSource.CLIENT_HINT, extraction_semantics="Validated advisory hint; None is explicit absence.",
        required=False, client_controlled=True, directly_derived=False,
        leakage_status=LeakageStatus.PRE_GENERATION_ALLOWED),
    FeatureGovernanceEntry(name="category_provenance", value_type="CategoryProvenance",
        source=FeatureSource.DERIVED_METADATA, extraction_semantics="CLIENT_HINT when supplied; otherwise ABSENT.",
        required=True, client_controlled=False, directly_derived=True,
        leakage_status=LeakageStatus.PRE_GENERATION_ALLOWED),
    FeatureGovernanceEntry(name="prompt_characters", value_type="int",
        source=FeatureSource.REQUEST_CONTENT, extraction_semantics="Python character length of the user prompt.",
        required=True, client_controlled=True, directly_derived=True,
        leakage_status=LeakageStatus.PRE_GENERATION_ALLOWED),
    FeatureGovernanceEntry(name="system_prompt_characters", value_type="int",
        source=FeatureSource.REQUEST_CONTENT, extraction_semantics="Character length of system prompt, or zero.",
        required=True, client_controlled=True, directly_derived=True,
        leakage_status=LeakageStatus.PRE_GENERATION_ALLOWED),
    FeatureGovernanceEntry(name="message_count", value_type="int",
        source=FeatureSource.REQUEST_CONTENT, extraction_semantics="One user message plus one when a system prompt exists.",
        required=True, client_controlled=True, directly_derived=True,
        leakage_status=LeakageStatus.PRE_GENERATION_ALLOWED),
    FeatureGovernanceEntry(name="system_prompt_present", value_type="bool",
        source=FeatureSource.REQUEST_CONTENT, extraction_semantics="True exactly when a system prompt exists.",
        required=True, client_controlled=True, directly_derived=True,
        leakage_status=LeakageStatus.PRE_GENERATION_ALLOWED),
    FeatureGovernanceEntry(name="approximate_input_tokens", value_type="int",
        source=FeatureSource.REQUEST_CONTENT, extraction_semantics="Whitespace-token count over system and user text.",
        required=True, client_controlled=True, directly_derived=True,
        leakage_status=LeakageStatus.PRE_GENERATION_ALLOWED),
    FeatureGovernanceEntry(name="contains_code", value_type="bool",
        source=FeatureSource.REQUEST_CONTENT, extraction_semantics="Frozen Phase 7 regular-expression indicator.",
        required=True, client_controlled=True, directly_derived=True,
        leakage_status=LeakageStatus.PRE_GENERATION_ALLOWED),
    FeatureGovernanceEntry(name="requests_structured_output", value_type="bool",
        source=FeatureSource.REQUEST_CONFIGURATION, extraction_semantics="Explicit internal structured-output requirement only.",
        required=True, client_controlled=True, directly_derived=True,
        leakage_status=LeakageStatus.PRE_GENERATION_ALLOWED),
    FeatureGovernanceEntry(name="max_output_tokens", value_type="int",
        source=FeatureSource.REQUEST_CONFIGURATION, extraction_semantics="Requested base output-token allowance.",
        required=True, client_controlled=True, directly_derived=False,
        leakage_status=LeakageStatus.PRE_GENERATION_ALLOWED),
    FeatureGovernanceEntry(name="constraint_indicator_count", value_type="int",
        source=FeatureSource.REQUEST_CONTENT, extraction_semantics="Count from the frozen Phase 7 constraint vocabulary.",
        required=True, client_controlled=True, directly_derived=True,
        leakage_status=LeakageStatus.PRE_GENERATION_ALLOWED),
    FeatureGovernanceEntry(name="reasoning_indicator_count", value_type="int",
        source=FeatureSource.REQUEST_CONTENT, extraction_semantics="Count from the frozen Phase 7 reasoning vocabulary.",
        required=True, client_controlled=True, directly_derived=True,
        leakage_status=LeakageStatus.PRE_GENERATION_ALLOWED),
)

FORBIDDEN_ROUTING_FEATURES = frozenset({
    "acceptable", "analysis_difficulty", "benchmark_label", "candidate_cost_usd",
    "error_category", "estimated_cost_usd", "evaluation_call_made",
    "evaluation_cost_usd", "evaluation_input_tokens", "evaluation_latency_ms",
    "evaluation_output_tokens", "evaluation_status", "evaluator_name",
    "evaluator_version", "generated_at", "ground_truth", "input_tokens",
    "label_status", "latency_ms", "missing_label_reason", "output_tokens",
    "provider_error_category", "provider_outcome", "quality_score",
    "realized_cost_usd", "reasoning_tokens", "response_text", "success",
})

ACCEPTED_INTERACTION_FEATURES = (
    "category", "candidate_id", "upstream_provider_pin", "reasoning_effort",
    "prompt_characters", "approximate_input_tokens", "requested_max_output_tokens",
    "constraint_indicator_count", "reasoning_indicator_count",
    "configured_input_cost_per_1m_tokens", "configured_output_cost_per_1m_tokens",
    "context_window", "effective_max_output_tokens", "contains_code",
    "requests_structured_output", "supports_temperature", "category_candidate",
)

TRAINING_SERVING_SKEW_AUDIT = (
    TrainingServingAuditEntry(feature_name="category", status=FeatureCompatibilityStatus.COMPATIBLE_WITH_DOCUMENTED_DIFFERENCE,
        production_semantics="Optional controlled enum with explicit absence and provenance.",
        difference="Phase 7 always had a category and used json; production uses structured_json. Phase 8C must map or retrain."),
    TrainingServingAuditEntry(feature_name="candidate_id", status=FeatureCompatibilityStatus.EXACT_MATCH,
        production_semantics="ModelConfig.model_id."),
    TrainingServingAuditEntry(feature_name="upstream_provider_pin", status=FeatureCompatibilityStatus.EXPERIMENT_ONLY,
        production_semantics="No generic registry field exists.", difference="Frozen benchmark protocol recorded an upstream pin behind the gateway."),
    TrainingServingAuditEntry(feature_name="reasoning_effort", status=FeatureCompatibilityStatus.COMPATIBLE_WITH_DOCUMENTED_DIFFERENCE,
        production_semantics="ModelConfig.reasoning_effort enum or None.", difference="Phase 7 normalized None to the string none."),
    TrainingServingAuditEntry(feature_name="prompt_characters", status=FeatureCompatibilityStatus.EXACT_MATCH,
        production_semantics="Length of the user prompt."),
    TrainingServingAuditEntry(feature_name="approximate_input_tokens", status=FeatureCompatibilityStatus.EXACT_MATCH,
        production_semantics="Whitespace count over optional system plus user text."),
    TrainingServingAuditEntry(feature_name="requested_max_output_tokens", status=FeatureCompatibilityStatus.EXACT_MATCH,
        production_semantics="RoutingRequestFeatures.max_output_tokens."),
    TrainingServingAuditEntry(feature_name="constraint_indicator_count", status=FeatureCompatibilityStatus.EXACT_MATCH,
        production_semantics="Shared frozen Phase 7 regular expression."),
    TrainingServingAuditEntry(feature_name="reasoning_indicator_count", status=FeatureCompatibilityStatus.EXACT_MATCH,
        production_semantics="Shared frozen Phase 7 regular expression."),
    TrainingServingAuditEntry(feature_name="configured_input_cost_per_1m_tokens", status=FeatureCompatibilityStatus.EXACT_MATCH,
        production_semantics="ModelConfig input Decimal price."),
    TrainingServingAuditEntry(feature_name="configured_output_cost_per_1m_tokens", status=FeatureCompatibilityStatus.EXACT_MATCH,
        production_semantics="ModelConfig output Decimal price."),
    TrainingServingAuditEntry(feature_name="context_window", status=FeatureCompatibilityStatus.EXACT_MATCH,
        production_semantics="ModelConfig context window."),
    TrainingServingAuditEntry(feature_name="effective_max_output_tokens", status=FeatureCompatibilityStatus.NOT_AVAILABLE_IN_PRODUCTION,
        production_semantics="No production candidate-budget resolver exists yet.", difference="Phase 7 used task-candidate protocol overrides."),
    TrainingServingAuditEntry(feature_name="contains_code", status=FeatureCompatibilityStatus.EXACT_MATCH,
        production_semantics="Shared frozen Phase 7 regular expression."),
    TrainingServingAuditEntry(feature_name="requests_structured_output", status=FeatureCompatibilityStatus.COMPATIBLE_WITH_DOCUMENTED_DIFFERENCE,
        production_semantics="Explicit internal extractor configuration.", difference="Phase 7 read explicit BenchmarkTask expected_output_type."),
    TrainingServingAuditEntry(feature_name="supports_temperature", status=FeatureCompatibilityStatus.EXACT_MATCH,
        production_semantics="ModelConfig capability flag."),
    TrainingServingAuditEntry(feature_name="category_candidate", status=FeatureCompatibilityStatus.COMPATIBLE_WITH_DOCUMENTED_DIFFERENCE,
        production_semantics="Future derived interaction of controlled category and model ID.", difference="Inherits category absence and structured_json mapping requirements."),
)


__all__ = [
    "ACCEPTED_INTERACTION_FEATURES", "CategoryProvenance", "FeatureCompatibilityStatus",
    "FeatureGovernanceEntry", "FeatureSource", "FORBIDDEN_ROUTING_FEATURES",
    "LeakageStatus", "PRODUCTION_FEATURE_GOVERNANCE", "ProductionRequestFeatureExtractor",
    "ROUTING_CATEGORY_TAXONOMY_VERSION", "RoutingCategory", "RoutingRequestFeatures",
    "TRAINING_SERVING_SKEW_AUDIT", "TrainingServingAuditEntry", "approximate_input_tokens",
    "constraint_indicator_count", "reasoning_indicator_count", "request_text",
    "text_contains_code",
]
