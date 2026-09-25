import ast
import inspect
from decimal import Decimal

import pytest
from pydantic import ValidationError

from adaptive_llm_gateway.api.schemas import InferencePayload
from adaptive_llm_gateway.benchmarks.features import extract_request_features
from adaptive_llm_gateway.benchmarks.models import BenchmarkTask
from adaptive_llm_gateway.models import InferenceRequest, ModelConfig
from adaptive_llm_gateway.pricing import calculate_projected_cost
from adaptive_llm_gateway.routing.features import (
    ACCEPTED_INTERACTION_FEATURES,
    CategoryProvenance,
    FeatureCompatibilityStatus,
    FORBIDDEN_ROUTING_FEATURES,
    LeakageStatus,
    PRODUCTION_FEATURE_GOVERNANCE,
    ProductionRequestFeatureExtractor,
    ROUTING_CATEGORY_TAXONOMY_VERSION,
    RoutingCategory,
    RoutingRequestFeatures,
    TRAINING_SERVING_SKEW_AUDIT,
)
from adaptive_llm_gateway.routing.ml_diagnostics import INTERACTION_FEATURE
from adaptive_llm_gateway.routing.ml_features import PREDICTIVE_FEATURES


def request(**updates) -> InferenceRequest:
    values = {
        "prompt": "Determine the result if x is 2 and return exactly one value.",
        "system_prompt": "Be precise.",
        "max_output_tokens": 64,
        "temperature": 0,
    }
    values.update(updates)
    return InferenceRequest(**values)


def extracted(**kwargs) -> RoutingRequestFeatures:
    return ProductionRequestFeatureExtractor().extract(request(), **kwargs)


def test_extraction_is_deterministic_and_identical_requests_match():
    extractor = ProductionRequestFeatureExtractor()
    first = extractor.extract(request(), category_hint=RoutingCategory.REASONING)
    second = extractor.extract(request(), category_hint=RoutingCategory.REASONING)
    assert first == second
    assert first.model_dump_json() == second.model_dump_json()


def test_approximate_tokens_use_shared_whitespace_semantics():
    features = ProductionRequestFeatureExtractor().extract(request(
        system_prompt="one two", prompt="three   four\nfive"
    ))
    assert features.approximate_input_tokens == 5


def test_character_lengths_message_count_and_system_presence():
    features = ProductionRequestFeatureExtractor().extract(request(
        prompt="abcd", system_prompt="xy"
    ))
    assert features.prompt_characters == 4
    assert features.system_prompt_characters == 2
    assert features.message_count == 2
    assert features.system_prompt_present is True


def test_no_system_prompt_is_one_message():
    features = ProductionRequestFeatureExtractor().extract(request(system_prompt=None))
    assert features.system_prompt_characters == 0
    assert features.message_count == 1
    assert features.system_prompt_present is False


@pytest.mark.parametrize("prompt", [
    "```python\ndef solve():\n    pass\n```",
    "Write python code for this.",
    "def solve(value): return value",
    "function solve(value) { return value; }",
    "class Solver handles this",
])
def test_code_indicator_positive_cases(prompt):
    assert ProductionRequestFeatureExtractor().extract(
        request(prompt=prompt, system_prompt=None)
    ).contains_code is True


def test_code_indicator_negative_case():
    assert ProductionRequestFeatureExtractor().extract(
        request(prompt="Explain how a computer program works.", system_prompt=None)
    ).contains_code is False


def test_structured_output_requires_explicit_configuration():
    extractor = ProductionRequestFeatureExtractor()
    mentions_json = request(prompt="Explain what JSON means.", system_prompt=None)
    assert extractor.extract(mentions_json).requests_structured_output is False
    assert extractor.extract(
        mentions_json, structured_output_required=True
    ).requests_structured_output is True


def test_invalid_structured_output_flag_is_rejected():
    with pytest.raises(TypeError):
        ProductionRequestFeatureExtractor().extract(
            request(), structured_output_required=1
        )


def test_reasoning_indicator_preserves_frozen_vocabulary():
    features = ProductionRequestFeatureExtractor().extract(request(
        prompt="Infer this if true; therefore calculate after the event.",
        system_prompt=None,
    ))
    assert features.reasoning_indicator_count == 5


def test_constraint_indicator_preserves_frozen_vocabulary():
    features = ProductionRequestFeatureExtractor().extract(request(
        prompt="You must return exactly one item only.", system_prompt=None
    ))
    assert features.constraint_indicator_count == 4


@pytest.mark.parametrize("category", list(RoutingCategory))
def test_every_valid_category_hint_is_accepted(category):
    features = extracted(category_hint=category)
    assert features.category is category
    assert features.category_provenance is CategoryProvenance.CLIENT_HINT


def test_valid_category_string_is_normalized_to_enum():
    features = extracted(category_hint="coding")
    assert features.category is RoutingCategory.CODING
    assert features.category_provenance is CategoryProvenance.CLIENT_HINT


def test_absent_category_is_explicit_without_substitution():
    features = extracted()
    assert features.category is None
    assert features.category_provenance is CategoryProvenance.ABSENT


@pytest.mark.parametrize("category", ["other", "json", "QA", "", "arbitrary-category"])
def test_invalid_or_arbitrary_category_hint_is_rejected(category):
    with pytest.raises(ValueError):
        extracted(category_hint=category)


def test_extractor_never_emits_inferred_provenance():
    outputs = [extracted(), *(extracted(category_hint=item) for item in RoutingCategory)]
    assert all(item.category_provenance is not CategoryProvenance.INFERRED for item in outputs)


def test_inferred_provenance_is_forward_compatible_but_requires_category():
    values = extracted(category_hint="qa").model_dump()
    inferred = RoutingRequestFeatures(**(
        values | {"category_provenance": CategoryProvenance.INFERRED}
    ))
    assert inferred.category_provenance is CategoryProvenance.INFERRED
    with pytest.raises(ValidationError):
        RoutingRequestFeatures(**(
            values | {"category": None, "category_provenance": CategoryProvenance.INFERRED}
        ))


def test_category_and_provenance_types_reject_invalid_values():
    with pytest.raises(ValueError):
        RoutingCategory("unsupported")
    with pytest.raises(ValueError):
        CategoryProvenance("guessed")


def test_taxonomy_is_exact_and_versioned():
    assert ROUTING_CATEGORY_TAXONOMY_VERSION == "phase7-v1"
    assert {item.value for item in RoutingCategory} == {
        "classification", "coding", "extraction", "structured_json", "qa",
        "reasoning", "summarization",
    }


def test_governance_covers_exact_production_contract():
    names = [item.name for item in PRODUCTION_FEATURE_GOVERNANCE]
    assert len(names) == len(set(names))
    assert set(names) == set(RoutingRequestFeatures.model_fields)
    assert all(item.leakage_status is LeakageStatus.PRE_GENERATION_ALLOWED
               for item in PRODUCTION_FEATURE_GOVERNANCE)


def test_production_contract_excludes_all_forbidden_fields():
    feature_names = set(RoutingRequestFeatures.model_fields)
    assert feature_names.isdisjoint(FORBIDDEN_ROUTING_FEATURES)
    for forbidden in (
        "quality_score", "latency_ms", "realized_cost_usd", "output_tokens",
        "analysis_difficulty", "acceptable", "ground_truth", "evaluator_name",
        "response_text", "provider_outcome", "estimated_cost_usd",
        "evaluation_status", "generated_at", "missing_label_reason",
    ):
        assert forbidden not in feature_names


def test_extra_post_generation_fields_are_rejected():
    values = extracted().model_dump()
    for forbidden in FORBIDDEN_ROUTING_FEATURES:
        with pytest.raises(ValidationError):
            RoutingRequestFeatures.model_validate(values | {forbidden: "forbidden"})


def test_projected_cost_uses_extracted_estimate_and_requested_allowance():
    features = ProductionRequestFeatureExtractor().extract(request(
        prompt="one two three", system_prompt=None, max_output_tokens=20
    ))
    model = ModelConfig(
        model_id="model-a", provider="local", provider_model_name="model-a",
        input_cost_per_1m_tokens="1.5", output_cost_per_1m_tokens="4.25",
        context_window=4096,
    )
    assert calculate_projected_cost(
        approximate_input_tokens=features.approximate_input_tokens,
        effective_max_output_tokens=features.max_output_tokens,
        model=model,
    ) == (Decimal(3) * Decimal("1.5") + Decimal(20) * Decimal("4.25")) / Decimal(1_000_000)


def test_skew_audit_covers_every_accepted_interaction_feature_once():
    names = [item.feature_name for item in TRAINING_SERVING_SKEW_AUDIT]
    assert len(names) == len(set(names))
    assert tuple(PREDICTIVE_FEATURES) + (INTERACTION_FEATURE,) == ACCEPTED_INTERACTION_FEATURES
    assert set(names) == set(ACCEPTED_INTERACTION_FEATURES)


def test_skew_audit_uses_only_controlled_statuses():
    assert {item.status for item in TRAINING_SERVING_SKEW_AUDIT} <= set(FeatureCompatibilityStatus)
    assert {item.status for item in TRAINING_SERVING_SKEW_AUDIT} == set(FeatureCompatibilityStatus)


@pytest.mark.parametrize(("category", "expected"), [
    ("qa", RoutingCategory.QA),
    ("json", RoutingCategory.STRUCTURED_JSON),
    ("coding", RoutingCategory.CODING),
])
def test_phase7_shared_extraction_semantics_remain_unchanged(category, expected):
    task = BenchmarkTask(
        task_id="compatibility-task",
        category=category,
        expected_output_type="json" if category == "json" else "text",
        prompt="If needed, return exactly this: def solve(x): return x",
        system_prompt="Determine the result.",
        max_output_tokens=42,
    )
    old = extract_request_features(task)
    new = ProductionRequestFeatureExtractor().extract(
        task.to_request(),
        category_hint=expected,
        structured_output_required=task.expected_output_type in {"json", "code"},
    )
    assert new.prompt_characters == old.prompt_characters
    assert new.system_prompt_characters == old.system_prompt_characters
    assert new.approximate_input_tokens == old.approximate_input_tokens
    assert new.contains_code == old.contains_code
    assert new.requests_structured_output == old.requests_structured_output
    assert new.max_output_tokens == old.max_output_tokens
    assert new.constraint_indicator_count == old.constraint_indicator_count
    assert new.reasoning_indicator_count == old.reasoning_indicator_count


def test_explicit_model_api_contract_is_unchanged():
    assert set(InferencePayload.model_fields) == {
        "prompt", "system_prompt", "max_output_tokens", "temperature", "model_id"
    }
    assert InferencePayload.model_fields["model_id"].is_required()
    assert "category" not in InferencePayload.model_fields


def test_extractor_source_has_no_provider_network_ml_or_benchmark_dependency():
    import adaptive_llm_gateway.routing.features as feature_module

    tree = ast.parse(inspect.getsource(feature_module))
    imports = {
        node.module or "" for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    forbidden = ("benchmark", "provider", "httpx", "sklearn", "numpy", "foundation")
    assert not any(any(term in name.lower() for term in forbidden) for name in imports)
    source = inspect.getsource(feature_module).lower()
    assert "foundation" not in source
    assert "benchmark-results" not in source
    assert "socket" not in source


def test_serialization_is_stable_across_repeated_extraction():
    extractor = ProductionRequestFeatureExtractor()
    serializations = {
        extractor.extract(
            request(), category_hint="classification", structured_output_required=True
        ).model_dump_json()
        for _ in range(10)
    }
    assert len(serializations) == 1


def test_construction_order_does_not_change_serialization():
    values = extracted(category_hint="summarization").model_dump()
    reversed_values = dict(reversed(tuple(values.items())))
    assert RoutingRequestFeatures(**values).model_dump_json() == (
        RoutingRequestFeatures(**reversed_values).model_dump_json()
    )
