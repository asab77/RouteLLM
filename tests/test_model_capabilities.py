import json
from decimal import Decimal

import httpx
import pytest
from pydantic import SecretStr

from adaptive_llm_gateway.bootstrap import configure_gateway, create_development_service
from adaptive_llm_gateway.models import (
    InferenceRequest, ModelCapabilities, ModelConfig, ReasoningBehavior, ReasoningEffort,
)
from adaptive_llm_gateway.providers.gateway_config import (
    CANDIDATE_MODELS, GatewaySettings, JUDGE_SELECTION_MODELS, REAL_MODELS,
    SEMANTIC_JUDGE_MODELS, UPSTREAM_PROVIDERS,
)
from adaptive_llm_gateway.providers.vercel import VercelGatewayProvider
from adaptive_llm_gateway.registry import ModelRegistry


def response_body():
    return {"choices": [{"message": {"content": "{}"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


async def capture_payload(model: ModelConfig, request: InferenceRequest):
    captured = []

    def handler(http_request):
        captured.append(json.loads(http_request.content))
        return httpx.Response(200, json=response_body())

    provider = VercelGatewayProvider(
        model, GatewaySettings(api_key=SecretStr("test-only-key")),
        transport=httpx.MockTransport(handler),
    )
    response = await provider.generate(request)
    return captured[0], response


@pytest.mark.asyncio
async def test_temperature_capability_controls_serialization_without_nulls():
    supporting = CANDIDATE_MODELS[0]
    unsupported = JUDGE_SELECTION_MODELS[0]
    supported_payload, _ = await capture_payload(
        supporting, InferenceRequest(prompt="test", temperature=0.25))
    unsupported_payload, _ = await capture_payload(
        unsupported, InferenceRequest(prompt="test", temperature=0))

    assert supported_payload["temperature"] == 0.25
    assert "temperature" not in unsupported_payload
    assert all(value is not None for value in unsupported_payload.values())


@pytest.mark.asyncio
@pytest.mark.parametrize("model", JUDGE_SELECTION_MODELS)
async def test_judge_models_respect_temperature_and_omit_reasoning_configuration(model):
    payload, _ = await capture_payload(model, InferenceRequest(prompt="judge", temperature=0))
    if model.capabilities.supports_temperature:
        assert payload["temperature"] == 0
    else:
        assert "temperature" not in payload
    assert "reasoning" not in payload
    assert "include_reasoning" not in payload


@pytest.mark.asyncio
async def test_reasoning_effort_is_typed_and_configuration_driven(monkeypatch):
    nemotron, luna, gemini, sonnet = CANDIDATE_MODELS
    payload, _ = await capture_payload(
        nemotron, InferenceRequest(prompt="test", temperature=0))
    assert payload["reasoning"] == {"effort": "none"}
    assert payload["temperature"] == 0

    for model in (luna, gemini, sonnet):
        default_payload, _ = await capture_payload(
            model, InferenceRequest(prompt="test", temperature=0))
        assert "reasoning" not in default_payload
    assert "temperature" not in (await capture_payload(
        sonnet, InferenceRequest(prompt="test", temperature=0)))[0]

    synthetic = ModelConfig(
        model_id="future-reasoning", provider="vercel",
        provider_model_name="future/reasoning",
        input_cost_per_1m_tokens="1", output_cost_per_1m_tokens="2",
        context_window=4096,
        capabilities=ModelCapabilities(
            reasoning=ReasoningBehavior.PROVIDER_DEFAULT),
        reasoning_effort=ReasoningEffort.LOW,
    )
    monkeypatch.setitem(UPSTREAM_PROVIDERS, "future/reasoning", "future")
    synthetic_payload, _ = await capture_payload(
        synthetic, InferenceRequest(prompt="test"))
    assert synthetic_payload["reasoning"] == {"effort": "low"}


def test_explicit_reasoning_rejected_when_capability_is_unsupported():
    with pytest.raises(ValueError, match="requires reasoning support"):
        ModelConfig(
            model_id="unsupported", provider="vercel",
            provider_model_name="future/unsupported",
            input_cost_per_1m_tokens="1", output_cost_per_1m_tokens="2",
            context_window=4096, reasoning_effort=ReasoningEffort.NONE,
        )


def test_capabilities_survive_registry_and_judge_models_are_separate():
    registry = ModelRegistry()
    synthetic = ModelConfig(
        model_id="future-model", provider="vercel", provider_model_name="future/model",
        input_cost_per_1m_tokens="1.25", output_cost_per_1m_tokens="2.50",
        context_window=8192,
        capabilities=ModelCapabilities(
            supports_temperature=False, supports_structured_output=True,
            reasoning=ReasoningBehavior.PROVIDER_DEFAULT),
    )
    registry.register(synthetic)
    resolved = registry.get("future-model")
    assert resolved.capabilities == synthetic.capabilities
    assert resolved.capabilities.reasoning is ReasoningBehavior.PROVIDER_DEFAULT

    candidate_ids = {model.model_id for model in CANDIDATE_MODELS}
    judge_ids = {model.model_id for model in JUDGE_SELECTION_MODELS}
    assert judge_ids == {"judge-opus-5.5", "judge-gpt-6-astra", "judge-fable-5.1"}
    assert candidate_ids == {
        "candidate-nemotron-3.5-lightning", "candidate-gpt-6-luna",
        "candidate-gemini-3-flash", "candidate-claude-sonnet-5",
    }
    assert candidate_ids.isdisjoint(judge_ids)
    assert set(SEMANTIC_JUDGE_MODELS) == set(REAL_MODELS + JUDGE_SELECTION_MODELS)


def test_judge_capabilities_and_decimal_pricing_are_exact():
    models = {model.model_id: model for model in JUDGE_SELECTION_MODELS}
    opus = models["judge-opus-5.5"]
    astra = models["judge-gpt-6-astra"]
    fable = models["judge-fable-5.1"]

    assert opus.provider_model_name == "anthropic/claude-opus-5.5"
    assert opus.input_cost_per_1m_tokens == Decimal("4.00")
    assert opus.output_cost_per_1m_tokens == Decimal("20.00")
    assert opus.capabilities == ModelCapabilities(
        supports_temperature=False, supports_structured_output=False,
        reasoning=ReasoningBehavior.ADAPTIVE)

    assert astra.provider_model_name == "openai/gpt-6-astra"
    assert astra.input_cost_per_1m_tokens == Decimal("10.00")
    assert astra.output_cost_per_1m_tokens == Decimal("50.00")
    assert astra.capabilities == ModelCapabilities(
        supports_temperature=False, supports_structured_output=True,
        reasoning=ReasoningBehavior.PROVIDER_DEFAULT)

    assert fable.provider_model_name == "anthropic/claude-fable-5.1"
    assert fable.input_cost_per_1m_tokens == Decimal("10.00")
    assert fable.output_cost_per_1m_tokens == Decimal("50.00")
    assert fable.capabilities == ModelCapabilities(
        supports_temperature=True, supports_structured_output=True,
        reasoning=ReasoningBehavior.PROVIDER_DEFAULT)


@pytest.mark.asyncio
async def test_judge_cost_calculation_uses_existing_exact_accounting():
    expected = {
        "judge-opus-5.5": Decimal("0.000140"),
        "judge-gpt-6-astra": Decimal("0.000350"),
        "judge-fable-5.1": Decimal("0.000350"),
    }
    for model in JUDGE_SELECTION_MODELS:
        _, response = await capture_payload(model, InferenceRequest(prompt="judge"))
        assert response.estimated_cost_usd == expected[model.model_id]


@pytest.mark.asyncio
async def test_synthetic_future_capability_combination_needs_only_configuration(monkeypatch):
    model = ModelConfig(
        model_id="future-model", provider="vercel", provider_model_name="future/model-x",
        input_cost_per_1m_tokens="1", output_cost_per_1m_tokens="2", context_window=4096,
        capabilities=ModelCapabilities(
            supports_temperature=False, supports_structured_output=True,
            reasoning=ReasoningBehavior.PROVIDER_DEFAULT),
    )
    monkeypatch.setitem(UPSTREAM_PROVIDERS, "future/model-x", "future")
    payload, _ = await capture_payload(model, InferenceRequest(prompt="future", temperature=1))
    assert payload["model"] == "future/model-x"
    assert payload["providerOptions"]["gateway"] == {"only": ["future"]}
    assert "temperature" not in payload and "reasoning" not in payload


def test_normal_gateway_bootstrap_excludes_judge_only_models():
    service = create_development_service()
    configure_gateway(service, GatewaySettings(api_key=SecretStr("test-only-key")))
    registered = {model.model_id for model in service.list_models()}
    assert {model.model_id for model in CANDIDATE_MODELS} <= registered
    assert registered.isdisjoint({model.model_id for model in JUDGE_SELECTION_MODELS})
