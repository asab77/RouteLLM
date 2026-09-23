from decimal import Decimal

import pytest

from adaptive_llm_gateway.models import InferenceRequest, InferenceResponse, ModelConfig
from adaptive_llm_gateway.providers import FakeProvider, LLMProvider
from adaptive_llm_gateway.registry import ModelRegistry


def test_contract_requires_generation_implementation():
    class IncompleteProvider(LLMProvider):
        pass

    with pytest.raises(TypeError):
        IncompleteProvider()


@pytest.mark.asyncio
async def test_fake_through_provider_contract_and_registry(model):
    registry = ModelRegistry()
    registry.register(model)
    provider: LLMProvider = FakeProvider(registry.get(model.model_id))
    request = InferenceRequest(prompt="one  two\nthree", system_prompt="be concise", max_output_tokens=2)
    response = await provider.generate(request)
    assert isinstance(response, InferenceResponse)
    assert response == await provider.generate(request)
    assert response.text == "one two"
    assert response.model_id == model.model_id
    assert response.provider == "fake"
    assert response.input_tokens == 5
    assert response.output_tokens == 2
    assert response.latency_ms == 0
    assert response.estimated_cost_usd == Decimal("0.00000195")


@pytest.mark.asyncio
async def test_actual_output_usage_and_exact_context_boundary(model):
    config = ModelConfig(**(model.model_dump() | {"context_window": 5}))
    response = await FakeProvider(config).generate(InferenceRequest(prompt="hello", max_output_tokens=4))
    assert response.text == "hello"
    assert response.input_tokens == response.output_tokens == 1
    assert response.estimated_cost_usd == Decimal("0.00000075")


@pytest.mark.asyncio
async def test_context_overflow_includes_system_prompt(model):
    config = ModelConfig(**(model.model_dump() | {"context_window": 3}))
    with pytest.raises(ValueError, match="context window"):
        await FakeProvider(config).generate(
            InferenceRequest(prompt="hello", system_prompt="system text", max_output_tokens=1)
        )


@pytest.mark.asyncio
async def test_disabled_model_cannot_generate(model):
    disabled = ModelConfig(**(model.model_dump() | {"enabled": False}))
    with pytest.raises(ValueError, match="disabled"):
        await FakeProvider(disabled).generate(InferenceRequest(prompt="hello"))


def test_fake_rejects_other_provider(model):
    config = ModelConfig(**(model.model_dump() | {"provider": "other"}))
    with pytest.raises(ValueError, match="provider='fake'"):
        FakeProvider(config)
