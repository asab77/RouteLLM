from decimal import Decimal

import pytest
from pydantic import ValidationError

from adaptive_llm_gateway.models import InferenceRequest, InferenceResponse, ModelConfig


def test_config_defaults_and_decimal_prices(model):
    assert model.enabled is True
    assert model.input_cost_per_1m_tokens == Decimal("0.15")
    with pytest.raises(ValidationError):
        model.enabled = False


@pytest.mark.parametrize("field,value", [
    ("model_id", "  "), ("provider", ""), ("provider_model_name", "\t"),
    ("input_cost_per_1m_tokens", "-1"), ("output_cost_per_1m_tokens", "-0.01"),
    ("input_cost_per_1m_tokens", "NaN"), ("output_cost_per_1m_tokens", "Infinity"),
    ("context_window", 0), ("context_window", -1), ("context_window", True),
    ("context_window", 1.5), ("enabled", "yes"), ("unknown", "value"),
])
def test_invalid_config(model, field, value):
    with pytest.raises(ValidationError):
        ModelConfig(**(model.model_dump() | {field: value}))


def test_free_model_and_normalized_identifiers(model):
    config = ModelConfig(**(model.model_dump() | {
        "model_id": " free ", "input_cost_per_1m_tokens": "0",
        "output_cost_per_1m_tokens": "0",
    }))
    assert config.model_id == "free"
    assert config.output_cost_per_1m_tokens == Decimal(0)


@pytest.mark.parametrize("overrides", [
    {"prompt": ""}, {"prompt": " \n"}, {"system_prompt": " "},
    {"max_output_tokens": 0}, {"max_output_tokens": -1},
    {"max_output_tokens": True}, {"max_output_tokens": 1.5},
    {"temperature": -0.1}, {"temperature": 2.1},
    {"temperature": float("nan")}, {"temperature": float("inf")},
    {"unexpected": 1},
])
def test_invalid_requests(overrides):
    with pytest.raises(ValidationError):
        InferenceRequest(**({"prompt": "hello"} | overrides))


def test_request_preserves_text_and_accepts_temperature_boundaries():
    for temperature in (0, 2):
        request = InferenceRequest(prompt="  hello\n", temperature=temperature)
        assert request.prompt == "  hello\n"
        assert request.system_prompt is None
        assert request.max_output_tokens == 256


@pytest.mark.parametrize("field,value", [
    ("input_tokens", -1), ("output_tokens", True), ("output_tokens", 1.5),
    ("latency_ms", -1), ("latency_ms", float("inf")),
    ("estimated_cost_usd", "-1"), ("estimated_cost_usd", "NaN"),
])
def test_invalid_response_metadata(field, value):
    data = dict(text="", model_id="test", provider="fake", input_tokens=0,
                output_tokens=0, latency_ms=0, estimated_cost_usd="0")
    with pytest.raises(ValidationError):
        InferenceResponse(**(data | {field: value}))
