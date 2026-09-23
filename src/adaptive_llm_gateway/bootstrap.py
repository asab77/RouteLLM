"""Offline development configuration, with synthetic prices."""

from adaptive_llm_gateway.application.service import InferenceService
from adaptive_llm_gateway.models import ModelConfig
from adaptive_llm_gateway.providers import FakeProvider
from adaptive_llm_gateway.providers.resolver import ProviderResolver
from adaptive_llm_gateway.registry import ModelRegistry


def create_development_service() -> InferenceService:
    registry = ModelRegistry()
    for model_id, input_rate, output_rate, context_window in (
        ("fake-small", "0.15", "0.60", 4096),
        ("fake-large", "1.00", "3.00", 16384),
    ):
        registry.register(ModelConfig(
            model_id=model_id,
            provider="fake",
            provider_model_name="echo-v1",
            input_cost_per_1m_tokens=input_rate,
            output_cost_per_1m_tokens=output_rate,
            context_window=context_window,
        ))
    resolver = ProviderResolver()
    resolver.register("fake", FakeProvider)
    return InferenceService(registry, resolver)
