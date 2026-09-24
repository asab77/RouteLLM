"""Offline development configuration, with synthetic prices."""

from adaptive_llm_gateway.application.service import InferenceService
from adaptive_llm_gateway.models import ModelConfig
from adaptive_llm_gateway.providers import FakeProvider
from adaptive_llm_gateway.providers.resolver import ProviderResolver
from adaptive_llm_gateway.registry import ModelRegistry
from adaptive_llm_gateway.providers.gateway_config import GatewaySettings


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


def configure_gateway(service: InferenceService, settings: GatewaySettings) -> None:
    """Real models are registered only with credentials; offline bootstrap stays unchanged."""
    from adaptive_llm_gateway.providers.gateway_config import REAL_MODELS
    from adaptive_llm_gateway.providers.vercel import VercelGatewayProvider
    if settings.api_key is None:
        return
    for model in REAL_MODELS:
        service.registry.register(model)
    service.resolver.register("vercel", lambda model: VercelGatewayProvider(model, settings))
