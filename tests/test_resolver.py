import asyncio

import pytest

from adaptive_llm_gateway.application.service import InferenceService
from adaptive_llm_gateway.errors import ProviderFailureError, ProviderUnavailableError
from adaptive_llm_gateway.models import InferenceRequest
from adaptive_llm_gateway.providers import FakeProvider
from adaptive_llm_gateway.providers.resolver import ProviderResolver
from adaptive_llm_gateway.registry import ModelRegistry


@pytest.mark.asyncio
async def test_resolves_factory_with_selected_configuration(model):
    seen = []

    def factory(config):
        seen.append(config)
        return FakeProvider(config)

    resolver = ProviderResolver()
    resolver.register("fake", factory)
    assert resolver.supports("fake")
    response = await resolver.resolve(model).generate(InferenceRequest(prompt="test"))
    assert seen == [model]
    assert response.model_id == model.model_id


def test_missing_factory(model):
    resolver = ProviderResolver()
    assert not resolver.supports("fake")
    with pytest.raises(ProviderUnavailableError):
        resolver.resolve(model)


def test_duplicate_factory_preserves_registration(model):
    resolver = ProviderResolver()
    resolver.register("fake", FakeProvider)
    with pytest.raises(ValueError, match="already registered"):
        resolver.register("fake", lambda model: None)
    assert isinstance(resolver.resolve(model), FakeProvider)


def test_blank_provider_name():
    with pytest.raises(ValueError, match="blank"):
        ProviderResolver().register(" ", FakeProvider)


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_response", [None, "wrong-model"])
async def test_invalid_provider_response_is_failure(model, invalid_response):
    class InvalidProvider(FakeProvider):
        async def generate(self, request):
            if invalid_response is None:
                return None
            response = await super().generate(request)
            return response.model_copy(update={"model_id": invalid_response})

    registry = ModelRegistry()
    registry.register(model)
    resolver = ProviderResolver()
    resolver.register("fake", InvalidProvider)
    with pytest.raises(ProviderFailureError):
        await InferenceService(registry, resolver).generate(model.model_id, InferenceRequest(prompt="hello"))


@pytest.mark.asyncio
async def test_cancellation_propagates(model):
    class CancelledProvider(FakeProvider):
        async def generate(self, request):
            raise asyncio.CancelledError()

    registry = ModelRegistry()
    registry.register(model)
    resolver = ProviderResolver()
    resolver.register("fake", CancelledProvider)
    with pytest.raises(asyncio.CancelledError):
        await InferenceService(registry, resolver).generate(model.model_id, InferenceRequest(prompt="hello"))
