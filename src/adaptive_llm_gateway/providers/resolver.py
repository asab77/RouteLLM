"""Small provider-name to model-bound adapter factory registry."""

from collections.abc import Callable

from adaptive_llm_gateway.errors import ProviderUnavailableError
from adaptive_llm_gateway.models import ModelConfig

from .base import LLMProvider

type ProviderFactory = Callable[[ModelConfig], LLMProvider]


class ProviderResolver:
    """Factories must be cheap and perform no blocking network I/O."""

    def __init__(self) -> None:
        self._factories: dict[str, ProviderFactory] = {}

    def register(self, provider: str, factory: ProviderFactory) -> None:
        provider = provider.strip()
        if not provider:
            raise ValueError("Provider name must not be blank")
        if provider in self._factories:
            raise ValueError(f"Provider already registered: {provider!r}")
        self._factories[provider] = factory

    def supports(self, provider: str) -> bool:
        return provider in self._factories

    def resolve(self, model: ModelConfig) -> LLMProvider:
        try:
            factory = self._factories[model.provider]
        except KeyError:
            raise ProviderUnavailableError("No provider adapter is registered") from None
        return factory(model)
