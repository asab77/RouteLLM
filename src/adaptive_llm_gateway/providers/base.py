from abc import ABC, abstractmethod

from adaptive_llm_gateway.models import InferenceRequest, InferenceResponse


class LLMProvider(ABC):
    """Adapter bound to one model; concrete adapters own configuration.

    Implementations translate requests and normalize usage, timing, and cost.
    A future caller selects a configured adapter before invoking generation.
    """

    @abstractmethod
    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Generate a provider-independent result for one request."""
        raise NotImplementedError
