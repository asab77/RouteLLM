from adaptive_llm_gateway.errors import ContextLimitError, ModelDisabledError
from adaptive_llm_gateway.models import InferenceRequest, InferenceResponse, ModelConfig
from adaptive_llm_gateway.pricing import calculate_cost

from .base import LLMProvider


class FakeProvider(LLMProvider):
    """Offline echo adapter with whitespace tokens and zero simulated latency.

    System text counts toward input usage but is not echoed. Temperature has no
    effect. This tokenizer is a test convention, not a real provider estimate.
    """

    def __init__(self, model: ModelConfig) -> None:
        if model.provider != "fake":
            raise ValueError("FakeProvider requires provider='fake'")
        self._model = model

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        if not self._model.enabled:
            raise ModelDisabledError(f"Model is disabled: {self._model.model_id!r}")
        words = request.prompt.split()
        input_tokens = len(words) + len((request.system_prompt or "").split())
        if input_tokens + request.max_output_tokens > self._model.context_window:
            raise ContextLimitError("Input plus requested output exceeds the model context window")
        output_words = words[: request.max_output_tokens]
        output_tokens = len(output_words)
        return InferenceResponse(
            text=" ".join(output_words),
            model_id=self._model.model_id,
            provider=self._model.provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=0.0,
            estimated_cost_usd=calculate_cost(
                input_tokens=input_tokens, output_tokens=output_tokens, model=self._model
            ),
        )
