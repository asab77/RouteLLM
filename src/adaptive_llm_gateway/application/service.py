import asyncio
import logging
from time import perf_counter
from uuid import uuid4

from adaptive_llm_gateway.telemetry.contracts import InferenceTelemetryRepository, TelemetryEvent
from adaptive_llm_gateway.errors import (
    ContextLimitError,
    ModelDisabledError,
    ProviderFailureError,
    ProviderUnavailableError,
)
from adaptive_llm_gateway.models import InferenceRequest, InferenceResponse, ModelConfig
from adaptive_llm_gateway.providers.resolver import ProviderResolver
from adaptive_llm_gateway.registry import ModelRegistry

logger = logging.getLogger(__name__)


class InferenceService:
    def __init__(self, registry: ModelRegistry, resolver: ProviderResolver,
                 telemetry: InferenceTelemetryRepository | None = None,
                 telemetry_timeout: float = 2.0) -> None:
        self.registry = registry
        self.resolver = resolver
        self.telemetry = telemetry
        self.telemetry_timeout = telemetry_timeout

    def list_models(self) -> list[ModelConfig]:
        """Only advertise enabled models with a registered adapter."""
        return [
            model for model in self.registry.list_enabled_models()
            if self.resolver.supports(model.provider)
        ]

    async def generate(self, model_id: str, request: InferenceRequest,
                       *, request_id: str | None = None) -> InferenceResponse:
        # Unknown IDs and HTTP validation errors are not model inference attempts.
        model = self.registry.get(model_id)
        correlation_id = request_id if request_id is not None else str(uuid4())
        started = perf_counter()
        try:
            result = await self._generate(model, request)
        except (ContextLimitError, ModelDisabledError, ProviderUnavailableError, ProviderFailureError) as exc:
            categories = {ContextLimitError: "context_limit_exceeded", ModelDisabledError: "model_disabled",
                          ProviderUnavailableError: "provider_unavailable", ProviderFailureError: "provider_failure"}
            category = next(value for kind, value in categories.items() if isinstance(exc, kind))
            await self._record(TelemetryEvent(
                request_id=correlation_id, model_id=model.model_id, provider=model.provider,
                success=False, error_category=category, latency_ms=(perf_counter() - started) * 1000,
                max_output_tokens=request.max_output_tokens, temperature=request.temperature,
                prompt_characters=len(request.prompt), system_prompt_characters=len(request.system_prompt or ""),
            ))
            raise
        await self._record(TelemetryEvent(
            request_id=correlation_id, model_id=result.model_id, provider=result.provider,
            success=True, input_tokens=result.input_tokens, output_tokens=result.output_tokens,
            latency_ms=result.latency_ms, estimated_cost_usd=result.estimated_cost_usd,
            max_output_tokens=request.max_output_tokens, temperature=request.temperature,
            prompt_characters=len(request.prompt), system_prompt_characters=len(request.system_prompt or ""),
        ))
        return result

    async def _record(self, event: TelemetryEvent) -> None:
        if self.telemetry is None:
            return
        try:
            async with asyncio.timeout(self.telemetry_timeout):
                await self.telemetry.record(event)
        except Exception:
            # Never emit exception text/tracebacks: driver errors may contain SQL values or credentials.
            logger.warning("telemetry_write_failed request_id=%s", event.request_id,
                           extra={"request_id": event.request_id, "event": "telemetry_write_failed"})

    async def _generate(self, model: ModelConfig, request: InferenceRequest) -> InferenceResponse:
        if not model.enabled:
            raise ModelDisabledError(f"Model is disabled: {model.model_id!r}")
        try:
            provider = self.resolver.resolve(model)
            response = await provider.generate(request)
            if not isinstance(response, InferenceResponse):
                raise TypeError("Provider returned an invalid response")
            if response.model_id != model.model_id or response.provider != model.provider:
                raise ValueError("Provider returned inconsistent model metadata")
            return response
        except (ContextLimitError, ModelDisabledError, ProviderUnavailableError):
            raise
        except Exception as exc:
            # Cancellation (BaseException) propagates; adapter details stay internal.
            raise ProviderFailureError("Provider could not complete inference") from exc
