"""Direct, non-streaming HTTP adapter. No retries, routing, or SDK objects escape."""
import asyncio
import re
from time import perf_counter

import httpx
from pydantic import ValidationError

from adaptive_llm_gateway.errors import GatewayError, GatewayErrorCategory, ModelDisabledError
from adaptive_llm_gateway.models import InferenceRequest, InferenceResponse, ModelConfig
from adaptive_llm_gateway.pricing import calculate_cost
from .base import LLMProvider
from .gateway_config import GatewaySettings, UPSTREAM_PROVIDERS

ENDPOINT = "https://ai-gateway.vercel.sh/v1/chat/completions"
_SAFE_VALUE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$")


def _safe_value(value):
    return value if isinstance(value, str) and _SAFE_VALUE.fullmatch(value) else None


def _response_identifiers(response: httpx.Response) -> dict:
    diagnostics = {"http_status": response.status_code}
    for header, field in (("x-vercel-id", "vercel_request_id"),
                          ("x-request-id", "request_id"),
                          ("traceparent", "trace_id")):
        value = _safe_value(response.headers.get(header))
        if value is not None:
            diagnostics[field] = value
    return diagnostics


def _error_diagnostics(response: httpx.Response, body) -> dict:
    diagnostics = _response_identifiers(response)
    if not isinstance(body, dict):
        return diagnostics
    error = body.get("error")
    if isinstance(error, dict):
        for source, field in (("code", "gateway_error_code"), ("type", "gateway_error_type")):
            value = _safe_value(error.get(source))
            if value is not None:
                diagnostics[field] = value
        for key in ("provider_error", "providerError", "cause"):
            nested = error.get(key)
            if isinstance(nested, dict):
                value = _safe_value(nested.get("code") or nested.get("type"))
                if value is not None:
                    diagnostics["upstream_error_code"] = value
                    break
    return diagnostics


def _completion_diagnostics(response: httpx.Response, choice, message, usage,
                            output_token_limit: int, *, input_tokens: int,
                            output_tokens: int, latency_ms: float,
                            estimated_cost_usd) -> dict:
    diagnostics = _response_identifiers(response)
    finish_reason = _safe_value(choice.get("finish_reason")) if isinstance(choice, dict) else None
    if finish_reason is not None:
        diagnostics["finish_reason"] = finish_reason
    diagnostics["content_empty"] = True
    diagnostics["output_token_limit"] = output_token_limit
    diagnostics["input_tokens"] = input_tokens
    diagnostics["output_tokens"] = output_tokens
    diagnostics["adapter_latency_ms"] = latency_ms
    diagnostics["estimated_cost_usd"] = str(estimated_cost_usd)
    completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
    if type(completion_tokens) is int and completion_tokens >= 0:
        diagnostics["completion_tokens"] = completion_tokens
    details = usage.get("completion_tokens_details") if isinstance(usage, dict) else None
    if isinstance(details, dict):
        reasoning_tokens = details.get("reasoning_tokens")
        if type(reasoning_tokens) is int and reasoning_tokens >= 0:
            diagnostics["reasoning_tokens"] = reasoning_tokens
    if isinstance(message, dict):
        diagnostics["reasoning_field_present"] = any(
            bool(message.get(key)) for key in ("reasoning", "reasoning_content", "reasoning_details"))
    return diagnostics


class VercelGatewayProvider(LLMProvider):
    def __init__(self, model: ModelConfig, settings: GatewaySettings,
                 *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        if model.provider != "vercel" or model.provider_model_name not in UPSTREAM_PROVIDERS:
            raise GatewayError(GatewayErrorCategory.INVALID_MODEL)
        self.model, self.settings, self.transport = model, settings, transport

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        if not self.model.enabled:
            raise ModelDisabledError("Model is disabled")
        if self.settings.api_key is None or not self.settings.api_key.get_secret_value().strip():
            raise GatewayError(GatewayErrorCategory.NOT_CONFIGURED)
        messages = []
        if request.system_prompt is not None:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})
        payload = {"model": self.model.provider_model_name, "messages": messages, "stream": False,
                   "max_tokens": request.max_output_tokens,
                   "providerOptions": {"gateway": {"only": [UPSTREAM_PROVIDERS[self.model.provider_model_name]]}}}
        if self.model.capabilities.supports_temperature:
            payload["temperature"] = request.temperature
        if self.model.reasoning_effort is not None:
            payload["reasoning"] = {"effort": self.model.reasoning_effort.value}
        started = perf_counter()
        try:
            async with asyncio.timeout(self.settings.timeout_seconds):
                async with httpx.AsyncClient(timeout=httpx.Timeout(self.settings.timeout_seconds),
                        transport=self.transport, follow_redirects=False, trust_env=False) as client:
                    response = await client.post(ENDPOINT, json=payload,
                        headers={"Authorization": "Bearer " + self.settings.api_key.get_secret_value()})
        except (httpx.TimeoutException, TimeoutError):
            raise GatewayError(GatewayErrorCategory.TIMEOUT) from None
        except httpx.RequestError:
            raise GatewayError(GatewayErrorCategory.UPSTREAM) from None
        latency = (perf_counter() - started) * 1000
        if not response.is_success:
            status_categories = {401: GatewayErrorCategory.AUTHENTICATION, 403: GatewayErrorCategory.AUTHENTICATION,
                404: GatewayErrorCategory.INVALID_MODEL, 429: GatewayErrorCategory.RATE_LIMIT,
                408: GatewayErrorCategory.TIMEOUT, 504: GatewayErrorCategory.TIMEOUT}
            category = status_categories.get(response.status_code, GatewayErrorCategory.UPSTREAM)
            try:
                body = response.json()
            except ValueError:
                body = None
            diagnostics = _error_diagnostics(response, body)
            if response.status_code in (400, 422):
                category = GatewayErrorCategory.INVALID_REQUEST
                code = diagnostics.get("gateway_error_code")
                if code in ("model_not_found", "invalid_model"):
                    category = GatewayErrorCategory.INVALID_MODEL
                elif code in ("context_length_exceeded", "context_window_exceeded"):
                    category = GatewayErrorCategory.CONTEXT_LIMIT
            raise GatewayError(category, diagnostics=diagnostics)
        try:
            body = response.json()
            choice = body["choices"][0]
            message = choice["message"]
            text = message["content"]
            if not isinstance(text, str):
                raise ValueError()
            usage = body.get("usage")
            if not isinstance(usage, dict) or any(k not in usage for k in ("prompt_tokens", "completion_tokens")):
                raise GatewayError(GatewayErrorCategory.MISSING_USAGE,
                                   diagnostics=_response_identifiers(response))
            inputs, outputs = usage["prompt_tokens"], usage["completion_tokens"]
            if any(type(count) is not int or count < 0 for count in (inputs, outputs)):
                raise ValueError()
            estimated_cost = calculate_cost(
                input_tokens=inputs, output_tokens=outputs, model=self.model)
            if not text.strip():
                raise GatewayError(GatewayErrorCategory.EMPTY_RESPONSE,
                    diagnostics=_completion_diagnostics(
                        response, choice, message, usage, request.max_output_tokens,
                        input_tokens=inputs, output_tokens=outputs, latency_ms=latency,
                        estimated_cost_usd=estimated_cost))
            return InferenceResponse(text=text, model_id=self.model.model_id, provider=self.model.provider,
                input_tokens=inputs, output_tokens=outputs, latency_ms=latency,
                estimated_cost_usd=estimated_cost)
        except GatewayError:
            raise
        except (KeyError, IndexError, TypeError, ValueError, ValidationError):
            raise GatewayError(GatewayErrorCategory.MALFORMED_RESPONSE,
                               diagnostics=_response_identifiers(response)) from None
