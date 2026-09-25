"""Environment settings and reviewed model portfolio, never fetched at startup."""
import os
from pydantic import BaseModel, ConfigDict, Field, SecretStr
from adaptive_llm_gateway.models import (
    CategoryOutputTokenAllowance, ModelCapabilities, ModelConfig,
    OutputTokenPolicy, ReasoningBehavior, ReasoningEffort,
)


class GatewaySettings(BaseModel):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)
    api_key: SecretStr | None = None
    timeout_seconds: float = Field(default=60, gt=0, le=300, allow_inf_nan=False)

    @classmethod
    def from_environment(cls) -> "GatewaySettings":
        key = os.environ.get("AI_GATEWAY_API_KEY", "").strip()
        return cls(api_key=SecretStr(key) if key else None,
                   timeout_seconds=os.environ.get("AI_GATEWAY_TIMEOUT_SECONDS", "60"))


# Verified against the public Vercel /v1/models catalog immediately before
# Phase 5.5C-B collection. Standard uncached text prices, USD per 1M tokens.
PRICING_SOURCE = "https://ai-gateway.vercel.sh/v1/models"
PRICING_VERIFIED = "2026-09-24"
CATALOG_VERIFIED_AT = "2026-09-24T03:06:54Z"

# Historical Phase 4/5 models remain registered for API and artifact compatibility.
LEGACY_MODELS = (
    ModelConfig(model_id="gateway-nano", provider="vercel", provider_model_name="openai/gpt-4.1-nano",
                input_cost_per_1m_tokens="0.10", output_cost_per_1m_tokens="0.40", context_window=1047576,
                capabilities=ModelCapabilities(supports_structured_output=True)),
    ModelConfig(model_id="gateway-mini", provider="vercel", provider_model_name="openai/gpt-4.1-mini",
                input_cost_per_1m_tokens="0.40", output_cost_per_1m_tokens="1.60", context_window=1047576,
                capabilities=ModelCapabilities(supports_structured_output=True)),
    ModelConfig(model_id="gateway-sonnet", provider="vercel", provider_model_name="anthropic/claude-sonnet-4.6",
                input_cost_per_1m_tokens="3.00", output_cost_per_1m_tokens="15.00", context_window=1000000,
                capabilities=ModelCapabilities(supports_structured_output=True,
                    reasoning=ReasoningBehavior.PROVIDER_DEFAULT)),
)

# Frozen Foundation V3 routing candidates. Nemotron disables reasoning; the
# remaining candidates use the lowest catalog-supported explicit effort.
CANDIDATE_MODELS = (
    ModelConfig(model_id="candidate-nemotron-3.5-lightning", provider="vercel",
                provider_model_name="nvidia/nemotron-3.5-lightning",
                input_cost_per_1m_tokens="0.05", output_cost_per_1m_tokens="0.15",
                context_window=262144,
                capabilities=ModelCapabilities(supports_temperature=True,
                    supports_structured_output=True,
                    reasoning=ReasoningBehavior.PROVIDER_DEFAULT),
                reasoning_effort=ReasoningEffort.NONE),
    ModelConfig(model_id="candidate-gpt-6-luna", provider="vercel",
                provider_model_name="openai/gpt-6-luna",
                input_cost_per_1m_tokens="0.10", output_cost_per_1m_tokens="0.50",
                context_window=1050000,
                capabilities=ModelCapabilities(supports_temperature=True,
                    supports_structured_output=True,
                    reasoning=ReasoningBehavior.PROVIDER_DEFAULT),
                reasoning_effort=ReasoningEffort.LOW),
    ModelConfig(model_id="candidate-gemini-3-flash", provider="vercel",
                provider_model_name="google/gemini-3-flash",
                input_cost_per_1m_tokens="0.50", output_cost_per_1m_tokens="3.00",
                context_window=1000000,
                capabilities=ModelCapabilities(supports_temperature=True,
                    supports_structured_output=True,
                    reasoning=ReasoningBehavior.PROVIDER_DEFAULT),
                reasoning_effort=ReasoningEffort.LOW,
                output_token_policy=OutputTokenPolicy(category_overrides=(
                    CategoryOutputTokenAllowance(
                        category="reasoning", max_output_tokens=256),
                ))),
    ModelConfig(model_id="candidate-claude-sonnet-5", provider="vercel",
                provider_model_name="anthropic/claude-sonnet-5",
                input_cost_per_1m_tokens="2.00", output_cost_per_1m_tokens="10.00",
                context_window=1000000,
                capabilities=ModelCapabilities(supports_temperature=False,
                    supports_structured_output=True,
                    reasoning=ReasoningBehavior.PROVIDER_DEFAULT),
                reasoning_effort=ReasoningEffort.LOW),
)

# These models are available only to explicitly enabled semantic-judge workflows.
JUDGE_SELECTION_MODELS = (
    ModelConfig(model_id="judge-opus-5.5", provider="vercel",
                provider_model_name="anthropic/claude-opus-5.5",
                input_cost_per_1m_tokens="4.00", output_cost_per_1m_tokens="20.00",
                context_window=1000000,
                capabilities=ModelCapabilities(supports_temperature=False,
                    supports_structured_output=False, reasoning=ReasoningBehavior.ADAPTIVE)),
    ModelConfig(model_id="judge-gpt-6-astra", provider="vercel",
                provider_model_name="openai/gpt-6-astra",
                input_cost_per_1m_tokens="10.00", output_cost_per_1m_tokens="50.00",
                context_window=1050000,
                capabilities=ModelCapabilities(supports_temperature=False,
                    supports_structured_output=True,
                    reasoning=ReasoningBehavior.PROVIDER_DEFAULT)),
    ModelConfig(model_id="judge-fable-5.1", provider="vercel",
                provider_model_name="anthropic/claude-fable-5.1",
                input_cost_per_1m_tokens="10.00", output_cost_per_1m_tokens="50.00",
                context_window=1000000,
                capabilities=ModelCapabilities(supports_temperature=True,
                    supports_structured_output=True,
                    reasoning=ReasoningBehavior.PROVIDER_DEFAULT)),
)

REAL_MODELS = LEGACY_MODELS + CANDIDATE_MODELS
SEMANTIC_JUDGE_MODELS = REAL_MODELS + JUDGE_SELECTION_MODELS

# Single-provider hard allowlists prevent gateway provider fallback.
UPSTREAM_PROVIDERS = {
    "openai/gpt-4.1-nano": "openai",
    "openai/gpt-4.1-mini": "openai",
    "anthropic/claude-sonnet-4.6": "anthropic",
    "nvidia/nemotron-3.5-lightning": "runinfra",
    "openai/gpt-6-luna": "openai",
    "google/gemini-3-flash": "google",
    "anthropic/claude-sonnet-5": "anthropic",
    "anthropic/claude-opus-5.5": "anthropic",
    "openai/gpt-6-astra": "openai",
    "anthropic/claude-fable-5.1": "anthropic",
}
