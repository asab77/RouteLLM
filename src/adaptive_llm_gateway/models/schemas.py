"""Validated, immutable configuration and inference data."""

from decimal import Decimal
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

Identifier = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Money = Annotated[Decimal, Field(ge=0, allow_inf_nan=False)]
TokenCount = Annotated[int, Field(ge=0, strict=True)]
PositiveTokenCount = Annotated[int, Field(gt=0, strict=True)]


class DomainModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelConfig(DomainModel):
    """Public model metadata. Prices are USD per one million tokens."""

    model_id: Identifier
    provider: Identifier
    provider_model_name: Identifier
    input_cost_per_1m_tokens: Money
    output_cost_per_1m_tokens: Money
    context_window: PositiveTokenCount
    enabled: bool = Field(default=True, strict=True)


class InferenceRequest(DomainModel):
    """Single-turn input; adapters map this schema to provider requests."""

    prompt: str = Field(min_length=1)
    system_prompt: str | None = None
    max_output_tokens: PositiveTokenCount = 256
    temperature: float = Field(default=1.0, ge=0, le=2, allow_inf_nan=False)

    @field_validator("prompt", "system_prompt")
    @classmethod
    def reject_blank_text(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("prompt text must contain non-whitespace characters")
        return value


class InferenceResponse(DomainModel):
    """Normalized generation result. Cost is an estimate in USD."""

    text: str
    model_id: Identifier
    provider: Identifier
    input_tokens: TokenCount
    output_tokens: TokenCount
    latency_ms: float = Field(ge=0, allow_inf_nan=False)
    estimated_cost_usd: Money
