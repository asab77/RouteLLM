from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Protocol
from uuid import UUID, uuid4

from pydantic import BaseModel


@dataclass(frozen=True, kw_only=True)
class TelemetryEvent:
    request_id: str
    model_id: str
    provider: str
    success: bool
    latency_ms: float
    max_output_tokens: int
    temperature: float
    prompt_characters: int
    system_prompt_characters: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    estimated_cost_usd: Decimal | None = None
    error_category: str | None = None
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class TelemetrySummary(BaseModel):
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_estimated_cost_usd: Decimal = Decimal(0)
    average_latency_ms: float | None = None
    total_input_tokens: int = 0
    total_output_tokens: int = 0


class InferenceTelemetryRepository(Protocol):
    async def record(self, event: TelemetryEvent) -> None: ...

    async def summary(self) -> TelemetrySummary: ...
