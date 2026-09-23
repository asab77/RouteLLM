from decimal import Decimal

import pytest

from adaptive_llm_gateway.bootstrap import create_development_service
from adaptive_llm_gateway.telemetry.contracts import TelemetryEvent, TelemetrySummary


class MemoryRepository:
    def __init__(self):
        self.events: list[TelemetryEvent] = []

    async def record(self, event):
        self.events.append(event)

    async def summary(self):
        return TelemetrySummary(
            total_requests=len(self.events),
            successful_requests=sum(event.success for event in self.events),
            failed_requests=sum(not event.success for event in self.events),
            total_estimated_cost_usd=sum((event.estimated_cost_usd or Decimal(0) for event in self.events), Decimal(0)),
            average_latency_ms=sum(event.latency_ms for event in self.events) / len(self.events) if self.events else None,
            total_input_tokens=sum(event.input_tokens or 0 for event in self.events),
            total_output_tokens=sum(event.output_tokens or 0 for event in self.events),
        )


@pytest.fixture
def repository():
    return MemoryRepository()


@pytest.fixture
def telemetry_service(repository):
    service = create_development_service()
    service.telemetry = repository
    return service
