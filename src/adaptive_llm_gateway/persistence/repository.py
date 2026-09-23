from dataclasses import asdict

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from adaptive_llm_gateway.telemetry.contracts import TelemetryEvent, TelemetrySummary

from .models import InferenceTelemetry as Row


class PostgresTelemetryRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self.sessions = sessions

    async def record(self, event: TelemetryEvent) -> None:
        # begin commits on success, rolls back on failure, and closes the session.
        async with self.sessions.begin() as session:
            session.add(Row(**asdict(event)))

    async def summary(self) -> TelemetrySummary:
        statement = select(
            func.count().label("total_requests"),
            func.count().filter(Row.success.is_(True)).label("successful_requests"),
            func.count().filter(Row.success.is_(False)).label("failed_requests"),
            func.coalesce(func.sum(Row.estimated_cost_usd), 0).label("total_estimated_cost_usd"),
            func.avg(Row.latency_ms).label("average_latency_ms"),
            func.coalesce(func.sum(Row.input_tokens), 0).label("total_input_tokens"),
            func.coalesce(func.sum(Row.output_tokens), 0).label("total_output_tokens"),
        ).select_from(Row)
        async with self.sessions() as session:
            result = await session.execute(statement)
            return TelemetrySummary(**result.mappings().one())
