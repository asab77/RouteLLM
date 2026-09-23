import asyncio
import logging

from .contracts import InferenceTelemetryRepository, TelemetrySummary

logger = logging.getLogger(__name__)


class TelemetryUnavailableError(RuntimeError):
    """Metrics are unavailable; never present missing data as zero activity."""


class TelemetryQueryService:
    def __init__(self, repository: InferenceTelemetryRepository | None, timeout: float = 2.0) -> None:
        self.repository = repository
        self.timeout = timeout

    async def summary(self) -> TelemetrySummary:
        if self.repository is None:
            raise TelemetryUnavailableError()
        try:
            async with asyncio.timeout(self.timeout):
                return await self.repository.summary()
        except Exception:
            logger.warning("telemetry_query_failed")
            raise TelemetryUnavailableError() from None
