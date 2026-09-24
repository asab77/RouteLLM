"""Composition and resource lifecycle, separate from HTTP and domain logic."""

import asyncio
import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from adaptive_llm_gateway.application.service import InferenceService
from adaptive_llm_gateway.bootstrap import create_development_service, configure_gateway
from adaptive_llm_gateway.providers.gateway_config import GatewaySettings
from adaptive_llm_gateway.persistence.config import DatabaseSettings
from adaptive_llm_gateway.persistence.database import Database

logger = logging.getLogger(__name__)


@asynccontextmanager
async def application_service() -> AsyncIterator[InferenceService]:
    settings = DatabaseSettings.from_environment()
    service = create_development_service()
    configure_gateway(service, GatewaySettings.from_environment())
    if settings.database_url is None:
        logger.warning("telemetry_disabled: DATABASE_URL is not configured")
        yield service
        return
    database = Database(settings)
    service.telemetry = database.repository
    service.telemetry_timeout = settings.telemetry_timeout_seconds
    try:
        try:
            # Read-only probe checks connectivity and migrated schema; never creates it.
            async with asyncio.timeout(settings.telemetry_timeout_seconds):
                await database.repository.summary()
        except Exception:
            logger.warning("telemetry_startup_probe_failed: check database connectivity and migrations")
        yield service
    finally:
        await database.close()
