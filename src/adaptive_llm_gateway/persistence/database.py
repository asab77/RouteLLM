from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from .config import DatabaseSettings
from .repository import PostgresTelemetryRepository


class Database:
    def __init__(self, settings: DatabaseSettings) -> None:
        if settings.database_url is None:
            raise ValueError("DATABASE_URL is required")
        self.engine = create_async_engine(
            settings.database_url.get_secret_value(), pool_pre_ping=True,
            hide_parameters=True, echo=False,
            connect_args={"timeout": settings.telemetry_timeout_seconds,
                          "command_timeout": settings.telemetry_timeout_seconds},
        )
        self.repository = PostgresTelemetryRepository(
            async_sessionmaker(self.engine, expire_on_commit=False)
        )

    async def close(self) -> None:
        await self.engine.dispose()
