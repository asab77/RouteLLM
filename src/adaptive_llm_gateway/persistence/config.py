import os

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator
from sqlalchemy.engine import make_url


class DatabaseSettings(BaseModel):
    model_config = ConfigDict(hide_input_in_errors=True)

    database_url: SecretStr | None = None
    telemetry_timeout_seconds: float = Field(default=2.0, gt=0, le=30, allow_inf_nan=False)

    @field_validator("database_url")
    @classmethod
    def validate_url(cls, value: SecretStr | None) -> SecretStr | None:
        if value is None:
            return None
        try:
            url = make_url(value.get_secret_value())
            if url.drivername != "postgresql+asyncpg" or not url.host or not url.database or not url.username:
                raise ValueError()
            if url.port is not None and not 1 <= url.port <= 65535:
                raise ValueError()
        except Exception:
            raise ValueError("DATABASE_URL must be a postgresql+asyncpg URL with user, host and database") from None
        return value

    @classmethod
    def from_environment(cls) -> "DatabaseSettings":
        return cls(database_url=os.environ.get("DATABASE_URL"),
                   telemetry_timeout_seconds=os.environ.get("TELEMETRY_TIMEOUT_SECONDS", "2"))
