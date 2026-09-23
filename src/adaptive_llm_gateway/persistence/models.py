from datetime import datetime
from decimal import Decimal
from uuid import UUID

from sqlalchemy import BigInteger, Boolean, CheckConstraint, DateTime, Double, Index, Numeric, String, Uuid
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class InferenceTelemetry(Base):
    __tablename__ = "inference_telemetry"
    __table_args__ = (
        CheckConstraint("input_tokens IS NULL OR input_tokens >= 0", name="ck_input_tokens"),
        CheckConstraint("output_tokens IS NULL OR output_tokens >= 0", name="ck_output_tokens"),
        CheckConstraint("estimated_cost_usd IS NULL OR estimated_cost_usd >= 0", name="ck_cost"),
        CheckConstraint("latency_ms >= 0", name="ck_latency"),
        CheckConstraint("max_output_tokens > 0", name="ck_max_output"),
        CheckConstraint("temperature >= 0 AND temperature <= 2", name="ck_temperature"),
        CheckConstraint("prompt_characters >= 0 AND system_prompt_characters >= 0", name="ck_lengths"),
        CheckConstraint("(success AND error_category IS NULL AND input_tokens IS NOT NULL AND output_tokens IS NOT NULL AND estimated_cost_usd IS NOT NULL) OR (NOT success AND error_category IS NOT NULL)", name="ck_outcome"),
        Index("ix_telemetry_request_id", "request_id"),
        Index("ix_telemetry_model_created", "model_id", "created_at"),
        Index("ix_telemetry_created", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    request_id: Mapped[str] = mapped_column(String(128))
    model_id: Mapped[str] = mapped_column(String)
    provider: Mapped[str] = mapped_column(String)
    input_tokens: Mapped[int | None] = mapped_column(BigInteger)
    output_tokens: Mapped[int | None] = mapped_column(BigInteger)
    latency_ms: Mapped[float] = mapped_column(Double)
    # Unconstrained PostgreSQL NUMERIC avoids rounding exact sub-cent domain costs.
    estimated_cost_usd: Mapped[Decimal | None] = mapped_column(Numeric(asdecimal=True))
    success: Mapped[bool] = mapped_column(Boolean)
    error_category: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    max_output_tokens: Mapped[int] = mapped_column(BigInteger)
    temperature: Mapped[float] = mapped_column(Double)
    prompt_characters: Mapped[int] = mapped_column(BigInteger)
    system_prompt_characters: Mapped[int] = mapped_column(BigInteger)
