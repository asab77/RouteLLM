"""Create metadata-only inference telemetry.

Revision ID: 0001
Revises: none
"""

from alembic import op
import sqlalchemy as sa

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "inference_telemetry",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("request_id", sa.String(128), nullable=False),
        sa.Column("model_id", sa.String(), nullable=False),
        sa.Column("provider", sa.String(), nullable=False),
        sa.Column("input_tokens", sa.BigInteger(), nullable=True),
        sa.Column("output_tokens", sa.BigInteger(), nullable=True),
        sa.Column("latency_ms", sa.Double(), nullable=False),
        sa.Column("estimated_cost_usd", sa.Numeric(asdecimal=True), nullable=True),
        sa.Column("success", sa.Boolean(), nullable=False),
        sa.Column("error_category", sa.String(64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("max_output_tokens", sa.BigInteger(), nullable=False),
        sa.Column("temperature", sa.Double(), nullable=False),
        sa.Column("prompt_characters", sa.BigInteger(), nullable=False),
        sa.Column("system_prompt_characters", sa.BigInteger(), nullable=False),
        sa.CheckConstraint("input_tokens IS NULL OR input_tokens >= 0", name="ck_input_tokens"),
        sa.CheckConstraint("output_tokens IS NULL OR output_tokens >= 0", name="ck_output_tokens"),
        sa.CheckConstraint("estimated_cost_usd IS NULL OR estimated_cost_usd >= 0", name="ck_cost"),
        sa.CheckConstraint("latency_ms >= 0", name="ck_latency"),
        sa.CheckConstraint("max_output_tokens > 0", name="ck_max_output"),
        sa.CheckConstraint("temperature >= 0 AND temperature <= 2", name="ck_temperature"),
        sa.CheckConstraint("prompt_characters >= 0 AND system_prompt_characters >= 0", name="ck_lengths"),
        sa.CheckConstraint("(success AND error_category IS NULL AND input_tokens IS NOT NULL AND output_tokens IS NOT NULL AND estimated_cost_usd IS NOT NULL) OR (NOT success AND error_category IS NOT NULL)", name="ck_outcome"),
    )
    op.create_index("ix_telemetry_request_id", "inference_telemetry", ["request_id"])
    op.create_index("ix_telemetry_model_created", "inference_telemetry", ["model_id", "created_at"])
    op.create_index("ix_telemetry_created", "inference_telemetry", ["created_at"])


def downgrade() -> None:
    op.drop_table("inference_telemetry")
