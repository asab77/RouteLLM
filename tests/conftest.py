import pytest

from adaptive_llm_gateway.models import ModelConfig


@pytest.fixture(autouse=True)
def isolate_database_environment(monkeypatch):
    """Normal tests must never write to a developer's configured application DB.

    Integration tests opt in using the separate TEST_DATABASE_URL variable.
    Configuration-specific tests can explicitly set these values themselves.
    """
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("TELEMETRY_TIMEOUT_SECONDS", raising=False)


@pytest.fixture
def model() -> ModelConfig:
    return ModelConfig(
        model_id="fake-small",
        provider="fake",
        provider_model_name="echo-v1",
        input_cost_per_1m_tokens="0.15",
        output_cost_per_1m_tokens="0.60",
        context_window=4096,
    )
