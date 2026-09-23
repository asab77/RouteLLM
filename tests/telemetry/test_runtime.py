from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from adaptive_llm_gateway.api.app import create_app
from adaptive_llm_gateway import runtime


def test_no_database_startup_logs_and_keeps_inference_available(monkeypatch, caplog):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with TestClient(create_app()) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/v1/metrics/summary").status_code == 503
        assert client.post("/v1/inference", json={"model_id": "fake-small", "prompt": "hello"}).status_code == 200
    assert "telemetry_disabled" in caplog.text


@pytest.mark.parametrize("probe_fails", [False, True])
def test_database_resource_wired_and_disposed(monkeypatch, caplog, probe_fails):
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://user:secret@localhost/db")
    repository = AsyncMock()
    if probe_fails:
        repository.summary.side_effect = RuntimeError("secret connection details")
    database = type("DatabaseStub", (), {"repository": repository, "close": AsyncMock()})()
    monkeypatch.setattr(runtime, "Database", lambda settings: database)
    application = create_app()
    with TestClient(application) as client:
        assert application.state.inference_service.telemetry is repository
        assert client.post("/v1/inference", json={"model_id": "fake-small", "prompt": "hello"}).status_code == 200
        repository.record.assert_awaited_once()
    database.close.assert_awaited_once()
    assert "secret" not in caplog.text
    if probe_fails:
        assert "telemetry_startup_probe_failed" in caplog.text


def test_bad_database_configuration_fails_startup_safely(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "invalid-secret")
    with pytest.raises(ValueError) as caught:
        with TestClient(create_app()):
            pass
    assert "invalid-secret" not in str(caught.value)
