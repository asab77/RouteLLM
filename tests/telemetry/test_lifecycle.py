import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from adaptive_llm_gateway.api.app import create_app
from adaptive_llm_gateway.errors import ContextLimitError, ModelDisabledError, ProviderFailureError, ProviderUnavailableError
from adaptive_llm_gateway.models import InferenceRequest, ModelConfig
from adaptive_llm_gateway.providers import FakeProvider
from adaptive_llm_gateway.providers.resolver import ProviderResolver
from adaptive_llm_gateway.telemetry.query import TelemetryQueryService


@pytest.mark.asyncio
async def test_success_metadata_without_raw_content(telemetry_service, repository):
    before = datetime.now(timezone.utc)
    result = await telemetry_service.generate("fake-small", InferenceRequest(
        prompt="private prompt text", system_prompt="private system", max_output_tokens=2,
    ), request_id="trace-123")
    event, = repository.events
    assert event.request_id == "trace-123"
    assert event.model_id == "fake-small" and event.provider == "fake"
    assert event.success and event.error_category is None
    assert event.input_tokens == 5 and event.output_tokens == 2
    assert event.estimated_cost_usd == result.estimated_cost_usd == Decimal("0.00000195")
    assert event.latency_ms == result.latency_ms
    assert before <= event.created_at <= datetime.now(timezone.utc)
    assert event.created_at.tzinfo is not None
    assert event.prompt_characters == 19 and event.system_prompt_characters == 14
    assert event.max_output_tokens == 2 and event.temperature == 1.0
    assert "private" not in str(asdict(event))
    assert not {"prompt", "system_prompt", "text", "response"} & asdict(event).keys()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,expected,exception", [
    ("context", "context_limit_exceeded", ContextLimitError),
    ("disabled", "model_disabled", ModelDisabledError),
    ("unavailable", "provider_unavailable", ProviderUnavailableError),
    ("broken", "provider_failure", ProviderFailureError),
])
async def test_failures_record_normalized_categories(telemetry_service, repository, kind, expected, exception):
    request = InferenceRequest(prompt="secret prompt", max_output_tokens=5000 if kind == "context" else 2)
    if kind == "disabled":
        config = telemetry_service.registry.get("fake-small").model_dump() | {"model_id": "disabled", "enabled": False}
        telemetry_service.registry.register(ModelConfig(**config))
    if kind in ("unavailable", "broken"):
        telemetry_service.resolver = ProviderResolver()
    if kind == "broken":
        class Broken(FakeProvider):
            async def generate(self, request):
                raise RuntimeError("secret provider credential")
        telemetry_service.resolver.register("fake", Broken)
    with pytest.raises(exception):
        await telemetry_service.generate("disabled" if kind == "disabled" else "fake-small", request, request_id="failure")
    event, = repository.events
    assert not event.success and event.error_category == expected
    assert event.request_id == "failure"
    assert event.input_tokens is event.output_tokens is event.estimated_cost_usd is None
    assert event.latency_ms >= 0
    assert "secret" not in str(asdict(event))


@pytest.mark.parametrize("failure", [False, True])
def test_write_failure_preserves_http_outcome(telemetry_service, caplog, failure):
    class BrokenRepository:
        async def record(self, event):
            raise RuntimeError("secret credentials and prompt")

    telemetry_service.telemetry = BrokenRepository()
    with TestClient(create_app(telemetry_service)) as client:
        response = client.post("/v1/inference", json={
            "model_id": "fake-small", "prompt": "private prompt",
            "max_output_tokens": 5000 if failure else 2,
        }, headers={"X-Request-ID": "write-failure"})
    assert response.status_code == (422 if failure else 200)
    if failure:
        assert response.json()["error"]["code"] == "context_limit_exceeded"
    else:
        assert response.json()["text"] == "private prompt"
    assert "telemetry_write_failed" in caplog.text
    assert "write-failure" in caplog.text
    assert "secret" not in caplog.text and "private prompt" not in caplog.text


@pytest.mark.asyncio
async def test_write_timeout_cancels_repository_operation(telemetry_service, caplog):
    cleaned = asyncio.Event()

    class SlowRepository:
        async def record(self, event):
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()

    telemetry_service.telemetry = SlowRepository()
    telemetry_service.telemetry_timeout = 0.01
    result = await asyncio.wait_for(telemetry_service.generate("fake-small", InferenceRequest(prompt="hello")), timeout=1)
    assert result.text == "hello"
    assert cleaned.is_set()
    assert "telemetry_write_failed" in caplog.text


def test_excluded_failures_and_repeated_correlation_ids(telemetry_service, repository):
    with TestClient(create_app(telemetry_service)) as client:
        for body in ({}, {"model_id": "missing", "prompt": "hello"}):
            assert client.post("/v1/inference", json=body).status_code in (404, 422)
        assert repository.events == []
        for _ in range(2):
            response = client.post("/v1/inference", json={"model_id": "fake-small", "prompt": "hello"},
                                   headers={"X-Request-ID": "repeated"})
            assert response.status_code == 200
    assert len(repository.events) == 2
    assert repository.events[0].id != repository.events[1].id
    assert all(event.request_id == "repeated" for event in repository.events)


def test_generated_http_id_is_persisted(telemetry_service, repository):
    with TestClient(create_app(telemetry_service)) as client:
        result = client.post("/v1/inference", json={"model_id": "fake-small", "prompt": "hello"})
    assert repository.events[0].request_id == result.json()["request_id"] == result.headers["x-request-id"]


@pytest.mark.asyncio
async def test_cancellation_not_mislabeled_as_failure(telemetry_service, repository):
    class Cancelled(FakeProvider):
        async def generate(self, request):
            raise asyncio.CancelledError()

    telemetry_service.resolver = ProviderResolver()
    telemetry_service.resolver.register("fake", Cancelled)
    with pytest.raises(asyncio.CancelledError):
        await telemetry_service.generate("fake-small", InferenceRequest(prompt="hello"))
    assert repository.events == []


@pytest.mark.asyncio
async def test_query_aggregates(telemetry_service, repository):
    for model_id in ("fake-small", "fake-large"):
        await telemetry_service.generate(model_id, InferenceRequest(prompt="hello", max_output_tokens=1))
    with pytest.raises(ContextLimitError):
        await telemetry_service.generate("fake-small", InferenceRequest(prompt="hello", max_output_tokens=5000))
    summary = await TelemetryQueryService(repository).summary()
    assert summary.total_requests == 3
    assert summary.successful_requests == 2 and summary.failed_requests == 1
    assert summary.total_estimated_cost_usd == Decimal("0.00000475")
    assert summary.total_input_tokens == summary.total_output_tokens == 2
    assert summary.average_latency_ms == sum(event.latency_ms for event in repository.events) / 3


def test_metrics_endpoint_empty_and_populated(telemetry_service):
    with TestClient(create_app(telemetry_service)) as client:
        empty = client.get("/v1/metrics/summary")
        assert empty.status_code == 200
        assert empty.json() == {"total_requests": 0, "successful_requests": 0, "failed_requests": 0,
            "total_estimated_cost_usd": "0", "average_latency_ms": None,
            "total_input_tokens": 0, "total_output_tokens": 0}
        client.post("/v1/inference", json={"model_id": "fake-small", "prompt": "hello"})
        data = client.get("/v1/metrics/summary").json()
        assert data["total_requests"] == data["successful_requests"] == 1
        assert isinstance(data["total_estimated_cost_usd"], str)
        assert Decimal(data["total_estimated_cost_usd"]) == Decimal("0.00000075")


@pytest.mark.parametrize("configured", [True, False])
def test_metrics_unavailable_is_not_empty_success(telemetry_service, configured, caplog):
    class Broken:
        async def summary(self):
            raise RuntimeError("secret database connection")

    telemetry_service.telemetry = Broken() if configured else None
    with TestClient(create_app(telemetry_service)) as client:
        response = client.get("/v1/metrics/summary")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "telemetry_unavailable"
    assert "secret" not in caplog.text + response.text
