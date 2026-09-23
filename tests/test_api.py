import asyncio
from decimal import Decimal
from uuid import UUID

import httpx
import pytest
from fastapi.testclient import TestClient

from adaptive_llm_gateway.api.app import app, create_app
from adaptive_llm_gateway.api.dependencies import get_service
from adaptive_llm_gateway.application.service import InferenceService
from adaptive_llm_gateway.bootstrap import create_development_service
from adaptive_llm_gateway.models import InferenceRequest, InferenceResponse, ModelConfig
from adaptive_llm_gateway.providers import FakeProvider, LLMProvider
from adaptive_llm_gateway.providers.resolver import ProviderResolver
from adaptive_llm_gateway.registry import ModelRegistry


@pytest.fixture
def service():
    return create_development_service()


@pytest.fixture
def client(service):
    with TestClient(create_app(service)) as test_client:
        yield test_client


def payload(**overrides):
    return {"model_id": "fake-small", "prompt": "Hello adaptive gateway",
            "max_output_tokens": 2} | overrides


def assert_error(response, status, code):
    assert response.status_code == status
    data = response.json()
    assert data["error"]["code"] == code
    assert data["request_id"] == response.headers["x-request-id"]
    assert set(data) == {"error", "request_id"}


def test_importable_app_and_health(client):
    assert app.title == "Adaptive LLM Gateway"
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_discovery_only_available_models(client, service):
    source = service.registry.get("fake-small").model_dump()
    for model_id, enabled, provider in (("disabled", False, "fake"), ("unsupported", True, "missing")):
        service.registry.register(ModelConfig(**(source | {
            "model_id": model_id, "enabled": enabled, "provider": provider,
        })))
    response = client.get("/v1/models")
    assert response.status_code == 200
    assert response.json() == {"models": [
        {"model_id": "fake-small", "provider": "fake", "context_window": 4096},
        {"model_id": "fake-large", "provider": "fake", "context_window": 16384},
    ]}


@pytest.mark.parametrize("model_id,cost", [("fake-small", "0.00000195"), ("fake-large", "0.000011")])
def test_inference_metadata_and_explicit_selection(client, model_id, cost):
    response = client.post("/v1/inference", json=payload(model_id=model_id, system_prompt="be concise"))
    assert response.status_code == 200
    data = response.json()
    assert data == {
        "text": "Hello adaptive", "model_id": model_id, "provider": "fake",
        "input_tokens": 5, "output_tokens": 2, "latency_ms": 0.0,
        "estimated_cost_usd": cost, "request_id": response.headers["x-request-id"],
    }
    assert Decimal(data["estimated_cost_usd"]) == Decimal(cost)


def test_generated_ids_are_unique(client):
    ids = [client.post("/v1/inference", json=payload()).json()["request_id"] for _ in range(2)]
    assert ids[0] != ids[1]
    assert all(UUID(request_id).version == 4 for request_id in ids)


def test_caller_id_on_success_and_error(client):
    for model_id, status in (("fake-small", 200), ("unknown", 404)):
        response = client.post("/v1/inference", json=payload(model_id=model_id),
                               headers={"X-Request-ID": "demo-123.test_id"})
        assert response.status_code == status
        assert response.headers["x-request-id"] == "demo-123.test_id"
        assert response.json()["request_id"] == "demo-123.test_id"


@pytest.mark.parametrize("request_id", ["", "has spaces", "x" * 129, "-starts-with-dash"])
def test_invalid_caller_id(client, request_id):
    response = client.post("/v1/inference", json=payload(), headers={"X-Request-ID": request_id})
    assert_error(response, 400, "invalid_request_id")
    UUID(response.json()["request_id"])


def test_duplicate_request_ids_rejected(client):
    response = client.post("/v1/inference", json=payload(),
                           headers=[("X-Request-ID", "first"), ("X-Request-ID", "second")])
    assert_error(response, 400, "invalid_request_id")


def test_unknown_model(client):
    assert_error(client.post("/v1/inference", json=payload(model_id="unknown")), 404, "model_not_found")


def test_disabled_model(client, service):
    source = service.registry.get("fake-small").model_dump()
    service.registry.register(ModelConfig(**(source | {"model_id": "off", "enabled": False})))
    assert_error(client.post("/v1/inference", json=payload(model_id="off")), 403, "model_disabled")


@pytest.mark.parametrize("overrides", [
    {"prompt": " "}, {"system_prompt": ""}, {"max_output_tokens": 0},
    {"max_output_tokens": True}, {"temperature": 3}, {"model_id": ""},
    {"unknown": "field"},
])
def test_invalid_requests(client, overrides):
    assert_error(client.post("/v1/inference", json=payload(**overrides)), 422, "invalid_request")


@pytest.mark.parametrize("body", ['{"prompt":', '{}', 'null', '[]'])
def test_malformed_or_missing_request_fields(client, body):
    response = client.post("/v1/inference", content=body, headers={"Content-Type": "application/json"})
    assert_error(response, 422, "invalid_request")


def test_validation_does_not_echo_input(client):
    response = client.post("/v1/inference", json=payload(prompt="private prompt", temperature=7))
    assert "private prompt" not in response.text
    assert_error(response, 422, "invalid_request")


def test_context_limit_and_explicit_larger_model(client):
    request = payload(max_output_tokens=4094)
    assert_error(client.post("/v1/inference", json=request), 422, "context_limit_exceeded")
    response = client.post("/v1/inference", json=request | {"model_id": "fake-large"})
    assert response.status_code == 200
    assert response.json()["model_id"] == "fake-large"


def test_unavailable_provider(client, service):
    source = service.registry.get("fake-small").model_dump()
    service.registry.register(ModelConfig(**(source | {"model_id": "unavailable", "provider": "missing"})))
    assert_error(client.post("/v1/inference", json=payload(model_id="unavailable")), 503, "provider_unavailable")


@pytest.mark.parametrize("failure", [RuntimeError, ValueError, TimeoutError])
def test_provider_failure_is_sanitized(model, failure):
    class BrokenProvider(LLMProvider):
        async def generate(self, request: InferenceRequest) -> InferenceResponse:
            raise failure("secret API key and internal endpoint")

    registry = ModelRegistry()
    registry.register(model)
    resolver = ProviderResolver()
    resolver.register("fake", lambda config: BrokenProvider())
    with TestClient(create_app(InferenceService(registry, resolver))) as client:
        response = client.post("/v1/inference", json=payload())
    assert_error(response, 502, "provider_failure")
    assert "secret" not in response.text
    assert "Traceback" not in response.text


def test_factory_failure_is_sanitized(model):
    def broken_factory(config):
        raise RuntimeError("internal construction details")

    registry = ModelRegistry()
    registry.register(model)
    resolver = ProviderResolver()
    resolver.register("fake", broken_factory)
    with TestClient(create_app(InferenceService(registry, resolver))) as client:
        response = client.post("/v1/inference", json=payload())
    assert_error(response, 502, "provider_failure")
    assert "internal construction" not in response.text


def test_dependency_override_and_app_isolation(service):
    first, second = create_app(), create_app()
    empty = InferenceService(ModelRegistry(), ProviderResolver())
    first.dependency_overrides[get_service] = lambda: empty
    with TestClient(first) as client:
        assert client.get("/v1/models").json() == {"models": []}
    with TestClient(second) as client:
        assert len(client.get("/v1/models").json()["models"]) == 2


def test_openapi_documents_response_schemas(client):
    schema = client.get("/openapi.json").json()
    responses = schema["paths"]["/v1/inference"]["post"]["responses"]
    assert {"200", "400", "403", "404", "422", "502", "503"} <= responses.keys()
    assert responses["422"]["content"]["application/json"]["schema"]["$ref"].endswith("ErrorResponse")


def test_unexpected_application_error_is_sanitized():
    def broken_dependency():
        raise RuntimeError("private internal details")

    application = create_app()
    application.dependency_overrides[get_service] = broken_dependency
    with TestClient(application, raise_server_exceptions=False) as client:
        response = client.post("/v1/inference", json=payload(), headers={"X-Request-ID": "failure-id"})
    assert_error(response, 500, "internal_error")
    assert response.json()["request_id"] == "failure-id"
    assert "private" not in response.text


def test_http_error_preserves_allow_header(client):
    response = client.get("/v1/inference")
    assert_error(response, 405, "http_error")
    assert response.headers["allow"] == "POST"


@pytest.mark.asyncio
async def test_concurrent_async_inference_keeps_ids_isolated(model):
    arrived = 0
    both_arrived = asyncio.Event()

    class YieldingProvider(FakeProvider):
        async def generate(self, request: InferenceRequest) -> InferenceResponse:
            nonlocal arrived
            arrived += 1
            if arrived == 2:
                both_arrived.set()
            await asyncio.wait_for(both_arrived.wait(), timeout=2)
            return await super().generate(request)

    registry = ModelRegistry()
    registry.register(model)
    resolver = ProviderResolver()
    resolver.register("fake", YieldingProvider)
    app = create_app(InferenceService(registry, resolver))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        responses = await asyncio.gather(*[
            client.post("/v1/inference", json=payload(prompt=f"prompt {i}"),
                        headers={"X-Request-ID": f"request-{i}"}) for i in range(2)
        ])
    for i, response in enumerate(responses):
        assert response.status_code == 200
        assert response.json()["request_id"] == f"request-{i}"
        assert response.json()["text"] == f"prompt {i}"
