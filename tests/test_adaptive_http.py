import asyncio
import inspect
import json
import shutil
import subprocess
import sys
from contextlib import asynccontextmanager
from decimal import Decimal
from pathlib import Path
from uuid import UUID

import httpx
import pytest
from fastapi.testclient import TestClient

from adaptive_llm_gateway.api.app import create_app
from adaptive_llm_gateway.application.adaptive import (
    AdaptiveInferenceResult as InternalAdaptiveResult,
    AdaptiveInferenceService,
)
from adaptive_llm_gateway.application.adaptive_config import (
    ADAPTIVE_ARTIFACT_PATH_ENV,
    ADAPTIVE_CANDIDATES_ENV,
    AdaptiveRoutingConfig,
    AdaptiveRuntime,
    build_adaptive_runtime,
)
from adaptive_llm_gateway.application.service import InferenceService
from adaptive_llm_gateway.errors import (
    CorruptPredictorArtifactError,
    IncompatibleArtifactFormatError,
    NoEligibleCandidatesError,
    PredictorArtifactNotFoundError,
    ProviderFailureError,
    UnsupportedPredictorCandidateError,
)
from adaptive_llm_gateway.models import InferenceRequest, InferenceResponse, ModelConfig
from adaptive_llm_gateway.providers.base import LLMProvider
from adaptive_llm_gateway.providers.gateway_config import CANDIDATE_MODELS
from adaptive_llm_gateway.providers.resolver import ProviderResolver
from adaptive_llm_gateway.registry import ModelNotFoundError, ModelRegistry
from adaptive_llm_gateway.routing.policy import (
    ModelAcceptabilityPrediction,
    RoutingDecision,
    RoutingDecisionReason,
)
from adaptive_llm_gateway.routing.predictor import (
    ARTIFACT_METADATA_FILENAME,
    SklearnQualityPredictor,
)
from adaptive_llm_gateway.routing.service import RoutingDecisionService
from adaptive_llm_gateway.routing.train_predictor import build_predictor_artifact

ROOT = Path("benchmark-results")
RUN_ID = UUID("61707aba-5ab2-4c16-8ec3-eed74555d69c")


def model(model_id, *, price="1", enabled=True, provider="fake"):
    return ModelConfig(
        model_id=model_id,
        provider=provider,
        provider_model_name=f"fixture/{model_id}",
        input_cost_per_1m_tokens=price,
        output_cost_per_1m_tokens=price,
        context_window=10_000,
        enabled=enabled,
    )


class RecordingProvider(LLMProvider):
    def __init__(self, configured_model, calls, failures=frozenset()):
        self.model = configured_model
        self.calls = calls
        self.failures = failures

    async def generate(self, request):
        self.calls.append((self.model.model_id, request.prompt))
        if self.model.model_id in self.failures:
            raise RuntimeError("private controlled provider failure")
        return InferenceResponse(
            text=f"served:{request.prompt}",
            model_id=self.model.model_id,
            provider=self.model.provider,
            input_tokens=2,
            output_tokens=1,
            latency_ms=1,
            estimated_cost_usd=Decimal("0.000004"),
        )


def explicit_service(models, calls, *, failures=frozenset()):
    registry = ModelRegistry()
    for configured_model in models:
        registry.register(configured_model)
    resolver = ProviderResolver()
    for provider in {item.provider for item in models}:
        if provider == "missing":
            continue
        resolver.register(
            provider,
            lambda configured_model, calls=calls, failures=failures: RecordingProvider(
                configured_model, calls, failures
            ),
        )
    return InferenceService(registry, resolver)


class FixedPredictor:
    def __init__(self, probabilities):
        self.probabilities = probabilities
        self.calls = 0

    def predict(self, request_features, candidates):
        self.calls += 1
        return tuple(
            ModelAcceptabilityPrediction(
                model_id=candidate.model_id,
                predicted_acceptability=self.probabilities[candidate.model_id],
            )
            for candidate in candidates
        )


def configured_app(probabilities, models, calls, *, failures=frozenset()):
    explicit = explicit_service(models, calls, failures=failures)
    adaptive = AdaptiveInferenceService(
        explicit, RoutingDecisionService(FixedPredictor(probabilities))
    )
    runtime = AdaptiveRuntime(adaptive, tuple(item.model_id for item in models))
    return create_app(explicit, runtime)


def adaptive_payload(**overrides):
    return {
        "prompt": "HTTP adaptive request",
        "max_output_tokens": 8,
        "category": "qa",
        "quality_threshold": 0.8,
    } | overrides


def assert_error(response, status, code):
    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    assert response.json()["request_id"] == response.headers["x-request-id"]


@pytest.fixture(scope="module")
def built_artifact(tmp_path_factory):
    directory = tmp_path_factory.mktemp("adaptive-http-artifact")
    build_predictor_artifact(ROOT, RUN_ID, directory)
    return directory


def test_config_absent_and_blank_pair_disable_adaptive(monkeypatch):
    assert AdaptiveRoutingConfig.from_environment().enabled is False
    monkeypatch.setenv(ADAPTIVE_ARTIFACT_PATH_ENV, "  ")
    monkeypatch.setenv(ADAPTIVE_CANDIDATES_ENV, "")
    assert AdaptiveRoutingConfig.from_environment() == AdaptiveRoutingConfig()


def test_valid_config_parses_candidate_whitespace_deterministically(monkeypatch):
    monkeypatch.setenv(ADAPTIVE_ARTIFACT_PATH_ENV, " artifacts/quality ")
    monkeypatch.setenv(ADAPTIVE_CANDIDATES_ENV, " alpha, beta ,gamma ")
    config = AdaptiveRoutingConfig.from_environment()
    assert config.enabled is True
    assert config.artifact_path == Path("artifacts/quality")
    assert config.candidate_model_ids == ("alpha", "beta", "gamma")


@pytest.mark.parametrize(("artifact", "candidates"), [
    ("artifact", ""),
    ("", "alpha"),
    ("artifact", "alpha,,beta"),
    ("artifact", "alpha,alpha"),
])
def test_partial_empty_or_duplicate_config_is_rejected(monkeypatch, artifact, candidates):
    monkeypatch.setenv(ADAPTIVE_ARTIFACT_PATH_ENV, artifact)
    monkeypatch.setenv(ADAPTIVE_CANDIDATES_ENV, candidates)
    with pytest.raises(ValueError):
        AdaptiveRoutingConfig.from_environment()


def test_runtime_config_contains_no_request_policy_or_machine_path():
    fields = set(AdaptiveRoutingConfig.model_fields)
    assert fields == {"artifact_path", "candidate_model_ids"}
    assert not ({"quality_threshold", "category", "prompt", "temperature"} & fields)
    source = inspect.getsource(sys.modules[AdaptiveRoutingConfig.__module__])
    assert "/Users/" not in source


def test_runtime_construction_rejects_unknown_candidate(built_artifact):
    calls = []
    config = AdaptiveRoutingConfig(
        artifact_path=built_artifact, candidate_model_ids=("unknown",)
    )
    with pytest.raises(ModelNotFoundError):
        build_adaptive_runtime(explicit_service((model("known"),), calls), config)
    assert calls == []


def test_runtime_construction_rejects_predictor_unsupported_candidate(built_artifact):
    calls = []
    unknown = model("registered-but-untrained")
    config = AdaptiveRoutingConfig(
        artifact_path=built_artifact,
        candidate_model_ids=(unknown.model_id,),
    )
    with pytest.raises(UnsupportedPredictorCandidateError):
        build_adaptive_runtime(explicit_service((unknown,), calls), config)
    assert calls == []


@pytest.mark.parametrize(("kind", "error"), [
    ("missing", PredictorArtifactNotFoundError),
    ("corrupt", CorruptPredictorArtifactError),
    ("incompatible", IncompatibleArtifactFormatError),
])
def test_artifact_startup_failures_are_typed_and_make_no_provider_calls(
    built_artifact, tmp_path, kind, error
):
    calls = []
    target = tmp_path / kind
    if kind != "missing":
        shutil.copytree(built_artifact, target)
        metadata_path = target / ARTIFACT_METADATA_FILENAME
        if kind == "corrupt":
            metadata_path.write_text("not-json")
        else:
            metadata = json.loads(metadata_path.read_text())
            metadata["artifact_format_version"] = "999"
            metadata_path.write_text(json.dumps(metadata))
    candidate = CANDIDATE_MODELS[0].model_copy(update={"provider": "fake"})
    config = AdaptiveRoutingConfig(
        artifact_path=target, candidate_model_ids=(candidate.model_id,)
    )
    with pytest.raises(error):
        build_adaptive_runtime(explicit_service((candidate,), calls), config)
    assert calls == []


def test_valid_artifact_loads_once_and_runtime_needs_no_foundation(
    built_artifact, monkeypatch
):
    calls = []
    candidates = tuple(
        item.model_copy(update={"provider": "fake"}) for item in CANDIDATE_MODELS[:2]
    )
    config = AdaptiveRoutingConfig(
        artifact_path=built_artifact,
        candidate_model_ids=tuple(item.model_id for item in candidates),
    )
    load_calls = 0
    original = SklearnQualityPredictor.from_trusted_artifact.__func__

    def counted(cls, directory):
        nonlocal load_calls
        load_calls += 1
        return original(cls, directory)

    monkeypatch.setattr(
        SklearnQualityPredictor,
        "from_trusted_artifact",
        classmethod(counted),
    )
    explicit = explicit_service(candidates, calls)
    runtime = build_adaptive_runtime(explicit, config)
    assert runtime is not None
    assert load_calls == 1

    def forbidden_foundation(*args, **kwargs):
        raise AssertionError("Foundation runtime access is forbidden")

    monkeypatch.setattr(
        "adaptive_llm_gateway.routing.ml_features.load_ml_dataset",
        forbidden_foundation,
    )
    with TestClient(create_app(explicit, runtime)) as client:
        first = client.post("/v1/inference/adaptive", json=adaptive_payload())
        second = client.post("/v1/inference/adaptive", json=adaptive_payload())
    assert first.status_code == second.status_code == 200
    assert load_calls == 1
    assert len(calls) == 2


def test_artifact_is_never_downloaded_or_built_at_startup():
    source = inspect.getsource(build_adaptive_runtime)
    assert "http" not in source.lower()
    assert "build_predictor_artifact" not in source
    assert "train" not in source.lower()


@pytest.mark.parametrize("category", [
    "classification", "coding", "extraction", "structured_json", "qa",
    "reasoning", "summarization",
])
def test_public_request_accepts_every_canonical_category(category):
    calls = []
    models = (model("cheap"),)
    with TestClient(configured_app({"cheap": 0.9}, models, calls)) as client:
        response = client.post(
            "/v1/inference/adaptive", json=adaptive_payload(category=category)
        )
    assert response.status_code == 200


@pytest.mark.parametrize("changes", [
    {"category": None},
    {"category": "other"},
    {"quality_threshold": None},
    {"quality_threshold": -0.1},
    {"quality_threshold": 1.1},
    {"quality_threshold": "0.8"},
    {"candidate_model_ids": ["cheap"]},
    {"artifact_path": "/private/artifact"},
    {"provider": "fake"},
])
def test_invalid_or_internal_public_inputs_return_422_and_call_no_provider(changes):
    calls = []
    models = (model("cheap"),)
    payload = adaptive_payload(**changes)
    if changes.get("category") is None:
        payload.pop("category")
    if changes.get("quality_threshold") is None:
        payload.pop("quality_threshold")
    with TestClient(configured_app({"cheap": 0.9}, models, calls)) as client:
        response = client.post("/v1/inference/adaptive", json=payload)
    assert_error(response, 422, "invalid_request")
    assert calls == []


def test_adaptive_http_selects_cheapest_qualifier_and_exposes_minimal_metadata():
    calls = []
    models = (model("strong", price="10"), model("cheap", price="1"))
    app = configured_app({"strong": 0.95, "cheap": 0.85}, models, calls)
    with TestClient(app) as client:
        response = client.post("/v1/inference/adaptive", json=adaptive_payload())
    assert response.status_code == 200
    data = response.json()
    assert data["model_id"] == "cheap"
    assert data["routing"] == {
        "selected_model_id": "cheap",
        "threshold_satisfied": True,
        "fallback_used": False,
        "reason": "quality_threshold_met",
    }
    assert calls == [("cheap", "HTTP adaptive request")]
    forbidden = {"candidate_probabilities", "projected_cost", "artifact_path", "features"}
    assert not (forbidden & set(data))
    assert not (forbidden & set(data["routing"]))


def test_adaptive_http_fallback_executes_exactly_once():
    calls = []
    models = (model("cheap"), model("strong"))
    with TestClient(
        configured_app({"cheap": 0.4, "strong": 0.6}, models, calls)
    ) as client:
        response = client.post(
            "/v1/inference/adaptive",
            json=adaptive_payload(quality_threshold=0.9),
        )
    assert response.status_code == 200
    assert response.json()["routing"]["fallback_used"] is True
    assert response.json()["model_id"] == "strong"
    assert calls == [("strong", "HTTP adaptive request")]


def test_routing_failure_is_sanitized_and_calls_no_provider():
    class BrokenAdaptive:
        async def generate(self, *args, **kwargs):
            raise NoEligibleCandidatesError("private candidate details")

    calls = []
    explicit = explicit_service((model("cheap"),), calls)
    runtime = AdaptiveRuntime(BrokenAdaptive(), ("cheap",))
    with TestClient(create_app(explicit, runtime)) as client:
        response = client.post("/v1/inference/adaptive", json=adaptive_payload())
    assert_error(response, 503, "adaptive_routing_unavailable")
    assert "private" not in response.text
    assert calls == []


def test_selected_provider_failure_uses_existing_semantics_without_retry():
    calls = []
    models = (model("cheap"), model("strong"))
    with TestClient(
        configured_app(
            {"cheap": 0.9, "strong": 0.8}, models, calls, failures={"cheap"}
        )
    ) as client:
        response = client.post("/v1/inference/adaptive", json=adaptive_payload())
    assert_error(response, 502, "provider_failure")
    assert calls == [("cheap", "HTTP adaptive request")]


def test_adaptive_endpoint_disabled_is_stable_503_and_explicit_still_works():
    calls = []
    explicit = explicit_service((model("cheap"),), calls)
    with TestClient(create_app(explicit)) as client:
        adaptive = client.post("/v1/inference/adaptive", json=adaptive_payload())
        explicit_response = client.post(
            "/v1/inference",
            json={"model_id": "cheap", "prompt": "explicit request"},
        )
    assert_error(adaptive, 503, "adaptive_routing_unavailable")
    assert explicit_response.status_code == 200
    assert explicit_response.json()["model_id"] == "cheap"
    assert calls == [("cheap", "explicit request")]


def test_gateway_owned_portfolio_changes_without_request_changes():
    payload = adaptive_payload()
    assert not ({"candidate_model_ids", "candidates"} & set(payload))
    first_calls, second_calls = [], []
    first_models = (model("first"),)
    second_models = (model("second"),)
    with TestClient(configured_app({"first": 0.9}, first_models, first_calls)) as client:
        first = client.post("/v1/inference/adaptive", json=payload)
    with TestClient(configured_app({"second": 0.9}, second_models, second_calls)) as client:
        second = client.post("/v1/inference/adaptive", json=payload)
    assert first.json()["model_id"] == "first"
    assert second.json()["model_id"] == "second"
    assert first_calls == [("first", "HTTP adaptive request")]
    assert second_calls == [("second", "HTTP adaptive request")]


def test_openapi_contract_is_minimal_and_explicit_endpoint_unchanged():
    calls = []
    with TestClient(
        configured_app({"cheap": 0.9}, (model("cheap"),), calls)
    ) as client:
        schema = client.get("/openapi.json").json()
    assert "/v1/inference/adaptive" in schema["paths"]
    adaptive_operation = schema["paths"]["/v1/inference/adaptive"]["post"]
    request_ref = adaptive_operation["requestBody"]["content"]["application/json"]["schema"]["$ref"]
    request_schema = schema["components"]["schemas"][request_ref.rsplit("/", 1)[-1]]
    assert {"category", "quality_threshold"} <= set(request_schema["required"])
    assert not ({"candidate_model_ids", "artifact_path", "provider"} & set(request_schema["properties"]))
    response_ref = adaptive_operation["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]
    response_schema = schema["components"]["schemas"][response_ref.rsplit("/", 1)[-1]]
    assert "routing" in response_schema["properties"]
    explicit_ref = schema["paths"]["/v1/inference"]["post"]["requestBody"]["content"]["application/json"]["schema"]["$ref"]
    explicit_schema = schema["components"]["schemas"][explicit_ref.rsplit("/", 1)[-1]]
    assert "model_id" in explicit_schema["required"]
    assert not ({"category", "quality_threshold"} & set(explicit_schema["properties"]))


@pytest.mark.asyncio
async def test_concurrent_adaptive_http_has_no_request_metadata_leakage():
    class RequestAwareRouting:
        def __init__(self):
            self.seen = []

        def route(self, request, candidates, threshold, *, category_hint, **kwargs):
            self.seen.append((request.prompt, str(category_hint), float(threshold)))
            selected = "code-model" if str(category_hint) == "coding" else "qa-model"
            fallback = float(threshold) > 0.9
            return RoutingDecision(
                selected_model_id=selected,
                selected_predicted_acceptability=0.8,
                selected_projected_cost_usd=Decimal("0.01"),
                quality_threshold=threshold,
                threshold_satisfied=not fallback,
                fallback_used=fallback,
                eligible_candidate_count=2,
                qualifying_candidate_count=0 if fallback else 1,
                reason=(RoutingDecisionReason.NO_MODEL_MET_THRESHOLD_FALLBACK
                        if fallback else RoutingDecisionReason.QUALITY_THRESHOLD_MET),
            )

    calls = []
    models = (model("code-model"), model("qa-model"))
    explicit = explicit_service(models, calls)
    routing = RequestAwareRouting()
    runtime = AdaptiveRuntime(
        AdaptiveInferenceService(explicit, routing),
        tuple(item.model_id for item in models),
    )
    app = create_app(explicit, runtime)
    requests = [
        adaptive_payload(prompt="code request", category="coding", quality_threshold=0.7),
        adaptive_payload(prompt="qa request", category="qa", quality_threshold=0.95),
    ]
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        responses = await asyncio.gather(*(
            client.post("/v1/inference/adaptive", json=payload) for payload in requests
        ))
    assert [response.status_code for response in responses] == [200, 200]
    assert responses[0].json()["model_id"] == "code-model"
    assert responses[0].json()["routing"]["fallback_used"] is False
    assert responses[1].json()["model_id"] == "qa-model"
    assert responses[1].json()["routing"]["fallback_used"] is True
    assert set(routing.seen) == {
        ("code request", "coding", 0.7),
        ("qa request", "qa", 0.95),
    }
    assert set(calls) == {
        ("code-model", "code request"),
        ("qa-model", "qa request"),
    }


def test_application_lifespan_reads_environment_configuration_once(monkeypatch):
    import adaptive_llm_gateway.api.app as app_module

    explicit = explicit_service((model("configured"),), [])
    runtime = object()
    seen = []

    @asynccontextmanager
    async def configured_service():
        yield explicit

    def configured_runtime(service, config):
        seen.append((service, config))
        return runtime

    monkeypatch.setenv(ADAPTIVE_ARTIFACT_PATH_ENV, "trusted/artifact")
    monkeypatch.setenv(ADAPTIVE_CANDIDATES_ENV, "configured")
    monkeypatch.setattr(app_module, "application_service", configured_service)
    monkeypatch.setattr(app_module, "build_adaptive_runtime", configured_runtime)
    application = create_app()
    with TestClient(application) as client:
        assert client.get("/health").status_code == 200
        assert application.state.adaptive_runtime is runtime
    assert len(seen) == 1
    assert seen[0][0] is explicit
    assert seen[0][1].candidate_model_ids == ("configured",)


def test_explicit_only_construction_does_not_import_sklearn_or_adaptive_service():
    code = """
import sys
from adaptive_llm_gateway.application.adaptive_config import AdaptiveRoutingConfig, build_adaptive_runtime
from adaptive_llm_gateway.application.service import InferenceService
from adaptive_llm_gateway.providers.resolver import ProviderResolver
from adaptive_llm_gateway.registry import ModelRegistry
runtime = build_adaptive_runtime(InferenceService(ModelRegistry(), ProviderResolver()), AdaptiveRoutingConfig())
assert runtime is None
assert 'sklearn' not in sys.modules
assert 'adaptive_llm_gateway.application.adaptive' not in sys.modules
assert 'adaptive_llm_gateway.routing.predictor' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_endpoint_is_thin_and_has_no_candidate_or_routing_implementation():
    from adaptive_llm_gateway.api.routes import adaptive_inference

    source = inspect.getsource(adaptive_inference)
    assert "predict" not in source
    assert "projected" not in source
    assert "provider" not in source
    assert "candidate-" not in source
    assert source.count("runtime.service.generate") == 1
