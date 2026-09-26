import ast
import asyncio
import inspect
import json
import shutil
import subprocess
import sys
from decimal import Decimal
from pathlib import Path
from uuid import UUID

import pytest
from pydantic import ValidationError

from adaptive_llm_gateway.application.adaptive import (
    AdaptiveInferenceResult,
    AdaptiveInferenceService,
)
from adaptive_llm_gateway.application.service import InferenceService
from adaptive_llm_gateway.errors import (
    CorruptPredictorArtifactError,
    IncompatibleArtifactFormatError,
    InvalidQualityThresholdError,
    MissingRoutingCategoryError,
    ModelDisabledError,
    NoEligibleCandidatesError,
    PredictorArtifactNotFoundError,
    ProviderFailureError,
    ProviderUnavailableError,
    UnsupportedPredictorCandidateError,
)
from adaptive_llm_gateway.models import InferenceRequest, InferenceResponse, ModelConfig
from adaptive_llm_gateway.providers.base import LLMProvider
from adaptive_llm_gateway.providers.gateway_config import CANDIDATE_MODELS
from adaptive_llm_gateway.providers.resolver import ProviderResolver
from adaptive_llm_gateway.registry import ModelRegistry
from adaptive_llm_gateway.routing.features import ProductionRequestFeatureExtractor
from adaptive_llm_gateway.routing.policy import (
    CostAwareRoutingPolicy,
    ModelAcceptabilityPrediction,
    RoutingDecision,
    RoutingDecisionReason,
)
from adaptive_llm_gateway.routing.predictor import ARTIFACT_METADATA_FILENAME
from adaptive_llm_gateway.routing.service import RoutingDecisionService
from adaptive_llm_gateway.routing.train_predictor import build_predictor_artifact

ROOT = Path("benchmark-results")
RUN_ID = UUID("61707aba-5ab2-4c16-8ec3-eed74555d69c")


def model(
    model_id: str,
    *,
    input_price: str = "1",
    output_price: str = "1",
    enabled: bool = True,
    provider: str = "recording",
) -> ModelConfig:
    return ModelConfig(
        model_id=model_id,
        provider=provider,
        provider_model_name=f"fixture/{model_id}",
        input_cost_per_1m_tokens=input_price,
        output_cost_per_1m_tokens=output_price,
        context_window=10_000,
        enabled=enabled,
    )


class RecordingProvider(LLMProvider):
    def __init__(self, configured_model, calls, *, failing_ids=frozenset()):
        self.model = configured_model
        self.calls = calls
        self.failing_ids = failing_ids

    async def generate(self, request):
        self.calls.append(self.model.model_id)
        if self.model.model_id in self.failing_ids:
            raise RuntimeError("controlled provider failure")
        return InferenceResponse(
            text=f"executed:{self.model.model_id}",
            model_id=self.model.model_id,
            provider=self.model.provider,
            input_tokens=2,
            output_tokens=1,
            latency_ms=1.25,
            estimated_cost_usd=Decimal("0.000003"),
        )


def inference_service(models, calls, *, failing_ids=frozenset(), telemetry=None):
    registry = ModelRegistry()
    for configured_model in models:
        registry.register(configured_model)
    resolver = ProviderResolver()
    providers = {configured_model.provider for configured_model in models}
    for provider in providers:
        resolver.register(
            provider,
            lambda configured_model, calls=calls, failing_ids=failing_ids: RecordingProvider(
                configured_model, calls, failing_ids=failing_ids
            ),
        )
    return InferenceService(registry, resolver, telemetry=telemetry)


class FixedPredictor:
    def __init__(self, probabilities):
        self.probabilities = probabilities
        self.calls = []

    def predict(self, request_features, candidates):
        self.calls.append((request_features, tuple(candidates)))
        return tuple(
            ModelAcceptabilityPrediction(
                model_id=candidate.model_id,
                predicted_acceptability=self.probabilities[candidate.model_id],
            )
            for candidate in candidates
        )


class RoutingSpy:
    def __init__(self, decision=None, error=None):
        self.decision = decision
        self.error = error
        self.calls = []

    def route(self, request, candidates, quality_threshold, **kwargs):
        self.calls.append((request, tuple(candidates), quality_threshold, kwargs))
        if self.error is not None:
            raise self.error
        return self.decision


def decision(
    selected="cheap",
    *,
    projected="0.25",
    fallback=False,
    threshold=0.8,
):
    return RoutingDecision(
        selected_model_id=selected,
        selected_predicted_acceptability=0.91 if not fallback else 0.49,
        selected_projected_cost_usd=Decimal(projected),
        quality_threshold=threshold,
        threshold_satisfied=not fallback,
        fallback_used=fallback,
        eligible_candidate_count=2,
        qualifying_candidate_count=0 if fallback else 1,
        reason=(
            RoutingDecisionReason.NO_MODEL_MET_THRESHOLD_FALLBACK
            if fallback
            else RoutingDecisionReason.QUALITY_THRESHOLD_MET
        ),
    )


def request():
    return InferenceRequest(prompt="small controlled request", max_output_tokens=8)


@pytest.fixture(scope="module")
def built_artifact(tmp_path_factory):
    directory = tmp_path_factory.mktemp("adaptive-quality-artifact")
    build_predictor_artifact(ROOT, RUN_ID, directory)
    return directory


@pytest.mark.asyncio
async def test_adaptive_service_reuses_explicit_execution_and_preserves_metadata():
    calls = []
    models = (model("cheap"), model("strong"))
    routing = RoutingSpy(decision())
    service = AdaptiveInferenceService(inference_service(models, calls), routing)
    result = await service.generate(
        request(), category="qa", quality_threshold=0.8,
        candidate_model_ids=("cheap", "strong"), request_id="adaptive-1",
    )
    assert isinstance(result, AdaptiveInferenceResult)
    assert result.response.model_id == "cheap"
    assert result.routing_decision == routing.decision
    assert calls == ["cheap"]
    assert len(routing.calls) == 1


@pytest.mark.asyncio
async def test_real_policy_cheapest_qualifier_executes_exactly_once():
    calls = []
    models = (
        model("cheap", output_price="1"),
        model("strong", output_price="10"),
    )
    predictor = FixedPredictor({"cheap": 0.85, "strong": 0.95})
    service = AdaptiveInferenceService(
        inference_service(models, calls), RoutingDecisionService(predictor)
    )
    result = await service.generate(
        request(), category="qa", quality_threshold=0.8,
        candidate_model_ids=("strong", "cheap"),
    )
    assert result.routing_decision.selected_model_id == "cheap"
    assert calls == ["cheap"]
    assert len(predictor.calls) == 1
    assert [item.model_id for item in predictor.calls[0][1]] == ["strong", "cheap"]


@pytest.mark.asyncio
async def test_real_policy_fallback_executes_selected_model_once():
    calls = []
    models = (model("cheap"), model("strong"))
    predictor = FixedPredictor({"cheap": 0.4, "strong": 0.6})
    service = AdaptiveInferenceService(
        inference_service(models, calls), RoutingDecisionService(predictor)
    )
    result = await service.generate(
        request(), category="reasoning", quality_threshold=0.9,
        candidate_model_ids=("cheap", "strong"),
    )
    assert result.routing_decision.fallback_used is True
    assert result.routing_decision.selected_model_id == "strong"
    assert calls == ["strong"]


@pytest.mark.asyncio
async def test_routing_failure_prevents_provider_execution():
    calls = []
    routing = RoutingSpy(error=InvalidQualityThresholdError("invalid"))
    service = AdaptiveInferenceService(
        inference_service((model("cheap"),), calls), routing
    )
    with pytest.raises(InvalidQualityThresholdError):
        await service.generate(
            request(), category="qa", quality_threshold=2,
            candidate_model_ids=("cheap",),
        )
    assert calls == []


@pytest.mark.asyncio
async def test_provider_failure_preserves_existing_error_and_never_escalates():
    calls = []
    models = (model("cheap"), model("strong"))
    service = AdaptiveInferenceService(
        inference_service(models, calls, failing_ids={"cheap"}),
        RoutingSpy(decision()),
    )
    with pytest.raises(ProviderFailureError):
        await service.generate(
            request(), category="qa", quality_threshold=0.8,
            candidate_model_ids=("cheap", "strong"),
        )
    assert calls == ["cheap"]


@pytest.mark.asyncio
async def test_explicit_inference_never_invokes_adaptive_router():
    calls = []
    explicit = inference_service((model("cheap"),), calls)
    routing = RoutingSpy(decision())
    AdaptiveInferenceService(explicit, routing)
    response = await explicit.generate("cheap", request())
    assert response.model_id == "cheap"
    assert calls == ["cheap"]
    assert routing.calls == []


@pytest.mark.asyncio
async def test_category_validation_has_no_default_or_inference(built_artifact):
    calls = []
    models = tuple(
        item.model_copy(update={"provider": "recording"})
        for item in CANDIDATE_MODELS
    )
    service = AdaptiveInferenceService.from_trusted_artifact(
        inference_service(models, calls), built_artifact
    )
    with pytest.raises(MissingRoutingCategoryError):
        await service.generate(
            request(), category=None, quality_threshold=0.5,
            candidate_model_ids=tuple(item.model_id for item in models),
        )
    with pytest.raises((ValueError, ValidationError)):
        await service.generate(
            request(), category="other", quality_threshold=0.5,
            candidate_model_ids=tuple(item.model_id for item in models),
        )
    assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), "0.5"])
async def test_invalid_threshold_prevents_provider_execution(threshold):
    calls = []
    predictor = FixedPredictor({"cheap": 0.8})
    service = AdaptiveInferenceService(
        inference_service((model("cheap"),), calls),
        RoutingDecisionService(predictor),
    )
    with pytest.raises(InvalidQualityThresholdError):
        await service.generate(
            request(), category="qa", quality_threshold=threshold,
            candidate_model_ids=("cheap",),
        )
    assert calls == []


@pytest.mark.asyncio
async def test_empty_candidate_set_fails_before_provider_execution():
    calls = []
    service = AdaptiveInferenceService(
        inference_service((model("cheap"),), calls),
        RoutingDecisionService(FixedPredictor({})),
    )
    with pytest.raises(NoEligibleCandidatesError):
        await service.generate(
            request(), category="qa", quality_threshold=0.5,
            candidate_model_ids=(),
        )
    assert calls == []


@pytest.mark.asyncio
async def test_registry_models_are_canonical_and_candidates_are_not_hardcoded():
    calls = []
    models = (model("future-a"), model("future-b"))
    predictor = FixedPredictor({"future-a": 0.9, "future-b": 0.7})
    service = AdaptiveInferenceService(
        inference_service(models, calls), RoutingDecisionService(predictor)
    )
    await service.generate(
        request(), category="classification", quality_threshold=0.8,
        candidate_model_ids=("future-b", "future-a"),
    )
    assert [item.model_id for item in predictor.calls[0][1]] == ["future-b", "future-a"]
    assert predictor.calls[0][1][0] is service._inference_service.registry.get("future-b")
    assert calls == ["future-a"]


@pytest.mark.asyncio
@pytest.mark.parametrize(("configured_model", "error"), [
    (model("disabled", enabled=False), ModelDisabledError),
    (model("unavailable", provider="missing"), ProviderUnavailableError),
])
async def test_ineligible_registry_candidate_fails_before_routing_or_provider(
    configured_model, error
):
    calls = []
    registry = ModelRegistry()
    registry.register(configured_model)
    routing = RoutingSpy(decision(selected=configured_model.model_id))
    service = AdaptiveInferenceService(
        InferenceService(registry, ProviderResolver()), routing
    )
    with pytest.raises(error):
        await service.generate(
            request(), category="qa", quality_threshold=0.5,
            candidate_model_ids=(configured_model.model_id,),
        )
    assert routing.calls == []
    assert calls == []


@pytest.mark.asyncio
async def test_unsupported_predictor_candidate_is_not_silently_dropped(built_artifact):
    calls = []
    unknown = model("new-compatible-provider-model")
    service = AdaptiveInferenceService.from_trusted_artifact(
        inference_service((unknown,), calls), built_artifact
    )
    with pytest.raises(UnsupportedPredictorCandidateError):
        await service.generate(
            request(), category="qa", quality_threshold=0.5,
            candidate_model_ids=(unknown.model_id,),
        )
    assert calls == []


@pytest.mark.asyncio
async def test_configured_trusted_artifact_loads_and_routes(built_artifact):
    calls = []
    models = tuple(
        item.model_copy(update={"provider": "recording"})
        for item in CANDIDATE_MODELS
    )
    service = AdaptiveInferenceService.from_trusted_artifact(
        inference_service(models, calls), built_artifact
    )
    result = await service.generate(
        request(), category="qa", quality_threshold=0.5,
        candidate_model_ids=tuple(item.model_id for item in models),
    )
    assert calls == [result.routing_decision.selected_model_id]


@pytest.mark.parametrize(("kind", "error"), [
    ("missing", PredictorArtifactNotFoundError),
    ("corrupt", CorruptPredictorArtifactError),
    ("incompatible", IncompatibleArtifactFormatError),
])
def test_artifact_configuration_failures_occur_before_provider_execution(
    built_artifact, tmp_path, kind, error
):
    calls = []
    target = tmp_path / kind
    if kind != "missing":
        shutil.copytree(built_artifact, target)
        metadata = target / ARTIFACT_METADATA_FILENAME
        if kind == "corrupt":
            metadata.write_text("not-json")
        else:
            values = json.loads(metadata.read_text())
            values["artifact_format_version"] = "999"
            metadata.write_text(json.dumps(values))
    with pytest.raises(error):
        AdaptiveInferenceService.from_trusted_artifact(
            inference_service((model("cheap"),), calls), target
        )
    assert calls == []


def test_artifact_factory_has_no_build_download_or_machine_path():
    source = inspect.getsource(AdaptiveInferenceService.from_trusted_artifact)
    assert "build_predictor_artifact" not in source
    assert "http" not in source.lower()
    assert "/Users/" not in source


@pytest.mark.asyncio
async def test_projected_and_realized_costs_remain_distinct():
    calls = []
    routing = RoutingSpy(decision(projected="0.25"))
    service = AdaptiveInferenceService(
        inference_service((model("cheap"), model("strong")), calls), routing
    )
    result = await service.generate(
        request(), category="qa", quality_threshold=0.8,
        candidate_model_ids=("cheap", "strong"),
    )
    assert result.routing_decision.selected_projected_cost_usd == Decimal("0.25")
    assert result.response.estimated_cost_usd == Decimal("0.000003")


@pytest.mark.asyncio
async def test_existing_telemetry_records_executed_model_without_routing_fields():
    class Repository:
        def __init__(self):
            self.events = []

        async def record(self, event):
            self.events.append(event)

    repository = Repository()
    calls = []
    explicit = inference_service(
        (model("cheap"), model("strong")), calls, telemetry=repository
    )
    service = AdaptiveInferenceService(explicit, RoutingSpy(decision()))
    await service.generate(
        request(), category="qa", quality_threshold=0.8,
        candidate_model_ids=("cheap", "strong"), request_id="adaptive-telemetry",
    )
    assert len(repository.events) == 1
    event = repository.events[0]
    assert event.model_id == "cheap"
    assert not (
        {"quality_threshold", "routing_reason", "fallback_used"}
        & set(event.__dataclass_fields__)
    )


@pytest.mark.asyncio
async def test_concurrent_requests_have_no_shared_request_state():
    calls = []
    models = (model("cheap"), model("strong"))
    predictor = FixedPredictor({"cheap": 0.9, "strong": 0.8})
    service = AdaptiveInferenceService(
        inference_service(models, calls), RoutingDecisionService(predictor)
    )
    results = await asyncio.gather(*(
        service.generate(
            InferenceRequest(prompt=f"request {index}"),
            category="qa", quality_threshold=0.85,
            candidate_model_ids=("cheap", "strong"),
        )
        for index in range(8)
    ))
    assert len(results) == 8
    assert calls == ["cheap"] * 8
    assert len(predictor.calls) == 8


def test_adaptive_layer_contains_no_routing_provider_or_telemetry_reimplementation():
    import adaptive_llm_gateway.application.adaptive as module

    source = inspect.getsource(module)
    tree = ast.parse(source)
    call_names = {
        node.func.attr for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "predict" not in call_names
    assert "calculate_projected_cost" not in source
    assert "resolve" not in call_names
    assert "_record" not in call_names
    assert "escalat" not in source.lower()
    assert "response_validator" not in source.lower()
    assert "foundation" not in source.lower()


def test_api_import_remains_lazy_and_explicit_contract_unchanged():
    from adaptive_llm_gateway.api.schemas import InferencePayload

    assert InferencePayload.model_fields["model_id"].is_required()
    assert "category" not in InferencePayload.model_fields
    assert "quality_threshold" not in InferencePayload.model_fields
    code = (
        "import sys; import adaptive_llm_gateway.api.app; "
        "assert 'sklearn' not in sys.modules; "
        "assert 'adaptive_llm_gateway.application.adaptive' not in sys.modules; "
        "assert 'adaptive_llm_gateway.routing.predictor' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_adaptive_generate_requires_category_threshold_and_candidates():
    signature = inspect.signature(AdaptiveInferenceService.generate)
    for name in ("category", "quality_threshold", "candidate_model_ids"):
        assert signature.parameters[name].default is inspect.Parameter.empty
