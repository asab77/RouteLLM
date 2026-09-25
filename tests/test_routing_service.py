import ast
import hashlib
import inspect
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from adaptive_llm_gateway.errors import (
    CorruptPredictorArtifactError,
    DuplicateCandidatePredictionError,
    IncompatibleArtifactFormatError,
    InvalidQualityThresholdError,
    MissingRoutingCategoryError,
    NoEligibleCandidatesError,
    PredictorArtifactNotFoundError,
    PredictorInputCompatibilityError,
    UnsupportedPredictorCandidateError,
)
from adaptive_llm_gateway.models import InferenceRequest, ModelConfig
from adaptive_llm_gateway.pricing import calculate_projected_cost
from adaptive_llm_gateway.providers.gateway_config import CANDIDATE_MODELS
from adaptive_llm_gateway.routing.features import (
    CategoryProvenance,
    ProductionRequestFeatureExtractor,
    RoutingRequestFeatures,
)
from adaptive_llm_gateway.routing.ml_experiment import THRESHOLDS
from adaptive_llm_gateway.routing.policy import (
    CostAwareRoutingPolicy,
    ModelAcceptabilityPrediction,
    RoutingDecision,
    RoutingDecisionReason,
)
from adaptive_llm_gateway.routing.predictor import (
    ARTIFACT_METADATA_FILENAME,
    ARTIFACT_MODEL_FILENAME,
    SklearnQualityPredictor,
)
from adaptive_llm_gateway.routing.service import RoutingDecisionService
from adaptive_llm_gateway.routing.train_predictor import build_predictor_artifact
from adaptive_llm_gateway.routing.validate_production_path import (
    validate_production_routing_path,
)

RUN_ID = "61707aba-5ab2-4c16-8ec3-eed74555d69c"
ROOT = Path("benchmark-results")


def request() -> InferenceRequest:
    return InferenceRequest(
        prompt="Determine the answer and return it exactly.",
        system_prompt="Be concise.",
        max_output_tokens=64,
        temperature=0,
    )


def model(model_id: str, input_price="1", output_price="1") -> ModelConfig:
    return ModelConfig(
        model_id=model_id,
        provider="offline",
        provider_model_name=f"offline/{model_id}",
        input_cost_per_1m_tokens=input_price,
        output_cost_per_1m_tokens=output_price,
        context_window=8192,
    )


class FixedPredictor:
    def __init__(self, probabilities):
        self.probabilities = dict(probabilities)
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


class RecordingExtractor(ProductionRequestFeatureExtractor):
    def __init__(self):
        self.calls = []

    def extract(self, request, **kwargs):
        self.calls.append((request, kwargs))
        return super().extract(request, **kwargs)


class RecordingPolicy(CostAwareRoutingPolicy):
    def __init__(self):
        self.calls = []

    def route(self, candidates, quality_threshold):
        candidates = tuple(candidates)
        self.calls.append((candidates, quality_threshold))
        return super().route(candidates, quality_threshold)


@pytest.fixture(scope="module")
def real_artifact(tmp_path_factory):
    directory = tmp_path_factory.mktemp("phase-8d-artifact")
    build_predictor_artifact(ROOT, __import__("uuid").UUID(RUN_ID), directory)
    return directory


@pytest.fixture(scope="module")
def validation_report(real_artifact, tmp_path_factory):
    report_path = tmp_path_factory.mktemp("phase-8d-report") / "report.json"
    return validate_production_routing_path(real_artifact, report_path=report_path)


def test_orchestration_calls_extractor_predictor_once_and_policy_once():
    candidates = (model("a", output_price="1"), model("b", output_price="2"))
    extractor = RecordingExtractor()
    predictor = FixedPredictor({"a": 0.8, "b": 0.9})
    policy = RecordingPolicy()
    service = RoutingDecisionService(
        predictor, feature_extractor=extractor, policy=policy
    )
    decision = service.route(
        request(), candidates, 0.7, category_hint="qa",
        structured_output_required=True,
    )
    assert isinstance(decision, RoutingDecision)
    assert len(extractor.calls) == len(predictor.calls) == len(policy.calls) == 1
    received_features, received_candidates = predictor.calls[0]
    assert isinstance(received_features, RoutingRequestFeatures)
    assert received_features.requests_structured_output is True
    assert received_candidates == candidates
    assert {item.model_id for item in policy.calls[0][0]} == {"a", "b"}


def test_valid_category_and_provenance_reach_predictor_without_inference():
    predictor = FixedPredictor({"a": 0.9})
    service = RoutingDecisionService(predictor)
    features = ProductionRequestFeatureExtractor().extract(
        request(), category_hint="reasoning"
    ).model_copy(update={"category_provenance": CategoryProvenance.INFERRED})
    service.route_features(features, (model("a"),), 0.5)
    received = predictor.calls[0][0]
    assert received.category == "reasoning"
    assert received.category_provenance is CategoryProvenance.INFERRED


def test_missing_and_malformed_categories_fail_without_inference(real_artifact):
    predictor = SklearnQualityPredictor.from_trusted_artifact(real_artifact)
    service = RoutingDecisionService(predictor)
    with pytest.raises(MissingRoutingCategoryError):
        service.route(request(), CANDIDATE_MODELS, 0.5)
    with pytest.raises((ValueError, ValidationError)):
        service.route(request(), CANDIDATE_MODELS, 0.5, category_hint="not-a-category")


def test_real_artifact_supports_known_candidates_and_rejects_unknown(real_artifact):
    predictor = SklearnQualityPredictor.from_trusted_artifact(real_artifact)
    service = RoutingDecisionService(predictor)
    decision = service.route(
        request(), CANDIDATE_MODELS, 0.5, category_hint="qa"
    )
    assert decision.selected_model_id in {item.model_id for item in CANDIDATE_MODELS}
    unknown = model("new-registry-model")
    with pytest.raises(UnsupportedPredictorCandidateError):
        service.route(request(), (unknown,), 0.5, category_hint="qa")


def test_empty_candidate_set_uses_existing_phase_8a_error(real_artifact):
    service = RoutingDecisionService(
        SklearnQualityPredictor.from_trusted_artifact(real_artifact)
    )
    with pytest.raises(NoEligibleCandidatesError):
        service.route(request(), (), 0.5, category_hint="qa")


def test_optional_eligibility_boundary_is_applied_without_silent_predictor_filtering():
    class EnabledOnly:
        def eligible_candidates(self, request_features, candidates):
            return tuple(candidate for candidate in candidates if candidate.enabled)

    enabled = model("enabled")
    disabled = model("disabled").model_copy(update={"enabled": False})
    predictor = FixedPredictor({"enabled": 0.9})
    service = RoutingDecisionService(predictor, eligibility_filter=EnabledOnly())
    decision = service.route(
        request(), (disabled, enabled), 0.5, category_hint="qa"
    )
    assert decision.selected_model_id == "enabled"
    assert [item.model_id for item in predictor.calls[0][1]] == ["enabled"]


def test_candidate_set_is_generic_and_input_order_does_not_change_decision():
    candidates = (
        model("future-z", output_price="1"),
        model("future-a", output_price="1"),
    )
    service = RoutingDecisionService(
        FixedPredictor({"future-z": 0.9, "future-a": 0.9})
    )
    forward = service.route(request(), candidates, 0.5, category_hint="qa")
    reverse = service.route(
        request(), tuple(reversed(candidates)), 0.5, category_hint="qa"
    )
    assert forward == reverse
    assert forward.selected_model_id == "future-a"


def test_caller_threshold_is_required_and_forwarded_exactly():
    policy = RecordingPolicy()
    service = RoutingDecisionService(FixedPredictor({"a": 0.9}), policy=policy)
    decision = service.route(request(), (model("a"),), 0.73, category_hint="qa")
    assert decision.quality_threshold == 0.73
    assert policy.calls[0][1] == 0.73
    assert inspect.signature(service.route).parameters["quality_threshold"].default is inspect.Parameter.empty


@pytest.mark.parametrize("threshold", THRESHOLDS)
def test_frozen_evaluation_thresholds_are_supported_without_selecting_one(threshold):
    service = RoutingDecisionService(FixedPredictor({"a": 0.7}))
    decision = service.route(request(), (model("a"),), threshold, category_hint="qa")
    assert decision.quality_threshold == threshold


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), True, "0.5"])
def test_invalid_threshold_uses_existing_phase_8a_error(threshold):
    service = RoutingDecisionService(FixedPredictor({"a": 0.7}))
    with pytest.raises(InvalidQualityThresholdError):
        service.route(request(), (model("a"),), threshold, category_hint="qa")


def test_projected_cost_reuses_canonical_calculator():
    candidate = model("a", input_price="1.25", output_price="3.5")
    policy = RecordingPolicy()
    service = RoutingDecisionService(FixedPredictor({"a": 0.9}), policy=policy)
    service.route(request(), (candidate,), 0.5, category_hint="qa")
    priced = policy.calls[0][0][0]
    features = ProductionRequestFeatureExtractor().extract(
        request(), category_hint="qa"
    )
    assert priced.projected_cost_usd == calculate_projected_cost(
        approximate_input_tokens=features.approximate_input_tokens,
        effective_max_output_tokens=features.max_output_tokens,
        model=candidate,
    )


def test_cheapest_of_multiple_qualifying_candidates_is_selected():
    candidates = (model("cheap", output_price="1"), model("expensive", output_price="5"))
    service = RoutingDecisionService(FixedPredictor({"cheap": 0.8, "expensive": 0.99}))
    assert service.route(request(), candidates, 0.7, category_hint="qa").selected_model_id == "cheap"


def test_only_qualifying_candidate_is_selected():
    candidates = (model("low"), model("high", output_price="5"))
    service = RoutingDecisionService(FixedPredictor({"low": 0.7, "high": 0.9}))
    assert service.route(request(), candidates, 0.8, category_hint="qa").selected_model_id == "high"


def test_no_qualifier_falls_back_to_highest_probability():
    candidates = (model("best", output_price="5"), model("other", output_price="1"))
    service = RoutingDecisionService(FixedPredictor({"best": 0.7, "other": 0.6}))
    decision = service.route(request(), candidates, 0.8, category_hint="qa")
    assert decision.selected_model_id == "best"
    assert decision.reason is RoutingDecisionReason.NO_MODEL_MET_THRESHOLD_FALLBACK


def test_qualifying_cost_tie_uses_model_id():
    candidates = (model("z"), model("a"))
    service = RoutingDecisionService(FixedPredictor({"z": 0.9, "a": 0.9}))
    assert service.route(request(), candidates, 0.5, category_hint="qa").selected_model_id == "a"


def test_fallback_probability_tie_uses_cost_then_model_id():
    candidates = (
        model("expensive", output_price="5"),
        model("z-cheap", output_price="1"),
        model("a-cheap", output_price="1"),
    )
    service = RoutingDecisionService(FixedPredictor({
        "expensive": 0.7, "z-cheap": 0.7, "a-cheap": 0.7,
    }))
    assert service.route(request(), candidates, 0.8, category_hint="qa").selected_model_id == "a-cheap"


def test_duplicate_candidate_configuration_uses_existing_typed_error():
    duplicate = model("same")
    service = RoutingDecisionService(FixedPredictor({"same": 0.9}))
    with pytest.raises(DuplicateCandidatePredictionError):
        service.route(request(), (duplicate, duplicate), 0.5, category_hint="qa")


def test_duplicate_or_missing_predictor_outputs_fail_typed():
    class BadPredictor:
        def __init__(self, duplicate):
            self.duplicate = duplicate

        def predict(self, request_features, candidates):
            if self.duplicate:
                return (
                    ModelAcceptabilityPrediction(model_id="a", predicted_acceptability=0.8),
                    ModelAcceptabilityPrediction(model_id="a", predicted_acceptability=0.9),
                )
            return ()

    candidates = (model("a"), model("b"))
    with pytest.raises(DuplicateCandidatePredictionError):
        RoutingDecisionService(BadPredictor(True)).route(
            request(), candidates, 0.5, category_hint="qa"
        )
    with pytest.raises(PredictorInputCompatibilityError):
        RoutingDecisionService(BadPredictor(False)).route(
            request(), candidates, 0.5, category_hint="qa"
        )


def test_real_artifact_build_load_and_route_are_local(real_artifact):
    assert (real_artifact / ARTIFACT_METADATA_FILENAME).is_file()
    assert (real_artifact / ARTIFACT_MODEL_FILENAME).is_file()
    predictor = SklearnQualityPredictor.from_trusted_artifact(real_artifact)
    decision = RoutingDecisionService(predictor).route(
        request(), CANDIDATE_MODELS, 0.5, category_hint="qa"
    )
    assert decision.selected_model_id in {item.model_id for item in CANDIDATE_MODELS}


def test_full_reference_prediction_and_decision_parity(validation_report):
    assert validation_report["reference_requests"] == 56
    assert validation_report["candidate_predictions"] == 224
    assert validation_report["prediction_parity_count"] == 224
    assert validation_report["maximum_probability_difference"] == 0
    assert validation_report["routing_decisions"] == 336
    for name in (
        "decision_parity_count", "selected_model_parity_count",
        "projected_cost_parity_count", "routing_reason_parity_count",
        "qualifying_count_parity_count", "fallback_state_parity_count",
    ):
        assert validation_report[name] == 336


def test_reference_validation_is_deterministic_and_order_independent(validation_report):
    assert validation_report["repeated_run_deterministic"] is True
    assert validation_report["candidate_order_independent"] is True
    assert tuple(validation_report["threshold_grid"]) == THRESHOLDS
    assert all(
        sum(summary["selected_model_distribution"].values()) == 56
        for summary in validation_report["per_threshold"].values()
    )


def test_validation_report_generation_is_deterministic(real_artifact, tmp_path):
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first = validate_production_routing_path(real_artifact, report_path=first_path)
    second = validate_production_routing_path(real_artifact, report_path=second_path)
    assert first == second
    assert hashlib.sha256(first_path.read_bytes()).digest() == hashlib.sha256(
        second_path.read_bytes()
    ).digest()


def test_artifact_failure_modes_remain_typed(real_artifact, tmp_path):
    with pytest.raises(PredictorArtifactNotFoundError):
        SklearnQualityPredictor.from_trusted_artifact(tmp_path / "missing")
    corrupt = tmp_path / "corrupt"
    shutil.copytree(real_artifact, corrupt)
    (corrupt / ARTIFACT_METADATA_FILENAME).write_text("not-json")
    with pytest.raises(CorruptPredictorArtifactError):
        SklearnQualityPredictor.from_trusted_artifact(corrupt)
    incompatible = tmp_path / "incompatible"
    shutil.copytree(real_artifact, incompatible)
    metadata_path = incompatible / ARTIFACT_METADATA_FILENAME
    metadata = json.loads(metadata_path.read_text())
    metadata["artifact_format_version"] = "999"
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(IncompatibleArtifactFormatError):
        SklearnQualityPredictor.from_trusted_artifact(incompatible)


def test_service_has_no_provider_telemetry_http_foundation_or_phase9_dependencies():
    import adaptive_llm_gateway.routing.service as module

    source = inspect.getsource(module).lower()
    tree = ast.parse(source)
    imports = {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    forbidden = ("provider", "telemetry", "fastapi", "benchmark", "foundation", "response")
    assert not any(any(term in name.lower() for term in forbidden) for name in imports)
    assert "generate(" not in source
    assert "persist" not in source
    assert "escalat" not in source


def test_public_inference_contract_and_import_isolation_are_unchanged():
    from adaptive_llm_gateway.api.schemas import InferencePayload

    assert InferencePayload.model_fields["model_id"].is_required()
    assert "category" not in InferencePayload.model_fields
    code = (
        "import sys; import adaptive_llm_gateway.api.app; "
        "assert 'sklearn' not in sys.modules; "
        "assert 'adaptive_llm_gateway.routing.predictor' not in sys.modules; "
        "assert 'adaptive_llm_gateway.routing.service' not in sys.modules"
    )
    completed = subprocess.run([sys.executable, "-c", code], check=False)
    assert completed.returncode == 0
