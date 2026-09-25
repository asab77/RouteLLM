import ast
import hashlib
import inspect
import json
import pickle
import shutil
from pathlib import Path
from uuid import UUID

import numpy as np
import pytest

from adaptive_llm_gateway.errors import (
    CorruptPredictorArtifactError,
    IncompatibleArtifactFormatError,
    IncompatibleCategoryTaxonomyError,
    IncompatibleFeatureSchemaError,
    IncompatiblePredictorFormulationError,
    MissingRoutingCategoryError,
    PredictorArtifactChecksumError,
    PredictorArtifactNotFoundError,
    UnsupportedPredictorCandidateError,
)
from adaptive_llm_gateway.models import ModelConfig, ReasoningEffort
from adaptive_llm_gateway.providers.gateway_config import CANDIDATE_MODELS
from adaptive_llm_gateway.registry import ModelRegistry
from adaptive_llm_gateway.routing.features import (
    CategoryProvenance,
    ProductionRequestFeatureExtractor,
    RoutingCategory,
    RoutingRequestFeatures,
)
from adaptive_llm_gateway.routing.ml_diagnostics import (
    CATEGORY_CANDIDATE_INTERACTION,
    REPRESENTATIONS,
)
from adaptive_llm_gateway.routing.ml_features import RANDOM_STATE, load_ml_dataset
from adaptive_llm_gateway.routing.policy import QualityPredictor
from adaptive_llm_gateway.routing.predictor import (
    ARTIFACT_FORMAT_VERSION,
    ARTIFACT_METADATA_FILENAME,
    ARTIFACT_MODEL_FILENAME,
    FOUNDATION_V3_PROTOCOL_SHA256,
    FOUNDATION_V3_SHA256,
    SklearnQualityPredictor,
    load_trusted_quality_artifact,
)
from adaptive_llm_gateway.routing.provider_pin_ablation import NO_PIN_REPRESENTATION
from adaptive_llm_gateway.routing.quality_features import (
    CANONICAL_PREDICTIVE_FEATURES,
    CANONICAL_QUALITY_FEATURE_SCHEMA_VERSION,
    CANONICAL_TRAINING_SERVING_AUDIT,
    FORBIDDEN_CANONICAL_FEATURES,
    INTERACTION_FEATURE,
    PREDICTOR_FORMULATION_ID,
    QUALITY_RANDOM_STATE,
    CanonicalCompatibilityStatus,
    canonical_feature_matrix,
    canonical_from_production,
    canonical_from_training_row,
    canonicalize_category,
    canonicalize_reasoning_effort,
    resolve_effective_output_allowance,
    validate_canonical_skew_audit,
)
from adaptive_llm_gateway.routing.train_predictor import (
    build_predictor_artifact,
    canonical_training_serving_pairs,
)

RUN_ID = UUID("61707aba-5ab2-4c16-8ec3-eed74555d69c")
ROOT = Path("benchmark-results")


@pytest.fixture(scope="module")
def dataset():
    return load_ml_dataset(ROOT, RUN_ID)


@pytest.fixture(scope="module")
def built_artifact(tmp_path_factory):
    directory = tmp_path_factory.mktemp("quality-artifact")
    result = build_predictor_artifact(ROOT, RUN_ID, directory)
    return directory, result


def _request_for_row(row, *, category=None, provenance=CategoryProvenance.CLIENT_HINT):
    values = row.features
    return RoutingRequestFeatures(
        category=category or canonicalize_category(row.category),
        category_provenance=provenance,
        prompt_characters=int(values["prompt_characters"]),
        system_prompt_characters=0,
        approximate_input_tokens=int(values["approximate_input_tokens"]),
        contains_code=bool(values["contains_code"]),
        requests_structured_output=bool(values["requests_structured_output"]),
        max_output_tokens=int(values["requested_max_output_tokens"]),
        constraint_indicator_count=int(values["constraint_indicator_count"]),
        reasoning_indicator_count=int(values["reasoning_indicator_count"]),
    )


def _copy_artifact(built_artifact, tmp_path):
    source, _ = built_artifact
    target = tmp_path / "artifact"
    shutil.copytree(source, target)
    return target


def _rewrite_metadata(directory, **changes):
    path = directory / ARTIFACT_METADATA_FILENAME
    values = json.loads(path.read_text())
    values.update(changes)
    path.write_text(json.dumps(values))


def test_corrected_formulation_is_exactly_the_approved_one_variable_ablation():
    assert CANONICAL_PREDICTIVE_FEATURES == NO_PIN_REPRESENTATION.features
    assert "upstream_provider_pin" not in CANONICAL_PREDICTIVE_FEATURES
    assert "candidate_id" in CANONICAL_PREDICTIVE_FEATURES
    assert "category" in CANONICAL_PREDICTIVE_FEATURES
    assert INTERACTION_FEATURE in CANONICAL_PREDICTIVE_FEATURES


def test_historical_phase_7_representation_remains_unchanged():
    historical = REPRESENTATIONS[CATEGORY_CANDIDATE_INTERACTION]
    assert "upstream_provider_pin" in historical.features
    assert len(historical.features) == len(CANONICAL_PREDICTIVE_FEATURES) + 1


@pytest.mark.parametrize(("value", "expected"), [
    ("classification", RoutingCategory.CLASSIFICATION),
    ("coding", RoutingCategory.CODING),
    ("extraction", RoutingCategory.EXTRACTION),
    ("json", RoutingCategory.STRUCTURED_JSON),
    ("structured_json", RoutingCategory.STRUCTURED_JSON),
    ("qa", RoutingCategory.QA),
    ("reasoning", RoutingCategory.REASONING),
    ("summarization", RoutingCategory.SUMMARIZATION),
])
def test_category_canonicalization(value, expected):
    assert canonicalize_category(value) is expected


def test_missing_category_is_typed_and_never_defaulted(dataset):
    row = dataset.rows[0]
    values = _request_for_row(row).model_dump()
    values.update(category=None, category_provenance=CategoryProvenance.ABSENT)
    absent = RoutingRequestFeatures(**values)
    with pytest.raises(MissingRoutingCategoryError):
        canonical_from_production(absent, CANDIDATE_MODELS[0])


def test_category_provenance_is_not_predictive(dataset):
    row = dataset.rows[0]
    candidate = next(item for item in CANDIDATE_MODELS if item.model_id == row.candidate_id)
    client = canonical_from_production(_request_for_row(row), candidate)
    inferred = canonical_from_production(
        _request_for_row(row, provenance=CategoryProvenance.INFERRED), candidate
    )
    assert client == inferred
    assert "category_provenance" not in CANONICAL_PREDICTIVE_FEATURES


def test_structured_output_is_explicit_and_json_words_do_not_activate_it():
    from adaptive_llm_gateway.models import InferenceRequest

    request = InferenceRequest(prompt="Return a JSON object matching this schema")
    extractor = ProductionRequestFeatureExtractor()
    assert extractor.extract(request).requests_structured_output is False
    assert extractor.extract(
        request, structured_output_required=True
    ).requests_structured_output is True


@pytest.mark.parametrize(("value", "expected"), [
    (None, ReasoningEffort.NONE),
    ("none", ReasoningEffort.NONE),
    ("low", ReasoningEffort.LOW),
    (ReasoningEffort.MEDIUM, ReasoningEffort.MEDIUM),
])
def test_reasoning_effort_canonicalization(value, expected):
    assert canonicalize_reasoning_effort(value) is expected


def test_reasoning_and_allowance_resolvers_have_no_model_identity_special_cases():
    import adaptive_llm_gateway.routing.quality_features as module

    source = inspect.getsource(module)
    assert "provider_model_name" not in source
    assert "candidate-gemini" not in source
    assert "candidate-claude" not in source
    assert "candidate-gpt" not in source
    assert "candidate-nemotron" not in source


def test_effective_output_allowance_uses_typed_category_configuration(dataset):
    reasoning_row = next(
        row for row in dataset.rows
        if row.category == "reasoning" and row.candidate_id == "candidate-gemini-3-flash"
    )
    gemini = next(
        item for item in CANDIDATE_MODELS
        if item.model_id == "candidate-gemini-3-flash"
    )
    request = _request_for_row(reasoning_row)
    assert request.max_output_tokens == 160
    assert resolve_effective_output_allowance(request, gemini) == 256
    for candidate in CANDIDATE_MODELS:
        if candidate.model_id != gemini.model_id:
            assert resolve_effective_output_allowance(request, candidate) == 160


def test_schema_governance_and_skew_audit_are_complete():
    assert CANONICAL_QUALITY_FEATURE_SCHEMA_VERSION == "1.0.0"
    assert not set(CANONICAL_PREDICTIVE_FEATURES) & FORBIDDEN_CANONICAL_FEATURES
    assert len(CANONICAL_PREDICTIVE_FEATURES) == len(set(CANONICAL_PREDICTIVE_FEATURES)) == 16
    assert {item.feature_name for item in CANONICAL_TRAINING_SERVING_AUDIT} == set(
        CANONICAL_PREDICTIVE_FEATURES
    )
    assert validate_canonical_skew_audit() == {
        "EXACT_MATCH": 11,
        "CANONICALIZED_MATCH": 5,
        "UNRESOLVED": 0,
    }
    assert all(
        item.status is not CanonicalCompatibilityStatus.UNRESOLVED
        for item in CANONICAL_TRAINING_SERVING_AUDIT
    )


def test_all_224_training_and_serving_canonical_rows_match(dataset):
    pairs = canonical_training_serving_pairs(dataset)
    assert len(pairs) == 224
    assert all(training == production for training, production in pairs)
    assert {
        training.category for training, _ in pairs
    } == set(RoutingCategory)
    assert {
        training.candidate_id for training, _ in pairs
    } == {candidate.model_id for candidate in CANDIDATE_MODELS}


def test_shared_vectorization_is_exact_for_training_and_serving(dataset):
    pairs = canonical_training_serving_pairs(dataset)
    training = canonical_feature_matrix(item[0] for item in pairs)
    production = canonical_feature_matrix(item[1] for item in pairs)
    assert np.array_equal(training, production)
    interaction_index = CANONICAL_PREDICTIVE_FEATURES.index(INTERACTION_FEATURE)
    assert all("::" in value for value in training[:, interaction_index])


def test_fitted_preprocessing_vectors_match_strictly(dataset, built_artifact):
    _, result = built_artifact
    pairs = canonical_training_serving_pairs(dataset)
    preprocess = result["pipeline"].named_steps["preprocess"]
    training = preprocess.transform(canonical_feature_matrix(item[0] for item in pairs))
    production = preprocess.transform(canonical_feature_matrix(item[1] for item in pairs))
    assert np.allclose(training, production, rtol=0, atol=1e-15)


def test_final_fit_uses_only_valid_boolean_labels_and_preserves_formulation(dataset, built_artifact):
    _, result = built_artifact
    pipeline, metadata = load_trusted_quality_artifact(result["model_path"].parent)
    assert metadata.valid_training_rows == 216
    assert metadata.missing_label_rows == 8
    assert sum(row.label_status == "valid" for row in dataset.rows) == 216
    assert all(type(row.acceptable) is bool for row in dataset.rows if row.label_status == "valid")
    assert all(row.acceptable is None for row in dataset.rows if row.label_status == "missing")
    classifier = pipeline.named_steps["classifier"]
    assert classifier.C == 1.0
    assert classifier.solver == "lbfgs"
    assert classifier.max_iter == 1000
    assert classifier.class_weight is None
    assert classifier.random_state == QUALITY_RANDOM_STATE == RANDOM_STATE
    assert "GradientBoost" not in type(classifier).__name__
    assert "Forest" not in type(classifier).__name__
    assert "Neural" not in type(classifier).__name__


def test_artifact_metadata_is_complete_and_frozen(built_artifact):
    _, result = built_artifact
    metadata = result["metadata"]
    assert metadata.artifact_format_version == ARTIFACT_FORMAT_VERSION
    assert metadata.predictor_formulation_id == PREDICTOR_FORMULATION_ID
    assert metadata.canonical_feature_schema_version == CANONICAL_QUALITY_FEATURE_SCHEMA_VERSION
    assert metadata.training_dataset_sha256 == FOUNDATION_V3_SHA256
    assert metadata.foundation_v3_protocol_sha256 == FOUNDATION_V3_PROTOCOL_SHA256
    assert metadata.valid_training_rows == 216 and metadata.missing_label_rows == 8
    assert set(metadata.known_candidate_ids) == {item.model_id for item in CANDIDATE_MODELS}
    assert set(metadata.known_categories) == set(RoutingCategory)
    assert metadata.build_entrypoint.endswith("routing.train_predictor")
    assert len(metadata.model_sha256) == 64


def test_valid_artifact_loads_and_conforms_to_quality_predictor(built_artifact):
    directory, _ = built_artifact
    predictor = SklearnQualityPredictor.from_trusted_artifact(directory)
    assert isinstance(predictor, QualityPredictor)


def test_missing_artifact_is_typed(tmp_path):
    with pytest.raises(PredictorArtifactNotFoundError):
        load_trusted_quality_artifact(tmp_path)


def test_corrupt_metadata_is_typed(built_artifact, tmp_path):
    directory = _copy_artifact(built_artifact, tmp_path)
    (directory / ARTIFACT_METADATA_FILENAME).write_text("not-json")
    with pytest.raises(CorruptPredictorArtifactError):
        load_trusted_quality_artifact(directory)


@pytest.mark.parametrize(("change", "error"), [
    ({"artifact_format_version": "999"}, IncompatibleArtifactFormatError),
    ({"predictor_formulation_id": "other"}, IncompatiblePredictorFormulationError),
    ({"canonical_feature_schema_version": "999"}, IncompatibleFeatureSchemaError),
    ({"category_taxonomy_version": "999"}, IncompatibleCategoryTaxonomyError),
])
def test_incompatible_artifact_metadata_is_typed(built_artifact, tmp_path, change, error):
    directory = _copy_artifact(built_artifact, tmp_path)
    _rewrite_metadata(directory, **change)
    with pytest.raises(error):
        load_trusted_quality_artifact(directory)


def test_checksum_mismatch_is_typed(built_artifact, tmp_path):
    directory = _copy_artifact(built_artifact, tmp_path)
    path = directory / ARTIFACT_MODEL_FILENAME
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(PredictorArtifactChecksumError):
        load_trusted_quality_artifact(directory)


def test_checksum_valid_but_corrupt_pickle_is_typed(built_artifact, tmp_path):
    directory = _copy_artifact(built_artifact, tmp_path)
    data = b"not-a-pickle"
    (directory / ARTIFACT_MODEL_FILENAME).write_bytes(data)
    _rewrite_metadata(directory, model_sha256=hashlib.sha256(data).hexdigest())
    with pytest.raises(CorruptPredictorArtifactError):
        load_trusted_quality_artifact(directory)


def test_unknown_candidate_is_typed_and_registry_extensibility_is_independent(
    built_artifact, dataset
):
    directory, _ = built_artifact
    predictor = SklearnQualityPredictor.from_trusted_artifact(directory)
    unknown = CANDIDATE_MODELS[0].model_copy(update={"model_id": "new-registry-model"})
    registry = ModelRegistry()
    registry.register(unknown)
    assert registry.get("new-registry-model") == unknown
    with pytest.raises(UnsupportedPredictorCandidateError, match="new-registry-model"):
        predictor.predict(_request_for_row(dataset.rows[0]), (unknown,))


def test_direct_loaded_probability_parity_and_determinism(built_artifact, dataset):
    directory, result = built_artifact
    pipeline = result["pipeline"]
    predictor = SklearnQualityPredictor.from_trusted_artifact(directory)
    row = dataset.rows[0]
    candidate = next(item for item in CANDIDATE_MODELS if item.model_id == row.candidate_id)
    request = _request_for_row(row)
    canonical = canonical_from_production(request, candidate)
    direct = float(pipeline.predict_proba(canonical_feature_matrix((canonical,)))[0, 1])
    first = predictor.predict(request, (candidate,))[0]
    second = predictor.predict(request, (candidate,))[0]
    assert first == second
    assert first.predicted_acceptability == pytest.approx(direct, abs=1e-15)
    assert 0 <= first.predicted_acceptability <= 1


def test_batch_candidate_association_and_order_are_preserved(built_artifact, dataset):
    directory, _ = built_artifact
    predictor = SklearnQualityPredictor.from_trusted_artifact(directory)
    request = _request_for_row(dataset.rows[0])
    forward = predictor.predict(request, CANDIDATE_MODELS)
    reverse = predictor.predict(request, tuple(reversed(CANDIDATE_MODELS)))
    assert [item.model_id for item in forward] == [item.model_id for item in CANDIDATE_MODELS]
    assert [item.model_id for item in reverse] == [
        item.model_id for item in reversed(CANDIDATE_MODELS)
    ]
    assert {item.model_id: item.predicted_acceptability for item in forward} == {
        item.model_id: item.predicted_acceptability for item in reverse
    }


def test_predictor_has_no_threshold_routing_provider_or_foundation_dependency():
    import adaptive_llm_gateway.routing.predictor as module

    source = inspect.getsource(module)
    assert "quality_threshold" not in source
    assert "CostAwareRoutingPolicy" not in source
    assert "adaptive_llm_gateway.providers" not in source
    assert "benchmark-results" not in source
    assert "foundation-v3.json" not in source
    tree = ast.parse(source)
    imports = {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    assert not any("provider" in name for name in imports)


def test_artifact_is_application_owned_and_not_exposed_by_http_api():
    from adaptive_llm_gateway.api.schemas import InferencePayload
    from adaptive_llm_gateway.api.routes import inference

    assert "artifact" not in InferencePayload.model_fields
    assert "predictor" not in inspect.signature(inference).parameters


def test_generated_artifact_contains_no_raw_benchmark_content(built_artifact):
    directory, _ = built_artifact
    metadata_text = (directory / ARTIFACT_METADATA_FILENAME).read_text().lower()
    forbidden = ("prompt", "response", "reasoning text", "provider_payload", "api_key")
    assert not any(name in metadata_text for name in forbidden)
    pipeline = pickle.loads((directory / ARTIFACT_MODEL_FILENAME).read_bytes())
    assert set(pipeline.named_steps) == {"preprocess", "classifier"}
