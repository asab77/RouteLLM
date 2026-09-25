"""Versioned trusted-artifact loading and production quality prediction."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import platform
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

import sklearn
from pydantic import ValidationError
from sklearn.pipeline import Pipeline

from adaptive_llm_gateway.errors import (
    CorruptPredictorArtifactError,
    IncompatibleArtifactFormatError,
    IncompatibleCategoryTaxonomyError,
    IncompatibleFeatureSchemaError,
    IncompatiblePredictorFormulationError,
    MissingRoutingCategoryError,
    PredictorArtifactChecksumError,
    PredictorArtifactNotFoundError,
    PredictorInputCompatibilityError,
    UnsupportedPredictorCandidateError,
)
from adaptive_llm_gateway.models import ModelConfig
from adaptive_llm_gateway.models.schemas import DomainModel, Identifier

from .features import ROUTING_CATEGORY_TAXONOMY_VERSION, RoutingCategory, RoutingRequestFeatures
from .policy import ModelAcceptabilityPrediction
from .quality_features import (
    CANONICAL_QUALITY_FEATURE_SCHEMA_VERSION,
    PREDICTOR_FORMULATION_ID,
    PREDICTOR_FORMULATION_VERSION,
    QUALITY_PREPROCESSING_ID,
    canonical_feature_matrix,
    canonical_from_production,
)

ARTIFACT_FORMAT_VERSION = "1.0.0"
ARTIFACT_METADATA_FILENAME = "metadata.json"
ARTIFACT_MODEL_FILENAME = "predictor.pkl"
FOUNDATION_V3_SHA256 = "64eda8e7388233f645906412a2524982ebfb9c848c42ee24a468ba04c31d16e6"
FOUNDATION_V3_PROTOCOL_SHA256 = "973c46e2dfd7c5739f623ff1c7f2378dc3e77c80e5815e23c975816700d4b861"
PHASE_8C0_FORMULATION_EVIDENCE = "PHASE_8C_0_PROVIDER_PIN_ABLATION"
FOUNDATION_V3_CANDIDATE_IDS = (
    "candidate-claude-sonnet-5",
    "candidate-gemini-3-flash",
    "candidate-gpt-6-luna",
    "candidate-nemotron-3.5-lightning",
)


class QualityPredictorArtifactMetadata(DomainModel):
    artifact_format_version: str
    predictor_formulation_id: str
    predictor_formulation_version: str
    canonical_feature_schema_version: str
    category_taxonomy_version: str
    training_dataset_name: str
    training_dataset_version: str
    training_dataset_sha256: str
    foundation_v3_protocol_sha256: str
    formulation_evidence: str
    valid_training_rows: int
    missing_label_rows: int
    known_candidate_ids: tuple[Identifier, ...]
    known_categories: tuple[RoutingCategory, ...]
    sklearn_version: str
    python_version: str
    preprocessing_id: str
    build_entrypoint: str
    model_filename: str
    model_sha256: str


def expected_metadata(
    *,
    model_sha256: str,
    known_candidate_ids: tuple[str, ...],
    known_categories: tuple[RoutingCategory, ...],
    valid_training_rows: int,
    missing_label_rows: int,
) -> QualityPredictorArtifactMetadata:
    return QualityPredictorArtifactMetadata(
        artifact_format_version=ARTIFACT_FORMAT_VERSION,
        predictor_formulation_id=PREDICTOR_FORMULATION_ID,
        predictor_formulation_version=PREDICTOR_FORMULATION_VERSION,
        canonical_feature_schema_version=CANONICAL_QUALITY_FEATURE_SCHEMA_VERSION,
        category_taxonomy_version=ROUTING_CATEGORY_TAXONOMY_VERSION,
        training_dataset_name="routellm-foundation-v3",
        training_dataset_version="3.0.0",
        training_dataset_sha256=FOUNDATION_V3_SHA256,
        foundation_v3_protocol_sha256=FOUNDATION_V3_PROTOCOL_SHA256,
        formulation_evidence=PHASE_8C0_FORMULATION_EVIDENCE,
        valid_training_rows=valid_training_rows,
        missing_label_rows=missing_label_rows,
        known_candidate_ids=known_candidate_ids,
        known_categories=known_categories,
        sklearn_version=sklearn.__version__,
        python_version=platform.python_version(),
        preprocessing_id=QUALITY_PREPROCESSING_ID,
        build_entrypoint="python -m adaptive_llm_gateway.routing.train_predictor",
        model_filename=ARTIFACT_MODEL_FILENAME,
        model_sha256=model_sha256,
    )


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_trusted_quality_artifact(
    output_directory: Path,
    pipeline: Pipeline,
    *,
    known_candidate_ids: tuple[str, ...],
    known_categories: tuple[RoutingCategory, ...],
    valid_training_rows: int,
    missing_label_rows: int,
) -> tuple[Path, Path, QualityPredictorArtifactMetadata]:
    """Serialize application-owned sklearn output and its validated metadata."""
    model_bytes = pickle.dumps(pipeline, protocol=5)
    checksum = hashlib.sha256(model_bytes).hexdigest()
    metadata = expected_metadata(
        model_sha256=checksum,
        known_candidate_ids=known_candidate_ids,
        known_categories=known_categories,
        valid_training_rows=valid_training_rows,
        missing_label_rows=missing_label_rows,
    )
    model_path = output_directory / ARTIFACT_MODEL_FILENAME
    metadata_path = output_directory / ARTIFACT_METADATA_FILENAME
    _atomic_write(model_path, model_bytes)
    _atomic_write(
        metadata_path,
        (json.dumps(metadata.model_dump(mode="json"), indent=2, sort_keys=True) + "\n").encode(),
    )
    return model_path, metadata_path, metadata


def _validate_metadata(metadata: QualityPredictorArtifactMetadata) -> None:
    if metadata.artifact_format_version != ARTIFACT_FORMAT_VERSION:
        raise IncompatibleArtifactFormatError("unsupported predictor artifact format")
    if (
        metadata.predictor_formulation_id != PREDICTOR_FORMULATION_ID
        or metadata.predictor_formulation_version != PREDICTOR_FORMULATION_VERSION
    ):
        raise IncompatiblePredictorFormulationError("predictor formulation mismatch")
    if metadata.canonical_feature_schema_version != CANONICAL_QUALITY_FEATURE_SCHEMA_VERSION:
        raise IncompatibleFeatureSchemaError("canonical feature schema mismatch")
    if metadata.category_taxonomy_version != ROUTING_CATEGORY_TAXONOMY_VERSION:
        raise IncompatibleCategoryTaxonomyError("category taxonomy mismatch")
    if (
        metadata.training_dataset_sha256 != FOUNDATION_V3_SHA256
        or metadata.foundation_v3_protocol_sha256 != FOUNDATION_V3_PROTOCOL_SHA256
        or metadata.formulation_evidence != PHASE_8C0_FORMULATION_EVIDENCE
        or metadata.preprocessing_id != QUALITY_PREPROCESSING_ID
    ):
        raise CorruptPredictorArtifactError("artifact governance metadata mismatch")
    if tuple(metadata.known_candidate_ids) != FOUNDATION_V3_CANDIDATE_IDS:
        raise CorruptPredictorArtifactError("artifact candidate compatibility set mismatch")
    if set(metadata.known_categories) != set(RoutingCategory):
        raise CorruptPredictorArtifactError("artifact category compatibility set mismatch")
    if (
        metadata.training_dataset_name != "routellm-foundation-v3"
        or metadata.training_dataset_version != "3.0.0"
        or metadata.valid_training_rows != 216
        or metadata.missing_label_rows != 8
        or metadata.model_filename != ARTIFACT_MODEL_FILENAME
        or len(metadata.model_sha256) != 64
        or any(character not in "0123456789abcdef" for character in metadata.model_sha256)
    ):
        raise CorruptPredictorArtifactError("artifact build metadata mismatch")
    if metadata.sklearn_version != sklearn.__version__:
        raise IncompatibleArtifactFormatError("artifact sklearn runtime mismatch")
    if metadata.python_version.split(".")[:2] != platform.python_version().split(".")[:2]:
        raise IncompatibleArtifactFormatError("artifact Python runtime mismatch")


def load_trusted_quality_artifact(
    artifact_directory: Path,
) -> tuple[Pipeline, QualityPredictorArtifactMetadata]:
    """Load trusted local build output after metadata and checksum validation.

    Pickle is intentionally restricted to application-owned artifacts. This function
    must never be connected to user uploads or automatic downloads.
    """
    metadata_path = artifact_directory / ARTIFACT_METADATA_FILENAME
    if not metadata_path.is_file():
        raise PredictorArtifactNotFoundError("predictor metadata file is missing")
    try:
        raw_metadata: Any = json.loads(metadata_path.read_text())
        metadata = QualityPredictorArtifactMetadata.model_validate(raw_metadata)
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError) as exc:
        raise CorruptPredictorArtifactError("predictor metadata is corrupt") from exc
    _validate_metadata(metadata)
    model_path = artifact_directory / metadata.model_filename
    if not model_path.is_file():
        raise PredictorArtifactNotFoundError("serialized predictor file is missing")
    try:
        model_bytes = model_path.read_bytes()
    except OSError as exc:
        raise CorruptPredictorArtifactError("serialized predictor cannot be read") from exc
    if hashlib.sha256(model_bytes).hexdigest() != metadata.model_sha256:
        raise PredictorArtifactChecksumError("serialized predictor checksum mismatch")
    try:
        pipeline = pickle.loads(model_bytes)
    except Exception as exc:
        raise CorruptPredictorArtifactError("serialized predictor cannot be decoded") from exc
    if not isinstance(pipeline, Pipeline) or not {"preprocess", "classifier"} <= set(
        pipeline.named_steps
    ):
        raise CorruptPredictorArtifactError("serialized object is not a quality pipeline")
    return pipeline, metadata


class SklearnQualityPredictor:
    """Production probability predictor; routing remains a separate Phase 8A concern."""

    def __init__(
        self,
        pipeline: Pipeline,
        metadata: QualityPredictorArtifactMetadata,
    ) -> None:
        _validate_metadata(metadata)
        self._pipeline = pipeline
        self.metadata = metadata
        self._known_candidates = frozenset(metadata.known_candidate_ids)
        self._known_categories = frozenset(metadata.known_categories)

    @classmethod
    def from_trusted_artifact(cls, artifact_directory: Path) -> "SklearnQualityPredictor":
        pipeline, metadata = load_trusted_quality_artifact(artifact_directory)
        return cls(pipeline, metadata)

    def predict(
        self,
        request_features: RoutingRequestFeatures,
        candidates: Sequence[ModelConfig],
    ) -> tuple[ModelAcceptabilityPrediction, ...]:
        if request_features.category is None:
            raise MissingRoutingCategoryError(
                "quality prediction requires an explicit routing category"
            )
        if request_features.category not in self._known_categories:
            raise PredictorInputCompatibilityError(
                "request category is not supported by this predictor artifact"
            )
        candidate_ids = [candidate.model_id for candidate in candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise PredictorInputCompatibilityError("candidate model IDs must be unique")
        unknown = sorted(set(candidate_ids) - self._known_candidates)
        if unknown:
            raise UnsupportedPredictorCandidateError(
                "candidate is not supported by this predictor artifact: " + ", ".join(unknown)
            )
        canonical = tuple(
            canonical_from_production(request_features, candidate)
            for candidate in candidates
        )
        if not canonical:
            return ()
        probabilities = self._pipeline.predict_proba(canonical_feature_matrix(canonical))[:, 1]
        return tuple(
            ModelAcceptabilityPrediction(
                model_id=candidate.model_id,
                predicted_acceptability=float(probability),
            )
            for candidate, probability in zip(candidates, probabilities)
        )


__all__ = [
    "ARTIFACT_FORMAT_VERSION", "ARTIFACT_METADATA_FILENAME", "ARTIFACT_MODEL_FILENAME",
    "FOUNDATION_V3_CANDIDATE_IDS", "FOUNDATION_V3_PROTOCOL_SHA256", "FOUNDATION_V3_SHA256",
    "PHASE_8C0_FORMULATION_EVIDENCE", "QualityPredictorArtifactMetadata",
    "SklearnQualityPredictor", "expected_metadata", "load_trusted_quality_artifact",
    "write_trusted_quality_artifact",
]
