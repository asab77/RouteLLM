"""Reproducibly fit the approved Phase 8C quality predictor from frozen local data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from uuid import UUID

import numpy as np

from adaptive_llm_gateway.providers.gateway_config import CANDIDATE_MODELS

from .analysis import FOUNDATION_V3_RUN_ID
from .features import CategoryProvenance, RoutingRequestFeatures
from .ml_features import load_ml_dataset
from .predictor import (
    FOUNDATION_V3_PROTOCOL_SHA256,
    FOUNDATION_V3_SHA256,
    SklearnQualityPredictor,
    write_trusted_quality_artifact,
)
from .provider_pin_ablation import NO_PIN_REPRESENTATION
from .quality_features import (
    CANONICAL_PREDICTIVE_FEATURES,
    CanonicalQualityFeatures,
    build_quality_pipeline,
    canonical_feature_matrix,
    canonical_from_production,
    canonical_from_training_row,
    canonicalize_category,
    validate_canonical_skew_audit,
)

DEFAULT_ARTIFACT_DIRECTORY = Path(
    "artifacts/routing-quality/interaction-no-provider-pin-v1"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_frozen_training_inputs() -> None:
    if _sha256(Path("benchmarks/datasets/foundation-v3.json")) != FOUNDATION_V3_SHA256:
        raise ValueError("Foundation V3 hash mismatch")
    if (
        _sha256(Path("benchmarks/protocols/foundation-v3.json"))
        != FOUNDATION_V3_PROTOCOL_SHA256
    ):
        raise ValueError("Foundation V3 protocol hash mismatch")


def _production_features_for_training_row(row) -> RoutingRequestFeatures:
    values = row.features
    return RoutingRequestFeatures(
        category=canonicalize_category(values["category"]),
        category_provenance=CategoryProvenance.CLIENT_HINT,
        prompt_characters=int(values["prompt_characters"]),
        system_prompt_characters=0,
        message_count=1,
        system_prompt_present=False,
        approximate_input_tokens=int(values["approximate_input_tokens"]),
        contains_code=bool(values["contains_code"]),
        requests_structured_output=bool(values["requests_structured_output"]),
        max_output_tokens=int(values["requested_max_output_tokens"]),
        constraint_indicator_count=int(values["constraint_indicator_count"]),
        reasoning_indicator_count=int(values["reasoning_indicator_count"]),
    )


def canonical_training_serving_pairs(dataset) -> tuple[
    tuple[CanonicalQualityFeatures, CanonicalQualityFeatures], ...
]:
    candidates = {candidate.model_id: candidate for candidate in CANDIDATE_MODELS}
    if set(candidates) != {row.candidate_id for row in dataset.rows}:
        raise ValueError("configured candidates differ from frozen Foundation V3 candidates")
    pairs = []
    for row in dataset.rows:
        training = canonical_from_training_row(row)
        production = canonical_from_production(
            _production_features_for_training_row(row), candidates[row.candidate_id]
        )
        pairs.append((training, production))
    return tuple(pairs)


def validate_training_serving_parity(dataset) -> tuple[CanonicalQualityFeatures, ...]:
    pairs = canonical_training_serving_pairs(dataset)
    if any(training != production for training, production in pairs):
        raise ValueError("canonical training/serving parity failed")
    training_rows = tuple(training for training, _ in pairs)
    production_rows = tuple(production for _, production in pairs)
    if not np.array_equal(
        canonical_feature_matrix(training_rows),
        canonical_feature_matrix(production_rows),
    ):
        raise ValueError("canonical vectorization parity failed")
    return training_rows


def build_predictor_artifact(
    root: Path,
    run_id: UUID,
    output_directory: Path,
):
    verify_frozen_training_inputs()
    audit_counts = validate_canonical_skew_audit()
    if tuple(NO_PIN_REPRESENTATION.features) != CANONICAL_PREDICTIVE_FEATURES:
        raise ValueError("canonical feature vector differs from the approved ablation")
    dataset = load_ml_dataset(root, run_id)
    canonical_rows = validate_training_serving_parity(dataset)
    valid_indices = [
        index for index, row in enumerate(dataset.rows) if row.label_status == "valid"
    ]
    missing_count = len(dataset.rows) - len(valid_indices)
    if len(valid_indices) != 216 or missing_count != 8:
        raise ValueError("Foundation V3 valid/missing label counts changed")
    training_rows = tuple(canonical_rows[index] for index in valid_indices)
    targets = np.asarray(
        [int(dataset.rows[index].acceptable is True) for index in valid_indices],
        dtype=int,
    )
    if set(targets.tolist()) != {0, 1}:
        raise ValueError("final fit requires both acceptable target classes")
    pipeline = build_quality_pipeline()
    pipeline.fit(canonical_feature_matrix(training_rows), targets)
    production_rows = tuple(production for _, production in canonical_training_serving_pairs(dataset))
    training_vectors = pipeline.named_steps["preprocess"].transform(
        canonical_feature_matrix(canonical_rows)
    )
    production_vectors = pipeline.named_steps["preprocess"].transform(
        canonical_feature_matrix(production_rows)
    )
    if not np.allclose(training_vectors, production_vectors, rtol=0, atol=1e-15):
        raise ValueError("fitted preprocessing training/serving parity failed")
    probabilities = pipeline.predict_proba(canonical_feature_matrix(training_rows))[:, 1]
    if not np.isfinite(probabilities).all() or not ((0 <= probabilities) & (probabilities <= 1)).all():
        raise ValueError("final fitted pipeline produced invalid probabilities")
    known_candidates = tuple(sorted({row.candidate_id for row in canonical_rows}))
    known_categories = tuple(sorted(
        {row.category for row in canonical_rows}, key=lambda item: item.value
    ))
    model_path, metadata_path, metadata = write_trusted_quality_artifact(
        output_directory,
        pipeline,
        known_candidate_ids=known_candidates,
        known_categories=known_categories,
        valid_training_rows=len(valid_indices),
        missing_label_rows=missing_count,
    )
    loaded = SklearnQualityPredictor.from_trusted_artifact(output_directory)
    sample_row = dataset.rows[0]
    sample_request = _production_features_for_training_row(sample_row)
    sample_candidate = next(
        candidate for candidate in CANDIDATE_MODELS
        if candidate.model_id == sample_row.candidate_id
    )
    prediction = loaded.predict(sample_request, (sample_candidate,))[0]
    if not 0 <= prediction.predicted_acceptability <= 1:
        raise ValueError("reloaded predictor smoke test failed")
    return {
        "model_path": model_path,
        "metadata_path": metadata_path,
        "metadata": metadata,
        "pipeline": pipeline,
        "audit_counts": audit_counts,
        "canonical_parity_rows": len(canonical_rows),
        "vectorization_parity_rows": len(canonical_rows),
        "smoke_prediction": prediction,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the offline INTERACTION_NO_PROVIDER_PIN deployment artifact"
    )
    parser.add_argument("--root", type=Path, default=Path("benchmark-results"))
    parser.add_argument("--run-id", type=UUID, default=FOUNDATION_V3_RUN_ID)
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT_DIRECTORY)
    args = parser.parse_args()
    result = build_predictor_artifact(args.root, args.run_id, args.output)
    metadata = result["metadata"]
    print(json.dumps({
        "artifact_directory": str(args.output),
        "model_sha256": metadata.model_sha256,
        "valid_training_rows": metadata.valid_training_rows,
        "missing_label_rows": metadata.missing_label_rows,
        "canonical_parity_rows": result["canonical_parity_rows"],
        "vectorization_parity_rows": result["vectorization_parity_rows"],
        "audit_counts": result["audit_counts"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
