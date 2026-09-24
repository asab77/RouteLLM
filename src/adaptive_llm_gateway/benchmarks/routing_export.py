"""Compact, auditable routing-label export derived from immutable run artifacts."""
from __future__ import annotations

import asyncio
import json
import os
from decimal import Decimal
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from pydantic import Field, JsonValue, model_validator

from adaptive_llm_gateway.evaluation.models import EvaluationResult
from adaptive_llm_gateway.evaluation.repository import FileEvaluationRepository
from adaptive_llm_gateway.models.schemas import DomainModel
from .features import RequestFeatures

ROUTING_DATASET_VERSION = "1.0.0"


class RoutingCandidateSnapshot(DomainModel):
    internal_id: str
    provider: str
    upstream_model_slug: str
    upstream_provider_pin: str
    service_tier: str
    region: str
    input_cost_per_1m_tokens: Decimal = Field(ge=0)
    output_cost_per_1m_tokens: Decimal = Field(ge=0)
    context_window: int = Field(gt=0)
    capabilities: dict[str, JsonValue]
    reasoning_effort: str | None = None


class RoutingDatasetRow(DomainModel):
    artifact_version: str = ROUTING_DATASET_VERSION
    run_id: UUID
    benchmark_result_id: UUID
    request_id: str
    task_id: str
    dataset_name: str
    dataset_version: str
    dataset_sha256: str
    request_features: RequestFeatures
    candidate: RoutingCandidateSnapshot
    provider_outcome: Literal["success", "failure"]
    provider_error_category: str | None = None
    generated_at: str
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    latency_ms: float = Field(ge=0)
    candidate_cost_usd: Decimal | None = Field(default=None, ge=0)
    label_status: Literal["valid", "missing"]
    acceptable: bool | None
    missing_label_reason: str | None = None
    quality_score: float | None = Field(default=None, ge=0, le=1)
    evaluator_name: str
    evaluator_version: str
    evaluation_status: str
    evaluation_call_made: bool
    evaluation_input_tokens: int = Field(ge=0)
    evaluation_output_tokens: int = Field(ge=0)
    evaluation_latency_ms: float = Field(ge=0)
    evaluation_cost_usd: Decimal = Field(ge=0)
    analysis_difficulty: Literal["easy", "medium", "hard"] | None = None
    raw_result_path: str
    evaluation_path: str

    @model_validator(mode="after")
    def label_is_only_present_after_valid_evaluation(self):
        if self.label_status == "valid":
            if self.provider_outcome != "success" or self.evaluation_status != "evaluated" or self.acceptable is None:
                raise ValueError("Valid labels require successful, fully evaluated responses")
            if self.missing_label_reason is not None:
                raise ValueError("Valid labels cannot have a missing-label reason")
        elif self.acceptable is not None or not self.missing_label_reason:
            raise ValueError("Missing labels require no binary target and an explicit reason")
        return self


def _failure_metric(result, key: str, expected_type):
    value = result.error_details.get(key)
    return value if type(value) is expected_type and value >= 0 else None


def _failure_cost(result) -> Decimal | None:
    value = result.error_details.get("estimated_cost_usd")
    try:
        cost = Decimal(str(value))
    except Exception:
        return None
    return cost if cost.is_finite() and cost >= 0 else None


def _atomic_write(path: Path, data: str) -> None:
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


async def export_routing_dataset(root: Path, run_id: UUID) -> tuple[Path, Path, tuple[RoutingDatasetRow, ...]]:
    repository = FileEvaluationRepository(root)
    run, results = await repository.load_run(run_id)
    directory = root / str(run_id)
    evaluations: dict[UUID, EvaluationResult] = {}
    for path in sorted((directory / "evaluations").glob("*.json")):
        evaluation = EvaluationResult.model_validate_json(await asyncio.to_thread(path.read_bytes))
        evaluations[evaluation.benchmark_result_id] = evaluation
    if set(evaluations) != {result.result_id for result in results}:
        raise ValueError("Every benchmark result must have exactly one persisted evaluation")

    task_by_id = {task.task_id: task for task in run.dataset.tasks}
    model_by_id = {model.model_id: model for model in run.models}
    frozen = run.configuration.get("frozen_model_configuration", {})
    features = run.configuration.get("request_features", {})
    if not isinstance(frozen, dict) or not isinstance(features, dict):
        raise ValueError("Run manifest lacks frozen model or request-feature snapshots")

    rows = []
    for result in sorted(results, key=lambda item: (item.task_id, item.model_id)):
        evaluation = evaluations[result.result_id]
        task = task_by_id[result.task_id]
        model = model_by_id[result.model_id]
        snapshot = frozen.get(result.model_id)
        feature_snapshot = features.get(result.task_id)
        if not isinstance(snapshot, dict) or not isinstance(feature_snapshot, dict):
            raise ValueError("Run manifest snapshot is incomplete")
        response = result.response
        if not result.success:
            label_status, acceptable = "missing", None
            missing_reason = f"provider_failure:{result.error_category or 'unknown'}"
        elif evaluation.evaluation_status != "evaluated" or evaluation.acceptable is None:
            label_status, acceptable = "missing", None
            missing_reason = f"evaluation_incomplete:{evaluation.evaluation_status}"
        else:
            label_status, acceptable, missing_reason = "valid", evaluation.acceptable, None
        rows.append(RoutingDatasetRow(
            run_id=run.run_id, benchmark_result_id=result.result_id,
            request_id=result.request_id, task_id=result.task_id,
            dataset_name=run.dataset.name, dataset_version=run.dataset.version,
            dataset_sha256=run.dataset_sha256,
            request_features=RequestFeatures.model_validate(feature_snapshot),
            candidate=RoutingCandidateSnapshot(
                internal_id=model.model_id, provider=model.provider,
                upstream_model_slug=model.provider_model_name,
                upstream_provider_pin=str(snapshot["upstream_provider_pin"]),
                service_tier=str(snapshot["service_tier"]), region=str(snapshot["region"]),
                input_cost_per_1m_tokens=model.input_cost_per_1m_tokens,
                output_cost_per_1m_tokens=model.output_cost_per_1m_tokens,
                context_window=model.context_window,
                capabilities=model.capabilities.model_dump(mode="json"),
                reasoning_effort=(str(snapshot["reasoning_effort"])
                                  if snapshot.get("reasoning_effort") is not None else None)),
            provider_outcome="success" if result.success else "failure",
            provider_error_category=result.error_category,
            generated_at=result.created_at.isoformat(),
            input_tokens=(response.input_tokens if response else
                          _failure_metric(result, "input_tokens", int)),
            output_tokens=(response.output_tokens if response else
                           _failure_metric(result, "output_tokens", int)),
            latency_ms=(response.latency_ms if response else
                        (_failure_metric(result, "adapter_latency_ms", float)
                         or result.latency_ms)),
            candidate_cost_usd=(response.estimated_cost_usd if response else
                                _failure_cost(result)),
            label_status=label_status, acceptable=acceptable,
            missing_label_reason=missing_reason,
            quality_score=evaluation.quality_score if label_status == "valid" else None,
            evaluator_name=evaluation.evaluator_name,
            evaluator_version=evaluation.evaluator_version,
            evaluation_status=evaluation.evaluation_status,
            evaluation_call_made=evaluation.evaluation_call_made,
            evaluation_input_tokens=evaluation.evaluation_input_tokens,
            evaluation_output_tokens=evaluation.evaluation_output_tokens,
            evaluation_latency_ms=evaluation.evaluation_latency_ms,
            evaluation_cost_usd=evaluation.evaluation_cost_usd,
            analysis_difficulty=task.difficulty,
            raw_result_path=f"results/{result.result_id}.json",
            evaluation_path=f"evaluations/{result.result_id}.json",
        ))

    dataset_path = directory / "routing-dataset.jsonl"
    schema_path = directory / "routing-dataset-schema.json"
    data = "".join(row.model_dump_json() + "\n" for row in rows)
    schema = {
        "artifact_version": ROUTING_DATASET_VERSION,
        "format": "JSON Lines; one prompt-model row per line",
        "primary_target": "acceptable, present only when label_status is valid",
        "request_feature_boundary": "Only request_features and candidate identity/configuration are eligible model inputs.",
        "analysis_only_fields": ["analysis_difficulty", "quality_score", "latency_ms", "input_tokens", "output_tokens", "candidate_cost_usd"],
        "raw_text_policy": "Generated responses remain in raw result artifacts referenced by raw_result_path.",
        "row_json_schema": RoutingDatasetRow.model_json_schema(),
    }
    await asyncio.to_thread(_atomic_write, dataset_path, data)
    await asyncio.to_thread(_atomic_write, schema_path, json.dumps(schema, indent=2))
    return dataset_path, schema_path, tuple(rows)
