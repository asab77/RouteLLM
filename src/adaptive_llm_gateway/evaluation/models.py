from datetime import datetime, timezone
from decimal import Decimal
from typing import Literal
from uuid import UUID

from pydantic import Field, JsonValue, model_validator

from adaptive_llm_gateway.models.schemas import DomainModel


class EvaluationResult(DomainModel):
    benchmark_result_id: UUID
    run_id: UUID
    task_id: str
    model_id: str
    evaluator_name: str
    evaluator_version: str
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    evaluation_status: Literal["evaluated", "partial", "requires_semantic_judge",
                               "requires_functional_execution"] = "evaluated"
    quality_score: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    acceptable_threshold: float = Field(ge=0, le=1, allow_inf_nan=False)
    acceptable: bool | None
    reason: str
    component_scores: dict[str, float] = Field(default_factory=dict)
    details: dict[str, JsonValue] = Field(default_factory=dict)
    evaluation_call_made: bool = False
    evaluation_input_tokens: int = Field(default=0, ge=0)
    evaluation_output_tokens: int = Field(default=0, ge=0)
    evaluation_latency_ms: float = Field(default=0, ge=0, allow_inf_nan=False)
    evaluation_cost_usd: Decimal = Field(default=Decimal(0), ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def consistent_acceptability(self):
        if self.evaluation_status == "evaluated":
            if self.quality_score is None or self.acceptable != (self.quality_score >= self.acceptable_threshold):
                raise ValueError("Evaluated results require score-derived acceptability")
        elif self.acceptable is not None:
            raise ValueError("Incomplete evaluations cannot claim acceptability")
        if any(not 0 <= score <= 1 for score in self.component_scores.values()):
            raise ValueError("Component scores must be between zero and one")
        return self


class MetricGroup(DomainModel):
    model_id: str | None = None
    category: str | None = None
    evaluated_tasks: int = Field(ge=0)
    fully_evaluated_tasks: int = Field(default=0, ge=0)
    incomplete_evaluations: int = Field(default=0, ge=0)
    successful_responses: int = Field(ge=0)
    acceptable_responses: int = Field(ge=0)
    mean_quality_score: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    acceptable_rate: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    total_estimated_cost_usd: Decimal = Field(ge=0, allow_inf_nan=False)
    average_cost_per_task_usd: Decimal = Field(ge=0, allow_inf_nan=False)
    cost_per_acceptable_response_usd: Decimal | None = Field(default=None, ge=0, allow_inf_nan=False)
    average_latency_ms: float = Field(ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def consistent_evaluation_counts(self):
        if self.evaluated_tasks and not self.fully_evaluated_tasks and not self.incomplete_evaluations:
            object.__setattr__(self, "fully_evaluated_tasks", self.evaluated_tasks)
        if self.fully_evaluated_tasks + self.incomplete_evaluations != self.evaluated_tasks:
            raise ValueError("Evaluation status counts must equal evaluated tasks")
        if self.acceptable_responses > self.fully_evaluated_tasks:
            raise ValueError("Acceptable responses cannot exceed fully evaluated tasks")
        return self


class ModelComparison(DomainModel):
    left_model_id: str
    right_model_id: str
    mean_quality_difference: float | None = Field(default=None, ge=-1, le=1, allow_inf_nan=False)
    total_cost_difference_usd: Decimal
    average_latency_difference_ms: float = Field(allow_inf_nan=False)


class EvaluationSummary(DomainModel):
    run_id: UUID
    dataset_sha256: str
    evaluation_version: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    overall: MetricGroup
    by_model: tuple[MetricGroup, ...]
    by_model_category: tuple[MetricGroup, ...]
    comparisons: tuple[ModelComparison, ...]
    candidate_inference_cost_usd: Decimal = Field(default=Decimal(0), ge=0, allow_inf_nan=False)
    judge_evaluation_cost_usd: Decimal = Field(default=Decimal(0), ge=0, allow_inf_nan=False)
    judge_calls: int = Field(default=0, ge=0)
    judge_input_tokens: int = Field(default=0, ge=0)
    judge_output_tokens: int = Field(default=0, ge=0)
    judge_latency_ms: float = Field(default=0, ge=0, allow_inf_nan=False)
    evaluation_configuration: dict[str, JsonValue] = Field(default_factory=dict)
