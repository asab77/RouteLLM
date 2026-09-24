import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from pydantic import Field, JsonValue, PrivateAttr, model_validator
from adaptive_llm_gateway.models import InferenceRequest, InferenceResponse, ModelConfig
from adaptive_llm_gateway.models.schemas import DomainModel


class BenchmarkTask(InferenceRequest):
    task_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    category: Literal["qa", "summarization", "extraction", "classification", "json", "reasoning", "coding"]
    expected_output_type: Literal["text", "json", "code"] = "text"
    evaluation_metadata: dict[str, JsonValue] = Field(default_factory=dict)
    acceptable_threshold: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    tags: tuple[str, ...] = ()
    temperature: float = Field(default=0, ge=0, le=2, allow_inf_nan=False)

    def to_request(self) -> InferenceRequest:
        return InferenceRequest(**self.model_dump(include=set(InferenceRequest.model_fields)))

    @property
    def difficulty(self) -> Literal["easy", "medium", "hard"] | None:
        value = self.evaluation_metadata.get("difficulty")
        return value if value in {"easy", "medium", "hard"} else None


class BenchmarkDataset(DomainModel):
    _source_sha256: str | None = PrivateAttr(default=None)

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    tasks: tuple[BenchmarkTask, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_ids(self):
        if len({task.task_id for task in self.tasks}) != len(self.tasks):
            raise ValueError("Dataset task IDs must be unique")
        return self

    @property
    def sha256(self) -> str:
        if self._source_sha256 is not None:
            return self._source_sha256
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


def load_dataset(path: Path) -> BenchmarkDataset:
    raw = path.read_bytes()
    dataset = BenchmarkDataset.model_validate_json(raw)
    dataset._source_sha256 = hashlib.sha256(raw).hexdigest()
    return dataset


class BenchmarkRun(DomainModel):
    run_id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    dataset: BenchmarkDataset
    dataset_sha256: str
    selected_task_ids: tuple[str, ...]
    models: tuple[ModelConfig, ...]
    configuration: dict[str, JsonValue]


class BenchmarkResult(DomainModel):
    result_id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    request_id: str
    task_id: str
    model_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    success: bool
    response: InferenceResponse | None = None
    error_category: str | None = None
    error_details: dict[str, JsonValue] = Field(default_factory=dict)
    latency_ms: float = Field(ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def consistent_outcome(self):
        if self.success != (self.response is not None) or self.success != (self.error_category is None):
            raise ValueError("Result must contain either a response or a failure category")
        if self.response is not None and self.response.model_id != self.model_id:
            raise ValueError("Response model must match the selected model")
        if self.success and self.error_details:
            raise ValueError("Successful results cannot contain provider error details")
        return self
