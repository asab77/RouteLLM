"""Trusted runtime configuration for optional adaptive inference."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import ConfigDict, model_validator

from adaptive_llm_gateway.models.schemas import DomainModel, Identifier

if TYPE_CHECKING:
    from .adaptive import AdaptiveInferenceService
    from .service import InferenceService

ADAPTIVE_ARTIFACT_PATH_ENV = "ROUTELLM_ADAPTIVE_ARTIFACT_PATH"
ADAPTIVE_CANDIDATES_ENV = "ROUTELLM_ADAPTIVE_CANDIDATES"


class AdaptiveRoutingConfig(DomainModel):
    """All-or-nothing process configuration; request policy is intentionally absent."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    artifact_path: Path | None = None
    candidate_model_ids: tuple[Identifier, ...] = ()

    @property
    def enabled(self) -> bool:
        return self.artifact_path is not None

    @model_validator(mode="after")
    def validate_complete_configuration(self) -> "AdaptiveRoutingConfig":
        if self.artifact_path is None and not self.candidate_model_ids:
            return self
        if self.artifact_path is None or not self.candidate_model_ids:
            raise ValueError(
                "adaptive routing requires both an artifact path and candidate models"
            )
        if len(self.candidate_model_ids) != len(set(self.candidate_model_ids)):
            raise ValueError("adaptive candidate model IDs must be unique")
        return self

    @classmethod
    def from_environment(cls) -> "AdaptiveRoutingConfig":
        artifact = os.environ.get(ADAPTIVE_ARTIFACT_PATH_ENV, "").strip()
        candidates = os.environ.get(ADAPTIVE_CANDIDATES_ENV, "").strip()
        if not artifact and not candidates:
            return cls()
        if not artifact or not candidates:
            raise ValueError(
                "adaptive routing environment configuration is incomplete"
            )
        parts = candidates.split(",")
        normalized = tuple(part.strip() for part in parts)
        if any(not part for part in normalized):
            raise ValueError("adaptive candidate list contains an empty model ID")
        return cls(
            artifact_path=Path(artifact),
            candidate_model_ids=normalized,
        )


@dataclass(frozen=True)
class AdaptiveRuntime:
    service: "AdaptiveInferenceService"
    candidate_model_ids: tuple[str, ...]


def build_adaptive_runtime(
    inference_service: "InferenceService",
    config: AdaptiveRoutingConfig,
) -> AdaptiveRuntime | None:
    """Load a configured trusted artifact once; disabled mode imports no sklearn."""
    if not config.enabled:
        return None
    from .adaptive import AdaptiveInferenceService

    service = AdaptiveInferenceService.from_trusted_artifact(
        inference_service,
        config.artifact_path,
        candidate_model_ids=config.candidate_model_ids,
    )
    return AdaptiveRuntime(
        service=service,
        candidate_model_ids=tuple(config.candidate_model_ids),
    )


__all__ = [
    "ADAPTIVE_ARTIFACT_PATH_ENV",
    "ADAPTIVE_CANDIDATES_ENV",
    "AdaptiveRoutingConfig",
    "AdaptiveRuntime",
    "build_adaptive_runtime",
]
