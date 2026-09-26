"""Application/provider errors independent of HTTP status codes."""

from enum import StrEnum


class ModelDisabledError(ValueError):
    """The selected model is disabled."""


class ContextLimitError(ValueError):
    """Input plus requested output exceeds the model context window."""


class ProviderUnavailableError(RuntimeError):
    """No adapter factory is registered for a provider."""


class ProviderFailureError(RuntimeError):
    """An adapter failed to construct or generate a valid result."""


class AdaptiveRoutingUnavailableError(RuntimeError):
    """Adaptive inference is not configured for this application instance."""



class GatewayErrorCategory(StrEnum):
    NOT_CONFIGURED = "gateway_not_configured"
    AUTHENTICATION = "gateway_authentication"
    INVALID_MODEL = "gateway_invalid_model"
    INVALID_REQUEST = "gateway_invalid_request"
    CONTEXT_LIMIT = "context_limit_exceeded"
    RATE_LIMIT = "gateway_rate_limit"
    TIMEOUT = "gateway_timeout"
    UPSTREAM = "gateway_upstream"
    MALFORMED_RESPONSE = "gateway_malformed_response"
    MISSING_USAGE = "gateway_missing_usage"
    EMPTY_RESPONSE = "gateway_empty_response"


class GatewayError(ProviderFailureError):
    """Public text stays normalized; safe structured diagnostics remain internal."""
    def __init__(self, category: GatewayErrorCategory, *, diagnostics: dict | None = None) -> None:
        self.category = category
        self.diagnostics = dict(diagnostics or {})
        super().__init__(category.value)


class EvaluationNotFoundError(FileNotFoundError):
    """A benchmark run or its evaluation summary does not exist."""


class EvaluationArtifactError(ValueError):
    """A benchmark artifact is malformed, inconsistent, or incomplete."""


class RoutingPolicyError(ValueError):
    """The cost-aware routing policy received an invalid selection request."""


class NoEligibleCandidatesError(RoutingPolicyError):
    """No eligible model was supplied to the routing policy."""


class InvalidQualityThresholdError(RoutingPolicyError):
    """The caller supplied a non-finite threshold outside [0, 1]."""


class DuplicateCandidatePredictionError(RoutingPolicyError):
    """More than one prediction was supplied for a model identifier."""


class QualityPredictorError(RuntimeError):
    """A deployable quality predictor could not safely serve a request."""


class PredictorInputCompatibilityError(QualityPredictorError, ValueError):
    """Request or candidate input is outside the fitted artifact contract."""


class MissingRoutingCategoryError(PredictorInputCompatibilityError):
    """The learned predictor requires an explicit canonical category."""


class UnsupportedPredictorCandidateError(PredictorInputCompatibilityError):
    """A candidate identity was not represented during artifact training."""


class PredictorArtifactError(QualityPredictorError):
    """Trusted predictor build output is missing, damaged, or incompatible."""


class PredictorArtifactNotFoundError(PredictorArtifactError, FileNotFoundError):
    """Required predictor metadata or binary output does not exist."""


class CorruptPredictorArtifactError(PredictorArtifactError):
    """Predictor metadata or serialized pipeline cannot be decoded safely."""


class IncompatibleArtifactFormatError(PredictorArtifactError):
    """Predictor artifact format is not supported by this application."""


class IncompatiblePredictorFormulationError(PredictorArtifactError):
    """Artifact uses a different learned formulation."""


class IncompatibleFeatureSchemaError(PredictorArtifactError):
    """Artifact canonical feature schema is incompatible."""


class IncompatibleCategoryTaxonomyError(PredictorArtifactError):
    """Artifact category taxonomy is incompatible."""


class PredictorArtifactChecksumError(PredictorArtifactError):
    """Serialized predictor bytes do not match trusted metadata."""
