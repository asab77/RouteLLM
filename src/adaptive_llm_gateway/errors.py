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
