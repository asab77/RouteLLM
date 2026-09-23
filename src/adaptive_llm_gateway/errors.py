"""Application/provider errors independent of HTTP status codes."""


class ModelDisabledError(ValueError):
    """The selected model is disabled."""


class ContextLimitError(ValueError):
    """Input plus requested output exceeds the model context window."""


class ProviderUnavailableError(RuntimeError):
    """No adapter factory is registered for a provider."""


class ProviderFailureError(RuntimeError):
    """An adapter failed to construct or generate a valid result."""
