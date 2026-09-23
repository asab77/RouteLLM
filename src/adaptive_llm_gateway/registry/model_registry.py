from adaptive_llm_gateway.models import ModelConfig


class DuplicateModelError(ValueError):
    """A model with this internal ID is already registered."""


class ModelNotFoundError(KeyError):
    """No model with this internal ID is registered."""


class ModelRegistry:
    """Process-local catalog. Lists preserve registration order."""

    def __init__(self) -> None:
        self._models: dict[str, ModelConfig] = {}

    def register(self, model: ModelConfig) -> None:
        if model.model_id in self._models:
            raise DuplicateModelError(f"Model already registered: {model.model_id!r}")
        self._models[model.model_id] = model

    def get(self, model_id: str) -> ModelConfig:
        try:
            return self._models[model_id]
        except KeyError:
            raise ModelNotFoundError(f"Unknown model: {model_id!r}") from None

    def list_models(self) -> list[ModelConfig]:
        return list(self._models.values())

    def list_enabled_models(self) -> list[ModelConfig]:
        return [model for model in self._models.values() if model.enabled]
