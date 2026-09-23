import pytest

from adaptive_llm_gateway.models import ModelConfig
from adaptive_llm_gateway.registry import DuplicateModelError, ModelNotFoundError, ModelRegistry


def test_registration_retrieval_and_filtering(model):
    registry = ModelRegistry()
    assert registry.list_models() == []
    assert registry.list_enabled_models() == []
    disabled = ModelConfig(**(model.model_dump() | {"model_id": "disabled", "enabled": False}))
    registry.register(disabled)
    registry.register(model)
    assert registry.get(model.model_id) == model
    assert registry.get("disabled") == disabled
    assert registry.list_models() == [disabled, model]
    assert registry.list_enabled_models() == [model]
    registry.list_models().clear()
    assert len(registry.list_models()) == 2


def test_duplicate_does_not_overwrite_existing_model(model):
    registry = ModelRegistry()
    registry.register(model)
    replacement = ModelConfig(**(model.model_dump() | {"enabled": False}))
    with pytest.raises(DuplicateModelError, match="fake-small"):
        registry.register(replacement)
    assert registry.get(model.model_id).enabled


def test_missing_model():
    with pytest.raises(ModelNotFoundError, match="missing"):
        ModelRegistry().get("missing")
