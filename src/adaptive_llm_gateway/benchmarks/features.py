from pydantic import Field

from adaptive_llm_gateway.models import InferenceRequest
from adaptive_llm_gateway.models.schemas import DomainModel
from adaptive_llm_gateway.routing.features import (
    approximate_input_tokens,
    constraint_indicator_count,
    reasoning_indicator_count,
    request_text,
    text_contains_code,
)


class RequestFeatures(DomainModel):
    category: str | None = None
    prompt_characters: int = Field(ge=0)
    system_prompt_characters: int = Field(ge=0)
    approximate_input_tokens: int = Field(ge=0)
    contains_code: bool
    requests_structured_output: bool
    max_output_tokens: int = Field(gt=0)
    constraint_indicator_count: int = Field(ge=0)
    reasoning_indicator_count: int = Field(ge=0)


def extract_request_features(request: InferenceRequest) -> RequestFeatures:
    """Derive only information available before model selection."""
    text = request_text(request)
    output_type = getattr(request, "expected_output_type", "text")
    return RequestFeatures(
        category=getattr(request, "category", None),
        prompt_characters=len(request.prompt),
        system_prompt_characters=len(request.system_prompt or ""),
        approximate_input_tokens=approximate_input_tokens(request),
        contains_code=text_contains_code(text),
        requests_structured_output=output_type in {"json", "code"},
        max_output_tokens=request.max_output_tokens,
        constraint_indicator_count=constraint_indicator_count(text),
        reasoning_indicator_count=reasoning_indicator_count(text),
    )
