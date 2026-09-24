import re

from pydantic import Field

from adaptive_llm_gateway.models import InferenceRequest
from adaptive_llm_gateway.models.schemas import DomainModel


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


_CONSTRAINTS = re.compile(r"\b(?:must|only|exactly|exclude|include|return|output|do not|at most|no more than)\b", re.I)
_REASONING = re.compile(r"\b(?:infer|determine|deduce|calculate|after|before|unless|if|therefore)\b", re.I)
_CODE = re.compile(
    r"\x60{3}|\bpython code\b|\bdef\s+\w+\s*\(|\bdefin(?:e|ing)\s+\w+\s*\(|"
    r"\bfunction\s+\w+\s*\(|\bclass\s+\w+",
    re.I,
)


def extract_request_features(request: InferenceRequest) -> RequestFeatures:
    """Derive only information available before model selection."""
    text = " ".join(part for part in (request.system_prompt, request.prompt) if part)
    output_type = getattr(request, "expected_output_type", "text")
    return RequestFeatures(
        category=getattr(request, "category", None),
        prompt_characters=len(request.prompt),
        system_prompt_characters=len(request.system_prompt or ""),
        approximate_input_tokens=len(text.split()),
        contains_code=bool(_CODE.search(text)),
        requests_structured_output=output_type in {"json", "code"},
        max_output_tokens=request.max_output_tokens,
        constraint_indicator_count=len(_CONSTRAINTS.findall(text)),
        reasoning_indicator_count=len(_REASONING.findall(text)),
    )
