"""Pricing independent of providers and the ambient Decimal precision."""

from decimal import Decimal, localcontext

from adaptive_llm_gateway.models import ModelConfig


def calculate_cost(
    *, input_tokens: int, output_tokens: int, model: ModelConfig
) -> Decimal:
    """Return exact USD cost without quantization or rounding to cents.

    Finite decimal prices multiplied by integers and divided by 10**6 have
    finite decimal results. Reserve enough precision to preserve every digit,
    including when adding prices with different scales. Round only for display.
    """
    for name, count in (("input_tokens", input_tokens), ("output_tokens", output_tokens)):
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError(f"{name} must be an integer")
        if count < 0:
            raise ValueError(f"{name} must be non-negative")

    rates = (model.input_cost_per_1m_tokens, model.output_cost_per_1m_tokens)
    required_digits = sum(
        len(rate.as_tuple().digits) + abs(int(rate.as_tuple().exponent))
        for rate in rates
    ) + max(len(str(input_tokens)), len(str(output_tokens))) + 10
    with localcontext() as context:
        context.prec = max(28, required_digits)
        return (
            rates[0] * input_tokens + rates[1] * output_tokens
        ) / Decimal(1_000_000)


def calculate_projected_cost(
    *, approximate_input_tokens: int, effective_max_output_tokens: int,
    model: ModelConfig,
) -> Decimal:
    """Return the exact pre-generation upper-bound estimate used for routing.

    Projected cost deliberately uses request-visible token estimates and the
    configured output allowance. It never depends on realized generation data.
    """
    return calculate_cost(
        input_tokens=approximate_input_tokens,
        output_tokens=effective_max_output_tokens,
        model=model,
    )
