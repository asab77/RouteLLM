from decimal import Decimal, localcontext

import pytest

from adaptive_llm_gateway.models import ModelConfig
from adaptive_llm_gateway.pricing import calculate_cost


@pytest.mark.parametrize("inputs,outputs,expected", [
    (0, 0, "0"), (1, 0, "0.00000015"), (0, 1, "0.00000060"),
    (1000, 500, "0.00045"), (1_000_000, 1_000_000, "0.75"),
])
def test_exact_cost(model, inputs, outputs, expected):
    assert calculate_cost(input_tokens=inputs, output_tokens=outputs, model=model) == Decimal(expected)


@pytest.mark.parametrize("name", ["input_tokens", "output_tokens"])
@pytest.mark.parametrize("value,error", [(-1, ValueError), (True, TypeError),
                                          (1.5, TypeError), ("3", TypeError)])
def test_invalid_usage(model, name, value, error):
    usage = {"input_tokens": 0, "output_tokens": 0, name: value}
    with pytest.raises(error):
        calculate_cost(**usage, model=model)


def test_precision_does_not_depend_on_callers_context(model):
    precise_model = ModelConfig(**(model.model_dump() | {
        "input_cost_per_1m_tokens": "123456789.123456789123456789",
        "output_cost_per_1m_tokens": "0.000000000000000001",
    }))
    with localcontext() as context:
        context.prec = 3
        result = calculate_cost(input_tokens=1_000_000, output_tokens=1_000_000,
                                model=precise_model)
        assert context.prec == 3
    assert result == Decimal("123456789.123456789123456790")
