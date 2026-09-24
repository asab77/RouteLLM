"""One inexpensive call only. Never run a benchmark from this test."""
import pytest
from adaptive_llm_gateway.bootstrap import create_development_service, configure_gateway
from adaptive_llm_gateway.models import InferenceRequest
from adaptive_llm_gateway.providers.gateway_config import GatewaySettings

pytestmark = [pytest.mark.real_provider, pytest.mark.asyncio]


async def test_gateway_nano_smoke():
    settings=GatewaySettings.from_environment()
    if settings.api_key is None:
        pytest.skip('Set AI_GATEWAY_API_KEY in the ignored local .env and export it')
    service=create_development_service()
    configure_gateway(service,settings)
    result=await service.generate('gateway-nano',InferenceRequest(prompt='Reply with OK.',max_output_tokens=8,temperature=0))
    assert result.model_id=='gateway-nano' and result.text.strip()
    assert result.input_tokens>0 and result.output_tokens>0
    assert result.estimated_cost_usd>=0 and result.latency_ms>0
