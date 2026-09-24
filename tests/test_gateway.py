import asyncio
import json
from decimal import Decimal

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError

from adaptive_llm_gateway.api.app import create_app
from adaptive_llm_gateway.bootstrap import configure_gateway, create_development_service
from adaptive_llm_gateway.errors import GatewayError, GatewayErrorCategory as Category
from adaptive_llm_gateway.models import InferenceRequest
from adaptive_llm_gateway.providers.gateway_config import GatewaySettings, REAL_MODELS, UPSTREAM_PROVIDERS
from adaptive_llm_gateway.providers.resolver import ProviderResolver
from adaptive_llm_gateway.providers.vercel import ENDPOINT, VercelGatewayProvider


def valid_body():
    return {'choices':[{'message':{'content':'answer'}, 'finish_reason':'stop'}],
            'usage':{'prompt_tokens':10,'completion_tokens':5}}


def adapter(handler, *, settings=None, model=None):
    return VercelGatewayProvider(model or REAL_MODELS[0], settings or GatewaySettings(api_key=SecretStr('test-only-key')),
                                  transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_request_translation_identity_usage_and_exact_cost():
    seen=[]
    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=valid_body())
    response = await adapter(handler).generate(InferenceRequest(prompt='hello', system_prompt='be brief',
                                                               max_output_tokens=12, temperature=0.25))
    assert len(seen)==1
    assert str(seen[0].url)==ENDPOINT
    assert seen[0].headers['authorization']=='Bearer test-only-key'
    assert json.loads(seen[0].content)=={'model':'openai/gpt-4.1-nano',
        'messages':[{'role':'system','content':'be brief'},{'role':'user','content':'hello'}],
        'stream':False,'max_tokens':12,'temperature':0.25,'providerOptions':{'gateway':{'only':['openai']}}}
    assert response.text=='answer' and response.model_id=='gateway-nano' and response.provider=='vercel'
    assert response.input_tokens==10 and response.output_tokens==5
    assert response.estimated_cost_usd==Decimal('0.000003')
    assert response.latency_ms > 0


@pytest.mark.asyncio
@pytest.mark.parametrize('model', REAL_MODELS)
async def test_explicit_model_and_single_provider(model):
    def handler(request):
        body=json.loads(request.content)
        assert body['model']==model.provider_model_name
        assert body['messages']==[{'role':'user','content':'hello'}]
        assert body['providerOptions']['gateway']=={'only':[UPSTREAM_PROVIDERS[model.provider_model_name]]}
        return httpx.Response(200,json=valid_body())
    response=await adapter(handler,model=model).generate(InferenceRequest(prompt='hello'))
    assert response.model_id==model.model_id


@pytest.mark.asyncio
@pytest.mark.parametrize('status,category', [(401,Category.AUTHENTICATION),(403,Category.AUTHENTICATION),
    (404,Category.INVALID_MODEL),(429,Category.RATE_LIMIT),(408,Category.TIMEOUT),(504,Category.TIMEOUT),
    (500,Category.UPSTREAM),(502,Category.UPSTREAM),(400,Category.INVALID_REQUEST),(302,Category.UPSTREAM)])
async def test_http_errors_are_sanitized_without_retries(status,category):
    calls=[]
    def handler(request):
        calls.append(request)
        return httpx.Response(status,json={'error':{'message':'secret credential and internal details'}})
    with pytest.raises(GatewayError) as error:
        await adapter(handler).generate(InferenceRequest(prompt='private prompt'))
    assert error.value.category==category and len(calls)==1
    assert 'secret' not in str(error.value) and 'private' not in str(error.value)


@pytest.mark.asyncio
@pytest.mark.parametrize('code,category', [('context_length_exceeded',Category.CONTEXT_LIMIT),('model_not_found',Category.INVALID_MODEL)])
async def test_normalizes_machine_error_codes(code,category):
    with pytest.raises(GatewayError) as error:
        await adapter(lambda request:httpx.Response(400,json={'error':{'code':code}})).generate(InferenceRequest(prompt='hi'))
    assert error.value.category==category


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['http_timeout','network','deadline'])
async def test_timeout_and_transport_failures(kind):
    async def handler(request):
        if kind=='http_timeout':
            raise httpx.ReadTimeout('sensitive')
        if kind=='network':
            raise httpx.ConnectError('sensitive')
        await asyncio.Event().wait()
    with pytest.raises(GatewayError) as error:
        await adapter(handler,settings=GatewaySettings(api_key=SecretStr('test-only-key'),timeout_seconds=0.01)).generate(InferenceRequest(prompt='hi'))
    assert error.value.category==(Category.UPSTREAM if kind=='network' else Category.TIMEOUT)
    assert 'sensitive' not in str(error.value)


@pytest.mark.asyncio
@pytest.mark.parametrize('usage', [None,{}, {'prompt_tokens':1}, {'completion_tokens':1}])
async def test_missing_usage_is_explicit(usage):
    body=valid_body(); body['usage']=usage
    with pytest.raises(GatewayError) as error:
        await adapter(lambda request:httpx.Response(200,json=body)).generate(InferenceRequest(prompt='hi'))
    assert error.value.category==Category.MISSING_USAGE


@pytest.mark.asyncio
@pytest.mark.parametrize('body', [[],{}, {'choices':[]}, {'choices':[{'message':{'content':None}}]},
    {'choices':[{'message':{'content':'x'}}], 'usage':{'prompt_tokens':True,'completion_tokens':1}},
    {'choices':[{'message':{'content':'x'}}], 'usage':{'prompt_tokens':-1,'completion_tokens':1}}])
async def test_malformed_response(body):
    with pytest.raises(GatewayError) as error:
        await adapter(lambda request:httpx.Response(200,json=body)).generate(InferenceRequest(prompt='hi'))
    assert error.value.category==Category.MALFORMED_RESPONSE


@pytest.mark.asyncio
async def test_invalid_json_response():
    with pytest.raises(GatewayError) as error:
        await adapter(lambda request:httpx.Response(200,text='not JSON')).generate(InferenceRequest(prompt='hi'))
    assert error.value.category==Category.MALFORMED_RESPONSE


@pytest.mark.asyncio
async def test_missing_key_never_calls_network():
    def handler(request):
        pytest.fail('Network should not be called')
    with pytest.raises(GatewayError) as error:
        await adapter(handler,settings=GatewaySettings()).generate(InferenceRequest(prompt='hi'))
    assert error.value.category==Category.NOT_CONFIGURED


def test_environment_redaction_missing_key_and_registration(monkeypatch):
    service=create_development_service()
    configure_gateway(service,GatewaySettings.from_environment())
    assert len(service.list_models())==2
    monkeypatch.setenv('AI_GATEWAY_API_KEY','private-value')
    settings=GatewaySettings.from_environment()
    assert 'private-value' not in repr(settings)
    configure_gateway(service,settings)
    assert {m.model_id for m in service.list_models()} >= {'gateway-nano','gateway-mini','gateway-sonnet'}


@pytest.mark.parametrize('timeout',[0,-1,301,float('inf'),float('nan')])
def test_invalid_timeout(timeout):
    with pytest.raises(ValidationError):
        GatewaySettings(timeout_seconds=timeout)


@pytest.mark.parametrize('status,expected',[(401,502),(429,429),(504,504)])
def test_api_and_telemetry_keep_normalized_gateway_error(status,expected):
    class Repository:
        events=[]
        async def record(self,event): self.events.append(event)
    service=create_development_service()
    service.registry.register(REAL_MODELS[0])
    service.resolver.register('vercel',lambda model:adapter(lambda request:httpx.Response(status,json={'error':'secret'})))
    repository=Repository(); service.telemetry=repository
    with TestClient(create_app(service)) as client:
        response=client.post('/v1/inference',json={'model_id':'gateway-nano','prompt':'private'})
    assert response.status_code==expected
    assert 'secret' not in response.text
    assert repository.events[0].error_category==response.json()['error']['code']


def test_successful_api_gateway_call_uses_existing_telemetry():
    class Repository:
        events=[]
        async def record(self,event): self.events.append(event)
    service=create_development_service()
    service.registry.register(REAL_MODELS[0])
    service.resolver.register('vercel',lambda model:adapter(lambda request:httpx.Response(200,json=valid_body())))
    repository=Repository(); service.telemetry=repository
    with TestClient(create_app(service)) as client:
        response=client.post('/v1/inference',json={'model_id':'gateway-nano','prompt':'private'})
    assert response.status_code==200
    assert repository.events[0].request_id==response.json()['request_id']
    assert repository.events[0].estimated_cost_usd==Decimal('0.000003')


@pytest.mark.asyncio
async def test_cancellation_is_not_a_gateway_error():
    async def handler(request): raise asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await adapter(handler).generate(InferenceRequest(prompt='hi'))


def test_lifespan_registers_gateway_models_only_when_configured(monkeypatch):
    monkeypatch.setenv('AI_GATEWAY_API_KEY','test-only-key')
    with TestClient(create_app()) as client:
        models=client.get('/v1/models').json()['models']
    assert len(models)==9
    assert {m['model_id'] for m in models}>={'gateway-nano','gateway-mini','gateway-sonnet'}


def test_unsupported_gateway_model_fails_before_network():
    with pytest.raises(GatewayError) as error:
        VercelGatewayProvider(REAL_MODELS[0].model_copy(update={'provider_model_name':'unknown/model'}),GatewaySettings())
    assert error.value.category==Category.INVALID_MODEL

@pytest.mark.asyncio
async def test_empty_visible_response_is_failure_with_safe_reasoning_diagnostics_only():
    body = {
        'choices': [{'message': {
            'content': '',
            'reasoning_content': 'private hidden reasoning that must not persist',
        }, 'finish_reason': 'length'}],
        'usage': {
            'prompt_tokens': 21,
            'completion_tokens': 8,
            'completion_tokens_details': {'reasoning_tokens': 8},
        },
    }
    with pytest.raises(GatewayError) as error:
        await adapter(lambda request: httpx.Response(
            200, json=body, headers={'x-vercel-id': 'iad1::abc-123'})).generate(
                InferenceRequest(prompt='hi', max_output_tokens=8))
    assert error.value.category == Category.EMPTY_RESPONSE
    diagnostics = dict(error.value.diagnostics)
    latency = diagnostics.pop('adapter_latency_ms')
    assert isinstance(latency, float) and latency >= 0
    assert diagnostics == {
        'http_status': 200,
        'vercel_request_id': 'iad1::abc-123',
        'finish_reason': 'length',
        'content_empty': True,
        'output_token_limit': 8,
        'input_tokens': 21,
        'output_tokens': 8,
        'estimated_cost_usd': '0.0000053',
        'completion_tokens': 8,
        'reasoning_tokens': 8,
        'reasoning_field_present': True,
    }
    assert 'private hidden reasoning' not in repr(error.value.diagnostics)
    assert str(error.value) == 'gateway_empty_response'


@pytest.mark.asyncio
async def test_http_error_retains_only_allowlisted_safe_diagnostics():
    body = {'error': {
        'code': 'invalid_parameter',
        'type': 'invalid_request_error',
        'message': 'secret credential private prompt',
        'providerError': {'code': 'max_tokens_too_low', 'message': 'private'},
    }}
    with pytest.raises(GatewayError) as error:
        await adapter(lambda request: httpx.Response(400, json=body, headers={
            'x-vercel-id': 'iad1::trace-456', 'x-request-id': 'req_789',
        })).generate(InferenceRequest(prompt='private prompt'))
    assert error.value.category == Category.INVALID_REQUEST
    assert error.value.diagnostics == {
        'http_status': 400,
        'vercel_request_id': 'iad1::trace-456',
        'request_id': 'req_789',
        'gateway_error_code': 'invalid_parameter',
        'gateway_error_type': 'invalid_request_error',
        'upstream_error_code': 'max_tokens_too_low',
    }
    diagnostic_text = repr(error.value.diagnostics)
    assert 'secret' not in diagnostic_text and 'private prompt' not in diagnostic_text
