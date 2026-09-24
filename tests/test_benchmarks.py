import json
from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from adaptive_llm_gateway.benchmarks.models import BenchmarkDataset, BenchmarkResult, load_dataset
from adaptive_llm_gateway.benchmarks.repository import FileBenchmarkRepository
from adaptive_llm_gateway.benchmarks.runner import BenchmarkRunner
from adaptive_llm_gateway.bootstrap import create_development_service
from adaptive_llm_gateway.providers.resolver import ProviderResolver

DATASET=Path('benchmarks/datasets/foundation-v1.json')


def test_dataset_categories_unique_ids_and_reproducibility():
    dataset=load_dataset(DATASET)
    assert len(dataset.tasks)==35
    assert Counter(t.category for t in dataset.tasks)==dict.fromkeys(['qa','summarization','extraction','classification','json','reasoning','coding'],5)
    assert dataset.sha256==load_dataset(DATASET).sha256
    assert len({t.task_id for t in dataset.tasks})==35
    assert dataset.tasks[0].to_request().temperature==0
    assert all(t.evaluation_metadata for t in dataset.tasks)
    assert dataset.version=='1.1.0'
    assert all(t.acceptable_threshold is not None for t in dataset.tasks)


@pytest.mark.parametrize('mutation',['duplicate','empty','blank_prompt'])
def test_invalid_dataset(mutation):
    data=load_dataset(DATASET).model_dump(mode='json')
    if mutation=='duplicate': data['tasks'].append(data['tasks'][0])
    elif mutation=='empty': data['tasks']=[]
    else: data['tasks'][0]['prompt']=' '
    with pytest.raises(ValidationError): BenchmarkDataset.model_validate(data)


@pytest.mark.asyncio
async def test_multi_model_results_snapshots_and_telemetry_separation(tmp_path):
    class ForbiddenProductionRepository:
        async def record(self,event): pytest.fail('Benchmark wrote production telemetry')
    service=create_development_service(); service.telemetry=ForbiddenProductionRepository()
    dataset=load_dataset(DATASET)
    run=await BenchmarkRunner(service,FileBenchmarkRepository(tmp_path)).run(dataset,['fake-small','fake-large'],limit=2)
    directory=tmp_path/str(run.run_id)
    manifest=json.loads((directory/'manifest.json').read_text())
    assert manifest['dataset_sha256']==dataset.sha256
    assert manifest['dataset']['version']=='1.1.0'
    assert len(manifest['models'])==2 and manifest['models'][0]['input_cost_per_1m_tokens']=='0.15'
    assert len(manifest['selected_task_ids'])==2
    frozen = manifest['configuration']['frozen_model_configuration']
    assert all(snapshot['reasoning_effort'] is None for snapshot in frozen.values())
    results=[BenchmarkResult.model_validate_json(p.read_bytes()) for p in (directory/'results').glob('*.json')]
    assert len(results)==4
    for task in dataset.tasks[:2]:
        pair=[r for r in results if r.task_id==task.task_id]
        assert {r.model_id for r in pair}=={'fake-small','fake-large'}
        assert pair[0].response.text==pair[1].response.text
        assert pair[0].response.estimated_cost_usd!=pair[1].response.estimated_cost_usd
    assert all(r.success and r.response.input_tokens>0 for r in results)
    assert json.loads((directory/'status.json').read_text())['status']=='completed'
    assert not list(directory.rglob('*.tmp'))


@pytest.mark.asyncio
async def test_per_call_failure_retained_and_other_models_continue(tmp_path):
    service=create_development_service()
    dataset=load_dataset(DATASET)
    tasks=[dataset.tasks[0].model_copy(update={'max_output_tokens':5000})]
    dataset=BenchmarkDataset(name='limits',version='1',tasks=tasks)
    run=await BenchmarkRunner(service,FileBenchmarkRepository(tmp_path)).run(dataset,['fake-small','fake-large'])
    results=[json.loads(p.read_text()) for p in (tmp_path/str(run.run_id)/'results').glob('*.json')]
    assert len(results)==2 and sum(r['success'] for r in results)==1
    failure=next(r for r in results if not r['success'])
    assert failure['error_category']=='context_limit_exceeded' and failure['response'] is None


@pytest.mark.asyncio
@pytest.mark.parametrize('models,limit',[([],1),(['fake-small','fake-small'],1),(['missing'],1),(['fake-small'],0)])
async def test_invalid_selection_creates_no_artifacts(tmp_path,models,limit):
    with pytest.raises((ValueError,KeyError)):
        await BenchmarkRunner(create_development_service(),FileBenchmarkRepository(tmp_path)).run(load_dataset(DATASET),models,limit=limit)
    assert not list(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_storage_failure_aborts_before_further_calls(tmp_path):
    class BrokenRepository(FileBenchmarkRepository):
        calls=0
        async def record(self,result):
            self.calls+=1
            raise OSError('disk full')
    repo=BrokenRepository(tmp_path)
    with pytest.raises(OSError):
        await BenchmarkRunner(create_development_service(),repo).run(load_dataset(DATASET),['fake-small','fake-large'])
    assert repo.calls==1
    status,=tmp_path.glob('*/status.json')
    assert json.loads(status.read_text())['status']=='aborted'


@pytest.mark.asyncio
async def test_run_id_cannot_overwrite_existing_experiment(tmp_path):
    repo=FileBenchmarkRepository(tmp_path)
    run=await BenchmarkRunner(create_development_service(),repo).run(load_dataset(DATASET),['fake-small'],limit=1)
    with pytest.raises(FileExistsError): await repo.start(run)
    assert len(list((tmp_path/str(run.run_id)/'results').glob('*.json')))==1


def test_cli_requires_explicit_paid_opt_in(monkeypatch, capsys, tmp_path):
    from adaptive_llm_gateway.benchmarks.__main__ import main
    monkeypatch.setenv('AI_GATEWAY_API_KEY','test-only-key')
    monkeypatch.setattr('sys.argv',['benchmark','--models','gateway-nano','--output',str(tmp_path)])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code==2
    assert '--allow-paid' in capsys.readouterr().err
    assert not list(tmp_path.iterdir())

@pytest.mark.asyncio
async def test_benchmark_persists_sanitized_gateway_diagnostics(tmp_path):
    from adaptive_llm_gateway.errors import GatewayError, GatewayErrorCategory
    from adaptive_llm_gateway.models import ModelConfig

    class DiagnosticProvider:
        async def generate(self, request):
            raise GatewayError(GatewayErrorCategory.INVALID_REQUEST, diagnostics={
                'http_status': 400, 'gateway_error_code': 'invalid_parameter'})

    service = create_development_service()
    model = ModelConfig(model_id='diagnostic-model', provider='diagnostic',
        provider_model_name='diagnostic/model', input_cost_per_1m_tokens='1',
        output_cost_per_1m_tokens='1', context_window=4096)
    service.registry.register(model)
    service.resolver.register('diagnostic', lambda model: DiagnosticProvider())
    run = await BenchmarkRunner(service, FileBenchmarkRepository(tmp_path)).run(
        load_dataset(DATASET), ['diagnostic-model'], limit=1)
    result_path, = (tmp_path / str(run.run_id) / 'results').glob('*.json')
    result = BenchmarkResult.model_validate_json(result_path.read_bytes())
    assert not result.success and result.error_category == 'gateway_invalid_request'
    assert result.error_details == {'http_status': 400, 'gateway_error_code': 'invalid_parameter'}
