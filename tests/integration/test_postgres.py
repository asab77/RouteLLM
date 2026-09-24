"""Opt-in real PostgreSQL tests; each test owns a randomly named schema."""

import os
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
from uuid import uuid4

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from sqlalchemy import select, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from adaptive_llm_gateway.persistence.config import DatabaseSettings
from adaptive_llm_gateway.persistence.models import InferenceTelemetry
from adaptive_llm_gateway.persistence.repository import PostgresTelemetryRepository
from adaptive_llm_gateway.telemetry.contracts import TelemetryEvent

pytestmark = pytest.mark.postgres


@pytest_asyncio.fixture
async def database():
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("Set TEST_DATABASE_URL to a dedicated PostgreSQL test database")
    settings = DatabaseSettings(database_url=url)
    target = make_url(settings.database_url.get_secret_value())
    if target.database != "routellm_test" or target.username != "routellm_test":
        pytest.fail("Integration tests require database and role routellm_test; refusing unsafe target")
    schema = "test_telemetry_" + uuid4().hex
    admin = create_async_engine(settings.database_url.get_secret_value(), poolclass=NullPool)
    engine = None
    created = False
    try:
        async with admin.begin() as connection:
            await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
        created = True
        engine = create_async_engine(url, poolclass=NullPool,
            connect_args={"server_settings": {"search_path": schema}})

        def migrate(connection):
            config = Config("alembic.ini")
            config.attributes["connection"] = connection
            command.upgrade(config, "head")

        async with engine.begin() as connection:
            await connection.run_sync(migrate)
        yield engine, async_sessionmaker(engine, expire_on_commit=False)
    finally:
        if engine is not None:
            await engine.dispose()
        if created:
            async with admin.begin() as connection:
                await connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        await admin.dispose()


def sample():
    return TelemetryEvent(request_id="same-correlation-id", model_id="fake-small", provider="fake",
        success=True, input_tokens=3, output_tokens=2, latency_ms=10,
        estimated_cost_usd=Decimal("0.000001950000000000000000000001"),
        max_output_tokens=2, temperature=1, prompt_characters=20, system_prompt_characters=0)


@pytest.mark.asyncio
async def test_roundtrip_decimal_timestamp_commit_and_aggregate(database):
    engine, sessions = database
    repository = PostgresTelemetryRepository(sessions)
    assert (await repository.summary()).total_requests == 0
    assert (await repository.summary()).average_latency_ms is None
    event = sample()
    await repository.record(event)
    failure = replace(event, id=uuid4(), success=False, error_category="provider_failure",
                      input_tokens=None, output_tokens=None, estimated_cost_usd=None, latency_ms=30)
    await repository.record(failure)
    async with sessions() as session:
        rows = (await session.scalars(select(InferenceTelemetry))).all()
        assert len(rows) == 2
        row = next(row for row in rows if row.success)
        assert row.estimated_cost_usd == event.estimated_cost_usd
        assert isinstance(row.estimated_cost_usd, Decimal)
        assert row.request_id == event.request_id
        assert row.created_at == event.created_at
        assert row.created_at.tzinfo is not None
        assert row.created_at <= datetime.now(timezone.utc)
    summary = await repository.summary()
    assert summary.total_requests == 2
    assert summary.successful_requests == summary.failed_requests == 1
    assert summary.total_estimated_cost_usd == event.estimated_cost_usd
    assert summary.average_latency_ms == 20
    assert summary.total_input_tokens == 3 and summary.total_output_tokens == 2


@pytest.mark.asyncio
async def test_constraint_rollback_and_fresh_session_recovery(database):
    _, sessions = database
    repository = PostgresTelemetryRepository(sessions)
    event = sample()
    await repository.record(event)
    with pytest.raises(IntegrityError):
        await repository.record(event)  # duplicate database ID, not correlation ID
    with pytest.raises(IntegrityError):
        await repository.record(replace(event, id=uuid4(), input_tokens=-1))
    await repository.record(replace(event, id=uuid4()))
    assert (await repository.summary()).total_requests == 2


@pytest.mark.asyncio
async def test_migration_matches_metadata_and_can_downgrade_upgrade(database):
    engine, _ = database

    def verify(connection):
        config = Config("alembic.ini")
        config.attributes["connection"] = connection
        command.check(config)
        command.downgrade(config, "base")
        command.upgrade(config, "head")
        command.check(config)

    async with engine.begin() as connection:
        await connection.run_sync(verify)


@pytest.mark.asyncio
async def test_mock_gateway_telemetry_and_benchmark_storage_separation(database, tmp_path):
    import httpx
    from pydantic import SecretStr
    from adaptive_llm_gateway.bootstrap import create_development_service
    from adaptive_llm_gateway.benchmarks.models import load_dataset
    from adaptive_llm_gateway.benchmarks.repository import FileBenchmarkRepository
    from adaptive_llm_gateway.benchmarks.runner import BenchmarkRunner
    from adaptive_llm_gateway.evaluation.service import EvaluationService
    from adaptive_llm_gateway.models import InferenceRequest
    from adaptive_llm_gateway.providers.gateway_config import GatewaySettings, REAL_MODELS
    from adaptive_llm_gateway.providers.vercel import VercelGatewayProvider
    from pathlib import Path

    _, sessions = database
    repository = PostgresTelemetryRepository(sessions)
    service = create_development_service()
    service.telemetry = repository
    service.registry.register(REAL_MODELS[0])
    service.resolver.register('vercel', lambda model: VercelGatewayProvider(model,
        GatewaySettings(api_key=SecretStr('test-only-key')),
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={
            'choices':[{'message':{'content':'controlled answer'}}],
            'usage':{'prompt_tokens':10,'completion_tokens':5}}))))
    await service.generate('gateway-nano', InferenceRequest(prompt='controlled input'), request_id='gateway-db')
    before = await repository.summary()
    assert before.total_requests == 1
    assert before.total_estimated_cost_usd == Decimal('0.000003')
    run = await BenchmarkRunner(service, FileBenchmarkRepository(tmp_path)).run(
        load_dataset(Path('benchmarks/datasets/foundation-v1.json')), ['gateway-nano','fake-small'], limit=1)
    assert (await repository.summary()) == before
    assert len(list((tmp_path / str(run.run_id) / 'results').glob('*.json'))) == 2
    evaluation = await EvaluationService(tmp_path).evaluate(run.run_id)
    assert evaluation.overall.evaluated_tasks == 2
    assert (await repository.summary()) == before
    assert (tmp_path / str(run.run_id) / 'evaluation-summary.json').exists()
