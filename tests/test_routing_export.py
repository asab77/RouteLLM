import json
from pathlib import Path

import pytest

from adaptive_llm_gateway.benchmarks.models import BenchmarkDataset, load_dataset
from adaptive_llm_gateway.benchmarks.repository import FileBenchmarkRepository
from adaptive_llm_gateway.benchmarks.routing_export import export_routing_dataset
from adaptive_llm_gateway.benchmarks.runner import BenchmarkRunner
from adaptive_llm_gateway.bootstrap import create_development_service
from adaptive_llm_gateway.evaluation.service import EvaluationService

V2 = Path("benchmarks/datasets/foundation-v2.json")


@pytest.mark.asyncio
async def test_routing_export_keeps_features_labels_and_raw_text_separate(tmp_path):
    dataset = load_dataset(V2)
    run = await BenchmarkRunner(create_development_service(), FileBenchmarkRepository(tmp_path)).run(
        dataset, ["fake-small"], limit=1)
    await EvaluationService(tmp_path).evaluate(run.run_id)
    dataset_path, schema_path, rows = await export_routing_dataset(tmp_path, run.run_id)
    assert dataset_path.exists() and schema_path.exists() and len(rows) == 1
    row = rows[0]
    assert row.label_status == "valid" and row.acceptable in {True, False}
    assert row.request_features.category == dataset.tasks[0].category
    assert row.candidate.reasoning_effort is None
    assert row.effective_max_output_tokens == dataset.tasks[0].max_output_tokens
    assert row.raw_result_path.startswith("results/") and row.evaluation_path.startswith("evaluations/")
    serialized = dataset_path.read_text()
    assert dataset.tasks[0].prompt not in serialized
    forbidden = {"difficulty", "expected", "acceptable", "quality_score", "latency_ms"}
    assert forbidden.isdisjoint(row.request_features.model_dump())
    assert json.loads(schema_path.read_text())["primary_target"].startswith("acceptable")


@pytest.mark.asyncio
async def test_provider_failure_exports_missing_label_not_negative_label(tmp_path):
    original = load_dataset(V2)
    task = original.tasks[0].model_copy(update={"max_output_tokens": 5000})
    dataset = BenchmarkDataset(name="failure-fixture", version="1", tasks=(task,))
    run = await BenchmarkRunner(create_development_service(), FileBenchmarkRepository(tmp_path)).run(
        dataset, ["fake-small"])
    await EvaluationService(tmp_path).evaluate(run.run_id)
    _, _, rows = await export_routing_dataset(tmp_path, run.run_id)
    assert len(rows) == 1
    assert rows[0].provider_outcome == "failure"
    assert rows[0].label_status == "missing" and rows[0].acceptable is None
    assert rows[0].missing_label_reason == "provider_failure:context_limit_exceeded"
    assert rows[0].reasoning_tokens is None
