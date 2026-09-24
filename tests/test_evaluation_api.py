import json
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from adaptive_llm_gateway.api.app import create_app
from adaptive_llm_gateway.evaluation.models import EvaluationSummary, MetricGroup
from adaptive_llm_gateway.evaluation.service import EvaluationService


def summary(run_id):
    group = MetricGroup(evaluated_tasks=1, successful_responses=1,
        acceptable_responses=1, mean_quality_score=1, acceptable_rate=1,
        total_estimated_cost_usd="0.01", average_cost_per_task_usd="0.01",
        cost_per_acceptable_response_usd="0.01", average_latency_ms=10)
    return EvaluationSummary(run_id=run_id, dataset_sha256="abc",
        evaluation_version="1.0.0", overall=group, by_model=(),
        by_model_category=(), comparisons=())


def test_read_only_summary_endpoint_uses_uuid_not_paths(tmp_path):
    run_id = uuid4()
    directory = tmp_path / str(run_id)
    directory.mkdir()
    (directory / "evaluation-summary.json").write_text(summary(run_id).model_dump_json())
    application = create_app()
    application.state.evaluation_service = EvaluationService(tmp_path)
    with TestClient(application) as client:
        response = client.get(f"/v1/benchmarks/{run_id}/summary")
        traversal = client.get("/v1/benchmarks/..%2F.env/summary")
    assert response.status_code == 200
    assert response.json()["run_id"] == str(run_id)
    assert response.json()["overall"]["total_estimated_cost_usd"] == "0.01"
    assert traversal.status_code in {404, 422}


def test_missing_and_malformed_summary_errors_are_structured(tmp_path):
    application = create_app()
    application.state.evaluation_service = EvaluationService(tmp_path)
    run_id = uuid4()
    with TestClient(application) as client:
        missing = client.get(f"/v1/benchmarks/{run_id}/summary")
        directory = tmp_path / str(run_id)
        directory.mkdir()
        (directory / "evaluation-summary.json").write_text("not json")
        malformed = client.get(f"/v1/benchmarks/{run_id}/summary")
    assert missing.status_code == 404 and missing.json()["error"]["code"] == "evaluation_not_found"
    assert malformed.status_code == 422 and malformed.json()["error"]["code"] == "evaluation_artifact_invalid"
