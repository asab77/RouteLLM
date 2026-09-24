import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from adaptive_llm_gateway.benchmarks.features import extract_request_features
from adaptive_llm_gateway.benchmarks.models import load_dataset
from adaptive_llm_gateway.benchmarks.repository import FileBenchmarkRepository
from adaptive_llm_gateway.benchmarks.routing_export import export_routing_dataset
from adaptive_llm_gateway.benchmarks.runner import BenchmarkRunner
from adaptive_llm_gateway.benchmarks.validation import (
    FOUNDATION_V3_CHANGED_TASK_IDS, validate_foundation_v3,
)
from adaptive_llm_gateway.application.service import InferenceService
from adaptive_llm_gateway.providers.fake import FakeProvider
from adaptive_llm_gateway.providers.gateway_config import CANDIDATE_MODELS
from adaptive_llm_gateway.providers.resolver import ProviderResolver
from adaptive_llm_gateway.registry import ModelRegistry
from adaptive_llm_gateway.evaluation.judge import FakeJudge
from adaptive_llm_gateway.evaluation.sandbox import FunctionalEvaluationResult
from adaptive_llm_gateway.evaluation.service import EvaluationService

V2 = Path("benchmarks/datasets/foundation-v2.json")
V3 = Path("benchmarks/datasets/foundation-v3.json")
PROTOCOL = Path("benchmarks/protocols/foundation-v3.json")
V2_SHA256 = "dcab5e9b6f1b8bc063555637e06db0d32ec51056959a7586f2cc411949adc964"
V3_SHA256 = "64eda8e7388233f645906412a2524982ebfb9c848c42ee24a468ba04c31d16e6"
PROTOCOL_SHA256 = "973c46e2dfd7c5739f623ff1c7f2378dc3e77c80e5815e23c975816700d4b861"


def protocol_data():
    return json.loads(PROTOCOL.read_text())


def test_foundation_v3_hash_shape_and_approved_changes_are_frozen():
    v2, v3 = load_dataset(V2), load_dataset(V3)
    assert hashlib.sha256(V2.read_bytes()).hexdigest() == v2.sha256 == V2_SHA256
    assert hashlib.sha256(V3.read_bytes()).hexdigest() == v3.sha256 == V3_SHA256
    validate_foundation_v3(v3, v2)
    assert len(FOUNDATION_V3_CHANGED_TASK_IDS) == 26
    assert Counter(task.category for task in v3.tasks) == Counter({
        "qa": 8, "classification": 8, "extraction": 8, "json": 8,
        "reasoning": 8, "coding": 8, "summarization": 8,
    })
    assert Counter(task.difficulty for task in v3.tasks) == Counter(
        {"easy": 14, "medium": 21, "hard": 21})


def test_foundation_v3_preserves_every_v2_task_field_except_approved_limits():
    v2 = {task.task_id: task for task in load_dataset(V2).tasks}
    v3 = {task.task_id: task for task in load_dataset(V3).tasks}
    assert set(v2) == set(v3)
    changed = set()
    for task_id in v2:
        assert v3[task_id].model_dump(exclude={"max_output_tokens"}) == \
            v2[task_id].model_dump(exclude={"max_output_tokens"})
        if v3[task_id].max_output_tokens != v2[task_id].max_output_tokens:
            changed.add(task_id)
            assert v3[task_id].max_output_tokens == 160
    assert changed == set(FOUNDATION_V3_CHANGED_TASK_IDS)


def test_foundation_v3_requests_and_features_do_not_leak_evaluation_metadata():
    forbidden = {"difficulty", "evaluation_metadata", "accepted_answers", "expected",
                 "functional_tests", "model_id", "acceptable"}
    for task in load_dataset(V3).tasks:
        request = task.to_request()
        assert set(request.model_dump()) == {
            "prompt", "system_prompt", "max_output_tokens", "temperature"}
        serialized = json.dumps(request.model_dump())
        assert "functional_tests" not in serialized
        assert task.difficulty not in request.model_dump()
        assert forbidden.isdisjoint(extract_request_features(task).model_dump())


def test_foundation_v3_protocol_freezes_collection_acceptance_criteria():
    protocol = protocol_data()
    assert hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() == PROTOCOL_SHA256
    assert Path("benchmarks/protocols/foundation-v3.sha256").read_text().split()[0] == PROTOCOL_SHA256
    assert protocol["dataset_sha256"] == V3_SHA256
    assert protocol["foundation_v2_sha256"] == V2_SHA256
    assert protocol["candidate_reasoning_effort"] == {
        "candidate-nemotron-3.5-lightning": "none",
        "candidate-gpt-6-luna": "low",
        "candidate-gemini-3-flash": "low",
        "candidate-claude-sonnet-5": "low",
    }
    assert protocol["acceptance_criteria"] == {
        "minimum_valid_labels_overall": 213,
        "minimum_valid_labels_per_candidate": 51,
        "configuration_failure_cluster_rejected_at": 3,
        "minimum_cross_model_label_disagreement_tasks": 10,
    }
    overrides = protocol["candidate_task_max_output_tokens"]
    assert set(overrides) == {"candidate-gemini-3-flash"}
    assert set(overrides["candidate-gemini-3-flash"]) == {
        "reasoning-easy-01", "reasoning-easy-02",
        "reasoning-medium-01", "reasoning-medium-02", "reasoning-medium-03",
        "reasoning-hard-01", "reasoning-hard-02", "reasoning-hard-03",
    }
    assert set(overrides["candidate-gemini-3-flash"].values()) == {256}


class RecordingFakeProvider:
    calls = []

    def __init__(self, model):
        self.model_id = model.model_id
        self.delegate = FakeProvider(model)

    async def generate(self, request):
        self.calls.append((self.model_id, request.max_output_tokens))
        return await self.delegate.generate(request)


class PassingFixtureSandbox:
    configuration = {"type": "deterministic-fixture", "evaluator_version": "1.0.0"}

    async def evaluate(self, request):
        total = len(request.tests)
        return FunctionalEvaluationResult(
            execution_status="passed", tests_attempted=total, tests_passed=total,
            tests_failed=0, total_tests=total, quality_score=1, acceptable=True)


@pytest.mark.asyncio
async def test_complete_v3_offline_pipeline_and_routing_export(tmp_path):
    dataset = load_dataset(V3)
    registry = ModelRegistry()
    for candidate in CANDIDATE_MODELS:
        registry.register(candidate.model_copy(update={
            "provider": "fake", "provider_model_name": "echo-v1",
            "context_window": 1_000_000,
        }))
    resolver = ProviderResolver()
    RecordingFakeProvider.calls = []
    resolver.register("fake", RecordingFakeProvider)
    service = InferenceService(registry, resolver)
    model_ids = [model.model_id for model in CANDIDATE_MODELS]
    overrides = protocol_data()["candidate_task_max_output_tokens"]
    run = await BenchmarkRunner(
        service, FileBenchmarkRepository(tmp_path),
        configuration={"execution_protocol_sha256": PROTOCOL_SHA256},
        output_token_overrides=overrides,
    ).run(dataset, model_ids)
    judge = FakeJudge()
    summary = await EvaluationService(
        tmp_path, functional_sandbox=PassingFixtureSandbox(), semantic_judge=judge
    ).evaluate(run.run_id)
    dataset_path, schema_path, rows = await export_routing_dataset(tmp_path, run.run_id)
    assert summary.overall.evaluated_tasks == summary.overall.fully_evaluated_tasks == 224
    assert summary.overall.incomplete_evaluations == 0
    assert summary.judge_calls == len(judge.requests) == 32
    assert len(rows) == 224 and all(row.label_status == "valid" for row in rows)
    assert len(RecordingFakeProvider.calls) == 224
    assert dataset_path.exists() and schema_path.exists()
    assert all(row.dataset_version == "3.0.0" for row in rows)
    assert all(row.dataset_sha256 == V3_SHA256 for row in rows)

    by_pair = {(row.task_id, row.candidate.internal_id): row for row in rows}
    reasoning_ids = set(overrides["candidate-gemini-3-flash"])
    gemini_overridden = [
        by_pair[(task_id, "candidate-gemini-3-flash")] for task_id in reasoning_ids]
    assert len(gemini_overridden) == 8
    assert all(row.effective_max_output_tokens == 256 for row in gemini_overridden)
    dataset_limits = {task.task_id: task.max_output_tokens for task in dataset.tasks}
    for task_id in dataset_limits.keys() - reasoning_ids:
        assert by_pair[(task_id, "candidate-gemini-3-flash")].effective_max_output_tokens == \
            dataset_limits[task_id]
    for task_id in reasoning_ids:
        assert by_pair[(task_id, "candidate-gpt-6-luna")].effective_max_output_tokens == 160
        assert by_pair[(task_id, "candidate-claude-sonnet-5")].effective_max_output_tokens == 160
        assert by_pair[(task_id, "candidate-nemotron-3.5-lightning")].effective_max_output_tokens == 160
    assert by_pair[("classification-medium-01",
                    "candidate-gemini-3-flash")].effective_max_output_tokens == 160
    v2_limits = {task.task_id: task.max_output_tokens for task in load_dataset(V2).tasks}
    assert by_pair[("classification-easy-01",
                    "candidate-gemini-3-flash")].effective_max_output_tokens == \
        v2_limits["classification-easy-01"]

    manifest = json.loads((tmp_path / str(run.run_id) / "manifest.json").read_text())
    effective = manifest["configuration"]["effective_max_output_tokens"]
    assert effective["reasoning-hard-01"]["candidate-gemini-3-flash"] == 256
    assert effective["reasoning-hard-01"]["candidate-claude-sonnet-5"] == 160
    assert manifest["configuration"]["execution_protocol_sha256"] == PROTOCOL_SHA256
