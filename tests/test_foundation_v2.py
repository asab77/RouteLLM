import hashlib
import json
from collections import Counter
from decimal import Decimal
from heapq import heappop, heappush
from itertools import permutations
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from adaptive_llm_gateway.benchmarks.features import extract_request_features
from adaptive_llm_gateway.benchmarks.models import (
    BenchmarkDataset, BenchmarkResult, BenchmarkTask, load_dataset,
)
from adaptive_llm_gateway.benchmarks.repository import FileBenchmarkRepository
from adaptive_llm_gateway.benchmarks.runner import BenchmarkRunner
from adaptive_llm_gateway.benchmarks.validation import validate_foundation_v2
from adaptive_llm_gateway.bootstrap import create_development_service
from adaptive_llm_gateway.evaluation.evaluators import evaluator_for
from adaptive_llm_gateway.evaluation.judge import FakeJudge
from adaptive_llm_gateway.evaluation.models import EvaluationResult
from adaptive_llm_gateway.evaluation.sandbox import FunctionalEvaluationResult
from adaptive_llm_gateway.evaluation.service import EvaluationService
from adaptive_llm_gateway.models import InferenceResponse

V1 = Path("benchmarks/datasets/foundation-v1.json")
V2 = Path("benchmarks/datasets/foundation-v2.json")


def test_foundation_v2_file_hash_is_frozen():
    expected_hash = "dcab5e9b6f1b8bc063555637e06db0d32ec51056959a7586f2cc411949adc964"
    assert hashlib.sha256(V2.read_bytes()).hexdigest() == expected_hash
    assert load_dataset(V2).sha256 == expected_hash


def response_for(task: BenchmarkTask, text: str) -> BenchmarkResult:
    return BenchmarkResult(
        run_id=uuid4(), request_id=str(uuid4()), task_id=task.task_id,
        model_id="fixture", success=True, latency_ms=1,
        response=InferenceResponse(text=text, model_id="fixture", provider="fake",
            input_tokens=1, output_tokens=1, latency_ms=1, estimated_cost_usd="0"),
    )


def test_v2_exact_shape_ids_and_difficulty_distribution():
    dataset = load_dataset(V2)
    validate_foundation_v2(dataset)
    assert len(dataset.tasks) == len({task.task_id for task in dataset.tasks}) == 56
    assert Counter(task.category for task in dataset.tasks) == Counter({
        "qa": 8, "classification": 8, "extraction": 8, "json": 8,
        "reasoning": 8, "coding": 8, "summarization": 8})
    assert Counter(task.difficulty for task in dataset.tasks) == Counter(
        {"easy": 14, "medium": 21, "hard": 21})


def test_v2_all_tasks_have_valid_evaluator_and_ground_truth():
    dataset = load_dataset(V2)
    for task in dataset.tasks:
        assert evaluator_for(task).category == task.category
        assert task.acceptable_threshold == 1
        assert task.evaluation_metadata


def test_reasoning_ground_truth_examples_are_programmatically_confirmed():
    tasks = {task.task_id: task for task in load_dataset(V2).tasks}
    assert tasks["reasoning-easy-01"].evaluation_metadata["accepted_answers"] == [str(18 - 6 + 4)]
    assert tasks["reasoning-easy-02"].evaluation_metadata["accepted_answers"] == ["15:15"]
    position = 5 + 4
    position = position - 2 if position % 2 else position + 2
    assert tasks["reasoning-medium-01"].evaluation_metadata["accepted_answers"] == [
        str((position * 3) % 12)]
    total = Decimal(3 * 18 + 2 * 7) - 10
    total = total * Decimal("1.05") + 5
    assert tasks["reasoning-medium-03"].evaluation_metadata["accepted_answers"] == [
        f"{total:.2f}", f"${total:.2f}"]

    medium_seating = [order for order in permutations("ABCDE")
        if order[2] == "C" and order.index("B") == order.index("A") + 1
        and order.index("E") > order.index("B") and order.index("D") not in {0, 4}]
    assert medium_seating == [tuple("ABCDE")]
    assert tasks["reasoning-medium-02"].evaluation_metadata["accepted_answers"] == ["A"]

    hard_order = [order for order in permutations("ABCDE")
        if order.index("B") == order.index("A") + 1
        and order.index("D") < order.index("A")
        and order.index("E") > order.index("B")
        and order.index("C") > order.index("E")]
    assert hard_order == [tuple("DABEC")]
    assert tasks["reasoning-hard-01"].evaluation_metadata["accepted_answers"] == ["DABEC"]

    codes = []
    for last in range(2, 10, 2):
        digits = (last + 1, last + 3, last // 2, last)
        if max(digits) <= 9 and len(set(digits)) == 4 and sum(digits) == 18:
            codes.append("".join(map(str, digits)))
    assert codes == ["5724"]
    assert tasks["reasoning-hard-02"].evaluation_metadata["accepted_answers"] == codes

    graph = {"P": [("Q", 2), ("R", 1)], "Q": [("S", 2), ("T", 7)],
             "R": [("S", 5), ("U", 1)], "U": [("S", 1), ("T", 6)],
             "S": [("T", 2)], "T": [("P", 0)]}
    queue, best = [(0, "P")], {"P": 0}
    while queue:
        cost, node = heappop(queue)
        for neighbor, edge_cost in graph[node]:
            candidate = cost + edge_cost
            if candidate < best.get(neighbor, 10**9):
                best[neighbor] = candidate
                heappush(queue, (candidate, neighbor))
    assert best["T"] == 5
    assert tasks["reasoning-hard-03"].evaluation_metadata["accepted_answers"] == ["5"]


def test_qa_multistep_ground_truth_is_programmatically_confirmed():
    tasks = {task.task_id: task for task in load_dataset(V2).tasks}
    assert tasks["qa-medium-01"].evaluation_metadata["accepted_answers"] == [
        str(18 + 12 - 3 - 4 - 2)]
    mentorship = {"Fae": "Dee", "Dee": "Cy", "Cy": "Bo", "Bo": "Ada"}
    mentor = "Fae"
    for _ in range(3):
        mentor = mentorship[mentor]
    assert mentor == "Bo"
    assert tasks["qa-hard-01"].evaluation_metadata["accepted_answers"] == [mentor]
    route = ["Luma", "Neri", "Ossa", "Pavo", "Quin"]
    route.remove("Neri")
    route.remove("Pavo")
    route.insert(route.index("Ossa"), "Pavo")
    route.insert(route.index("Ossa") + 1, "Reka")
    route[0], route[-1] = route[-1], route[0]
    assert route == ["Quin", "Pavo", "Ossa", "Reka", "Luma"]
    grant_total = 1220
    training = (grant_total - 120 - 60) // 4
    hardware = 2 * training
    assert tasks["qa-hard-03"].evaluation_metadata["accepted_answers"] == [
        f"${hardware}", str(hardware)]


def test_v2_coding_metadata_is_future_functional_fixture_only():
    coding = [task for task in load_dataset(V2).tasks if task.category == "coding"]
    assert len(coding) == 8
    for task in coding:
        metadata = task.evaluation_metadata
        assert metadata["evaluation_mode"] == "static_pending_functional"
        assert metadata["functional_tests"]
        assert all("args" in case and "expected" in case for case in metadata["functional_tests"])


def test_v2_summaries_keep_semantics_out_of_deterministic_score():
    summary = next(task for task in load_dataset(V2).tasks if task.category == "summarization")
    evaluation = evaluator_for(summary).evaluate(
        summary, response_for(summary, "A completely unrelated but short sentence."))
    assert evaluation.evaluation_status == "requires_semantic_judge"
    assert evaluation.acceptable is None
    assert evaluation.details["semantic_requirements_used_for_scoring"] is False
    assert summary.evaluation_metadata["semantic_requirements"]


def test_v2_coding_is_not_claimed_functionally_acceptable():
    coding = next(task for task in load_dataset(V2).tasks if task.task_id == "coding-easy-01")
    evaluation = evaluator_for(coding).evaluate(
        coding, response_for(coding, "def clamp(value, low, high):\n    if value < low: return low\n    return high if value > high else value"))
    assert evaluation.evaluation_status == "requires_functional_execution"
    assert evaluation.acceptable is None
    assert evaluation.details["executed"] is False


def test_request_construction_and_features_cannot_leak_evaluation_data():
    task = next(task for task in load_dataset(V2).tasks if task.category == "coding")
    request = task.to_request()
    assert set(request.model_dump()) == {"prompt", "system_prompt", "max_output_tokens", "temperature"}
    serialized = json.dumps(request.model_dump())
    assert "functional_tests" not in serialized and task.difficulty not in serialized
    features = extract_request_features(task)
    forbidden = {"difficulty", "evaluation_metadata", "accepted_answers", "expected",
                 "functional_tests", "model_id", "acceptable"}
    assert forbidden.isdisjoint(features.model_dump())
    assert features.contains_code and features.requests_structured_output
    assert features.category == "coding"


def test_feature_extraction_is_explainable_and_provider_independent():
    task = next(task for task in load_dataset(V2).tasks if task.task_id == "json-hard-01")
    features = extract_request_features(task)
    assert features.prompt_characters == len(task.prompt)
    assert features.approximate_input_tokens > 0
    assert features.requests_structured_output
    assert features.max_output_tokens == task.max_output_tokens
    assert features.constraint_indicator_count > 0


def test_v2_validator_rejects_shape_and_missing_ground_truth():
    dataset = load_dataset(V2)
    with pytest.raises(ValueError):
        validate_foundation_v2(dataset.model_copy(update={"tasks": dataset.tasks[:-1]}))
    broken = dataset.tasks[0].model_copy(update={"evaluation_metadata": {}})
    tasks = (broken,) + dataset.tasks[1:]
    with pytest.raises(ValueError):
        validate_foundation_v2(dataset.model_copy(update={"tasks": tasks}))


@pytest.mark.asyncio
async def test_full_v2_offline_run_persists_and_excludes_incomplete_from_failures(tmp_path):
    dataset = load_dataset(V2)
    run = await BenchmarkRunner(
        create_development_service(), FileBenchmarkRepository(tmp_path)
    ).run(dataset, ["fake-small"])
    summary = await EvaluationService(tmp_path).evaluate(run.run_id)
    assert summary.overall.evaluated_tasks == 56
    assert summary.overall.fully_evaluated_tasks == 40
    assert summary.overall.incomplete_evaluations == 16
    coding_group = next(group for group in summary.by_model_category if group.category == "coding")
    summary_group = next(group for group in summary.by_model_category if group.category == "summarization")
    assert coding_group.acceptable_rate is coding_group.mean_quality_score is None
    assert summary_group.acceptable_rate is summary_group.mean_quality_score is None
    evaluations = [
        EvaluationResult.model_validate_json(path.read_bytes())
        for path in (tmp_path / str(run.run_id) / "evaluations").glob("*.json")
    ]
    assert Counter(item.evaluation_status for item in evaluations) == Counter({
        "evaluated": 40, "requires_functional_execution": 8,
        "requires_semantic_judge": 8})
    assert (tmp_path / str(run.run_id) / "evaluation-summary.json").exists()


def test_v1_remains_unchanged_and_legacy_evaluation_defaults_to_evaluated():
    dataset = load_dataset(V1)
    assert dataset.version == "1.1.0" and all(task.difficulty is None for task in dataset.tasks)
    legacy = {
        "benchmark_result_id": str(uuid4()), "run_id": str(uuid4()), "task_id": "qa-01",
        "model_id": "old", "evaluator_name": "accepted_answer_exact_match",
        "evaluator_version": "1.0.0", "quality_score": 1, "acceptable_threshold": 1,
        "acceptable": True, "reason": "legacy"
    }
    assert EvaluationResult.model_validate(legacy).evaluation_status == "evaluated"


class PassingFixtureSandbox:
    configuration = {"type": "deterministic-fixture", "evaluator_version": "1.0.0"}

    async def evaluate(self, request):
        total = len(request.tests)
        return FunctionalEvaluationResult(execution_status="passed", tests_attempted=total,
            tests_passed=total, tests_failed=0, total_tests=total, quality_score=1,
            acceptable=True)


@pytest.mark.asyncio
async def test_complete_v2_fake_provider_fake_judge_and_sandbox_fixture_flow(tmp_path):
    dataset = load_dataset(V2)
    run = await BenchmarkRunner(
        create_development_service(), FileBenchmarkRepository(tmp_path)
    ).run(dataset, ["fake-small"])
    judge = FakeJudge()
    summary = await EvaluationService(tmp_path, functional_sandbox=PassingFixtureSandbox(),
        semantic_judge=judge).evaluate(run.run_id)
    assert summary.overall.evaluated_tasks == summary.overall.fully_evaluated_tasks == 56
    assert summary.overall.incomplete_evaluations == 0
    assert summary.judge_calls == 8 and summary.judge_evaluation_cost_usd == 0
    assert summary.candidate_inference_cost_usd == summary.overall.total_estimated_cost_usd
    assert len(judge.requests) == 8
    assert all(set(request.model_dump()) == {"source_text", "candidate_summary",
        "semantic_requirements", "output_constraints"} for request in judge.requests)
    evaluations = [EvaluationResult.model_validate_json(path.read_bytes())
        for path in (tmp_path / str(run.run_id) / "evaluations").glob("*.json")]
    assert all(item.evaluation_status == "evaluated" for item in evaluations)
