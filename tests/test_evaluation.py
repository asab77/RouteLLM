import json
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from adaptive_llm_gateway.benchmarks.models import (
    BenchmarkDataset, BenchmarkResult, BenchmarkRun, BenchmarkTask,
)
from adaptive_llm_gateway.errors import EvaluationArtifactError, EvaluationNotFoundError
from adaptive_llm_gateway.evaluation.aggregation import aggregate
from adaptive_llm_gateway.evaluation.evaluators import EVALUATION_VERSION, evaluator_for
from adaptive_llm_gateway.evaluation.models import EvaluationResult
from adaptive_llm_gateway.evaluation.normalization import (
    count_sentences, final_answer, flatten_json, normalize_answer, parse_json_response,
)
from adaptive_llm_gateway.evaluation.repository import FileEvaluationRepository
from adaptive_llm_gateway.evaluation.service import EvaluationService
from adaptive_llm_gateway.models import InferenceResponse, ModelConfig


def task(category, metadata, *, threshold=None, output_type="text", task_id=None):
    return BenchmarkTask(
        task_id=task_id or f"{category}-test", category=category,
        prompt="Controlled prompt", system_prompt="Follow the format.",
        max_output_tokens=32, temperature=0, expected_output_type=output_type,
        evaluation_metadata=metadata, acceptable_threshold=threshold,
    )


def result(task, text="answer", *, model_id="model-a", success=True, cost="0.01",
           latency=10, error="provider_failure"):
    return BenchmarkResult(
        run_id=uuid4(), request_id=str(uuid4()), task_id=task.task_id,
        model_id=model_id, success=success,
        response=InferenceResponse(text=text, model_id=model_id, provider="fake",
            input_tokens=2, output_tokens=1, latency_ms=latency,
            estimated_cost_usd=cost) if success else None,
        error_category=None if success else error,
        error_details={} if success else {
            "input_tokens": 2, "output_tokens": 1,
            "adapter_latency_ms": 9.5, "estimated_cost_usd": "0.003",
        },
        latency_ms=latency,
    )


def evaluate(task, text="answer", **kwargs):
    return evaluator_for(task).evaluate(task, result(task, text, **kwargs))


@pytest.mark.parametrize(("raw", "expected"), [
    ("  Gold\n", "gold"), ("'Positive.'", "positive"),
    ("A-B", "a-b"), ("1.50", "1.50"), ("one, two", "one, two"),
])
def test_conservative_answer_normalization(raw, expected):
    assert normalize_answer(raw) == expected


def test_final_answer_and_json_helpers():
    assert final_answer("work\nFinal answer: 15.") == "15."
    assert parse_json_response('```json\n{"x": 1}\n```') == {"x": 1}
    assert flatten_json({"a": [1, {"b": 2}]}) == {"$.a[0]": 1, "$.a[1].b": 2}
    with pytest.raises(json.JSONDecodeError):
        parse_json_response("prefix {\"x\": 1}")


@pytest.mark.parametrize(("text", "expected"), [
    ("The library closes at 6 p.m. Friday for maintenance.", 1),
    ("The office opens at 8 a.m. Monday.", 1),
    ("Bring supplies, e.g. pens and paper.", 1),
    ("Bring supplies, e.g., pens and paper.", 1),
    ("Several items, i.e. pens and paper, are ready.", 1),
    ("Several items, i.e., pens and paper, are ready.", 1),
    ("Mr. Jones arrived.", 1),
    ("Mrs. Patel arrived.", 1),
    ("Dr. Rivera arrived.", 1),
    ("The U.S. office opens Friday.", 1),
    ("The measured value is 3.14 exactly.", 1),
    ("J. R. Smith arrived.", 1),
    ("First sentence. Second sentence.", 2),
    ("First line.\nSecond line.", 2),
    ("Is it ready? Yes.", 2),
    ("Stop! Continue.", 2),
    ('She said "Ready." Then left.', 2),
    ("", 0),
    ("The East Library closes at 6 p.m. Friday for electrical maintenance, while digital borrowing remains available and weekend hours remain unchanged.", 1),
    ("The office closes at 5 p.m. It reopens tomorrow.", 2),
    ("He works in the U.S. He travels often.", 2),
    ("He works in the U.S. The role requires travel.", 2),
])
def test_sentence_counting_handles_abbreviations_and_real_boundaries(text, expected):
    assert count_sentences(text) == expected


def test_live_good_summary_regression_passes_deterministic_sentence_constraint():
    item = task("summarization", {
        "semantic_requirements": ["library closes", "maintenance", "digital borrowing",
                                  "weekend hours unchanged"],
        "deterministic_constraints": {"max_sentences": 1, "max_words": 30},
    }, threshold=1)
    candidate = ("The East Library closes at 6 p.m. Friday for electrical maintenance, while "
                 "digital borrowing remains available and weekend hours remain unchanged.")
    evaluation = evaluate(item, candidate)
    assert evaluation.evaluation_status == "requires_semantic_judge"
    assert evaluation.component_scores["sentence_limit"] == 1
    assert evaluation.details["sentence_count"] == 1


def test_qa_only_accepts_explicit_representations():
    item = task("qa", {"accepted_answers": ["Au"]})
    assert evaluate(item, "AU.").acceptable
    rejected = evaluate(item, "gold")
    assert rejected.quality_score == 0 and not rejected.acceptable
    assert rejected.details["accepted_answers"] == ["Au"]


def test_classification_label_match_is_case_and_format_tolerant_only():
    item = task("classification", {"expected_label": "positive"})
    passed = evaluate(item, " Positive. ")
    assert passed.quality_score == 1 and passed.details["predicted"] == "positive"
    assert not evaluate(item, "mostly positive").acceptable


def test_reasoning_checks_final_answer_without_chain_of_thought_requirement():
    item = task("reasoning", {"accepted_answers": ["15"]})
    passed = evaluate(item, "I computed privately.\nFinal answer: 15")
    assert passed.acceptable and passed.evaluator_name == "reasoning_final_answer_match"
    assert not evaluate(item, "Final answer: 14").acceptable


def test_extraction_uses_field_level_f1_and_penalizes_extras():
    item = task("extraction", {"expected": {"name": "Mina", "age": 30}}, output_type="json")
    exact = evaluate(item, '{"name":"mina","age":30}')
    assert exact.quality_score == 1 and exact.acceptable
    partial = evaluate(item, '{"name":"Mina","extra":true}')
    assert partial.quality_score == pytest.approx(0.5)
    assert partial.component_scores["field_f1"] == pytest.approx(0.5)
    assert evaluate(item, "name: Mina").quality_score == 0


def test_json_separates_syntax_schema_and_values():
    item = task("json", {"expected": {"city": "Paris", "count": 2}}, output_type="json")
    exact = evaluate(item, '```json\n{"city":"PARIS","count":2}\n```')
    assert exact.component_scores == {"syntax": 1, "schema": 1, "values": 1}
    wrong = evaluate(item, '{"city":"Lima","count":"2"}')
    assert wrong.component_scores["syntax"] == 1
    assert wrong.component_scores["schema"] == pytest.approx(0.5)
    assert wrong.component_scores["values"] == 0
    assert not wrong.acceptable
    invalid = evaluate(item, "not json")
    assert invalid.component_scores == {"syntax": 0, "schema": 0, "values": 0}


def test_summarization_transparent_rubric_and_task_threshold():
    item = task("summarization", {"required_facts": [
        ["closed monday", "close monday"], ["reopens tuesday", "open tuesday"],
        ["online borrowing available"]], "forbidden_claims": ["closed forever"]}, threshold=0.8)
    full = evaluate(item, "The library will close Monday, reopen Tuesday, and keep online borrowing available.")
    assert full.quality_score == 1 and full.acceptable
    partial = evaluate(item, "The library will close Monday and reopen Tuesday.")
    assert partial.quality_score == pytest.approx(0.8 * (2 / 3) + 0.2)
    assert not partial.acceptable
    forbidden = evaluate(item, "The library is closed forever, but online borrowing available.")
    assert forbidden.quality_score == 0 and forbidden.details["forbidden_violations"] == ["closed forever"]


def test_coding_static_analysis_never_executes_generated_code(tmp_path):
    sentinel = tmp_path / "executed"
    item = task("coding", {"function_name": "is_even", "parameters": 1,
        "test_cases": [[2, True]], "required_ast": ["Mod", "Compare"]}, output_type="code")
    code = f'''```python\n# open({str(sentinel)!r}, "w").write("bad")\ndef is_even(n):\n    return n % 2 == 0\n```'''
    passed = evaluate(item, code)
    assert passed.quality_score == 1 and passed.acceptable
    assert passed.details["executed"] is False and not sentinel.exists()
    dangerous = evaluate(item, f'''open({str(sentinel)!r}, "w").write("bad")\ndef is_even(n):\n return n % 2 == 0''')
    assert dangerous.component_scores["safety"] == 0
    assert dangerous.details["forbidden_calls"] == ["open"] and not sentinel.exists()
    unsafe = evaluate(item, "import os\ndef is_even(n):\n return n % 2 == 0")
    assert unsafe.component_scores["safety"] == 0 and not unsafe.acceptable
    syntax = evaluate(item, "def is_even(")
    assert syntax.quality_score == 0


def test_threshold_controls_acceptability_without_changing_score():
    metadata = {"required_facts": ["one", "two"], "forbidden_claims": []}
    strict = evaluate(task("summarization", metadata, threshold=0.9), "one.")
    lenient = evaluate(task("summarization", metadata, threshold=0.6), "one.")
    assert strict.quality_score == lenient.quality_score == pytest.approx(0.6)
    assert not strict.acceptable and lenient.acceptable
    with pytest.raises(ValidationError):
        task("qa", {"accepted_answers": ["x"]}, threshold=1.1)


def test_failed_calls_receive_zero_without_invoking_response_evaluator():
    item = task("qa", {"accepted_answers": ["answer"]})
    evaluation = evaluate(item, success=False)
    assert evaluation.evaluator_name == "call_failure"
    assert evaluation.evaluation_status == "partial"
    assert evaluation.quality_score is None and evaluation.acceptable is None
    assert evaluation.details["error_category"] == "provider_failure"


def test_evaluator_selection_and_versioning():
    expected = {"qa": "accepted_answer_exact_match", "classification": "classification_label_match",
        "extraction": "structured_extraction_f1", "json": "json_structure_and_value",
        "reasoning": "reasoning_final_answer_match", "coding": "constrained_python_static_analysis",
        "summarization": "required_fact_coverage"}
    metadata = {"qa": {"accepted_answers": ["x"]}, "classification": {"expected_label": "x"},
        "extraction": {"expected": {}}, "json": {"expected": {}},
        "reasoning": {"accepted_answers": ["x"]},
        "coding": {"function_name": "f", "parameters": 0, "required_ast": []},
        "summarization": {"required_facts": ["x"], "forbidden_claims": []}}
    for category, name in expected.items():
        selected = evaluator_for(task(category, metadata[category]))
        assert selected.name == name and selected.version == EVALUATION_VERSION


def make_run(tmp_path, *, failed=False):
    controlled = task("qa", {"accepted_answers": ["Au"]}, task_id="qa-01")
    model_a = ModelConfig(model_id="model-a", provider="fake", provider_model_name="a",
        input_cost_per_1m_tokens="1", output_cost_per_1m_tokens="1", context_window=100)
    model_b = model_a.model_copy(update={"model_id": "model-b", "provider_model_name": "b"})
    run = BenchmarkRun(dataset=BenchmarkDataset(name="fixture", version="1", tasks=(controlled,)),
        dataset_sha256="abc", selected_task_ids=(controlled.task_id,), models=(model_a, model_b),
        configuration={})
    directory = tmp_path / str(run.run_id)
    (directory / "results").mkdir(parents=True)
    (directory / "manifest.json").write_text(run.model_dump_json(indent=2))
    (directory / "status.json").write_text(json.dumps({"status": "completed"}))
    first = result(controlled, "Au", model_id="model-a", cost="0.01", latency=10)
    second = result(controlled, "wrong", model_id="model-b", success=not failed, cost="0.03", latency=30)
    first = first.model_copy(update={"run_id": run.run_id})
    second = second.model_copy(update={"run_id": run.run_id})
    for item in (first, second):
        (directory / "results" / f"{item.result_id}.json").write_text(item.model_dump_json(indent=2))
    return run, first, second


@pytest.mark.asyncio
async def test_stored_artifact_evaluation_and_aggregation(tmp_path):
    run, first, second = make_run(tmp_path)
    summary = await EvaluationService(tmp_path).evaluate(run.run_id)
    assert summary.dataset_sha256 == "abc" and summary.evaluation_version == EVALUATION_VERSION
    assert summary.overall.evaluated_tasks == 2
    assert summary.overall.mean_quality_score == 0.5
    assert summary.overall.acceptable_rate == 0.5
    assert summary.overall.total_estimated_cost_usd == Decimal("0.04")
    assert summary.overall.average_cost_per_task_usd == Decimal("0.02")
    assert summary.overall.cost_per_acceptable_response_usd == Decimal("0.04")
    assert summary.overall.average_latency_ms == 20
    assert len(summary.by_model) == len(summary.by_model_category) == 2
    comparison, = summary.comparisons
    assert comparison.mean_quality_difference == 1
    assert comparison.total_cost_difference_usd == Decimal("-0.02")
    assert comparison.average_latency_difference_ms == -20
    directory = tmp_path / str(run.run_id)
    assert len(list((directory / "evaluations").glob("*.json"))) == 2
    assert (directory / "evaluation-summary.json").exists()
    assert await EvaluationService(tmp_path).summary(run.run_id) == summary


@pytest.mark.asyncio
async def test_failed_result_is_missing_label_with_unknown_cost(tmp_path):
    run, _, _ = make_run(tmp_path, failed=True)
    summary = await EvaluationService(tmp_path).evaluate(run.run_id)
    assert summary.overall.successful_responses == 1
    assert summary.overall.evaluated_tasks == 2
    assert summary.overall.fully_evaluated_tasks == 1
    assert summary.overall.incomplete_evaluations == 1
    assert summary.overall.total_estimated_cost_usd == Decimal("0.013")
    assert summary.overall.mean_quality_score == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["missing", "duplicate", "malformed", "wrong_run", "running"])
async def test_malformed_or_incomplete_artifacts_fail_explicitly(tmp_path, mutation):
    run, first, second = make_run(tmp_path)
    directory = tmp_path / str(run.run_id)
    paths = sorted((directory / "results").glob("*.json"))
    if mutation == "missing":
        paths[0].unlink()
    elif mutation == "duplicate":
        duplicate = second.model_copy(update={"result_id": uuid4(), "task_id": first.task_id, "model_id": first.model_id})
        (directory / "results" / f"{duplicate.result_id}.json").write_text(duplicate.model_dump_json())
    elif mutation == "malformed":
        paths[0].write_text("not json")
    elif mutation == "wrong_run":
        paths[0].write_text(first.model_copy(update={"run_id": uuid4()}).model_dump_json())
    else:
        (directory / "status.json").write_text(json.dumps({"status": "running"}))
    with pytest.raises(EvaluationArtifactError):
        await EvaluationService(tmp_path).evaluate(run.run_id)


@pytest.mark.asyncio
async def test_missing_run_and_summary_are_not_found(tmp_path):
    service = EvaluationService(tmp_path)
    with pytest.raises(EvaluationNotFoundError):
        await service.evaluate(uuid4())
    with pytest.raises(EvaluationNotFoundError):
        await service.summary(uuid4())


def test_cli_evaluates_and_reads_existing_summary_offline(monkeypatch, capsys, tmp_path):
    from adaptive_llm_gateway.evaluation.__main__ import main

    run, _, _ = make_run(tmp_path)
    monkeypatch.setattr("sys.argv", ["evaluation", "--root", str(tmp_path),
                        "--run-id", str(run.run_id)])
    main()
    evaluated = json.loads(capsys.readouterr().out)
    assert evaluated["run_id"] == str(run.run_id)
    monkeypatch.setattr("sys.argv", ["evaluation", "--root", str(tmp_path),
                        "--run-id", str(run.run_id), "--summary-only"])
    main()
    assert json.loads(capsys.readouterr().out)["overall"]["evaluated_tasks"] == 2


def test_evaluation_result_contract_rejects_inconsistent_pass():
    with pytest.raises(ValidationError):
        EvaluationResult(benchmark_result_id=uuid4(), run_id=uuid4(), task_id="t",
            model_id="m", evaluator_name="x", evaluator_version="1",
            quality_score=0.5, acceptable_threshold=0.8, acceptable=True, reason="bad")
