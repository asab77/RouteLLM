import json
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import pytest

from adaptive_llm_gateway.benchmarks.models import BenchmarkDataset, BenchmarkResult, BenchmarkRun, BenchmarkTask
from adaptive_llm_gateway.evaluation.judge import (
    FakeJudge, JUDGE_PROMPT_VERSION, SemanticJudgeRequest, SemanticRubric, VercelSemanticJudge,
)
from adaptive_llm_gateway.evaluation.service import EvaluationService
from adaptive_llm_gateway.errors import ProviderFailureError
from adaptive_llm_gateway.models import InferenceResponse, ModelConfig


class RecordingProvider:
    def __init__(self, text: str, *, error: Exception | None = None):
        self.text = text
        self.error = error
        self.requests = []

    async def generate(self, request):
        self.requests.append(request)
        if self.error:
            raise self.error
        return InferenceResponse(text=self.text, model_id="explicit-judge", provider="vercel",
            input_tokens=20, output_tokens=8, latency_ms=12.5, estimated_cost_usd="0.0004")


def blind_request(candidate="A concise summary."):
    return SemanticJudgeRequest(source_text="Source facts only.", candidate_summary=candidate,
        semantic_requirements=("include the source fact",), output_constraints={"max_words": 8})


def test_semantic_judge_prompt_version_and_blind_schema_are_frozen():
    assert JUDGE_PROMPT_VERSION == "summary-rubric-1.0.0"
    assert set(blind_request().model_dump()) == {
        "source_text", "candidate_summary", "semantic_requirements", "output_constraints"}


@pytest.mark.asyncio
async def test_fake_judge_is_deterministic_and_receives_only_blind_fields():
    judge = FakeJudge(SemanticRubric(fact_coverage=0.5, factual_consistency=1,
        instruction_compliance=1, reason="One required fact was absent."))
    outcome = await judge.judge(blind_request())
    assert outcome.rubric.fact_coverage == 0.5
    assert set(judge.requests[0].model_dump()) == {
        "source_text", "candidate_summary", "semantic_requirements", "output_constraints"}
    serialized = judge.requests[0].model_dump_json()
    for forbidden in ("model_id", "price", "latency", "token", "difficulty", "previous"):
        assert forbidden not in serialized


@pytest.mark.asyncio
async def test_vercel_judge_uses_stable_strict_json_prompt_and_tracks_cost():
    provider = RecordingProvider(json.dumps({"fact_coverage": 1,
        "factual_consistency": 1, "instruction_compliance": 0.5,
        "reason": "The required fact is present, but the format is incomplete."}))
    judge = VercelSemanticJudge(provider, provider_name="vercel", model_id="explicit-judge")
    outcome = await judge.judge(blind_request())
    sent = provider.requests[0]
    assert sent.temperature == 0 and sent.system_prompt and JUDGE_PROMPT_VERSION in judge.configuration["prompt_version"]
    assert "explicit-judge" not in sent.prompt
    assert outcome.status == "evaluated" and outcome.usage.estimated_cost_usd == Decimal("0.0004")
    assert outcome.usage.input_tokens == 20 and outcome.usage.output_tokens == 8


@pytest.mark.asyncio
@pytest.mark.parametrize(("text", "category"), [
    ("not json", "malformed_json"),
    ('{"fact_coverage":2,"factual_consistency":1,"instruction_compliance":1,"reason":"bad"}', "invalid_rubric"),
    ('{"fact_coverage":1,"factual_consistency":1,"reason":"missing"}', "invalid_rubric"),
    ('{"fact_coverage":1,"factual_consistency":1,"instruction_compliance":1,"reason":"ok","extra":1}', "invalid_rubric"),
])
async def test_real_judge_rejects_malformed_or_out_of_range_results(text, category):
    outcome = await VercelSemanticJudge(RecordingProvider(text), provider_name="vercel",
                                        model_id="explicit-judge").judge(blind_request())
    assert outcome.status == "infrastructure_failure"
    assert outcome.error_category == category and outcome.rubric is None
    assert outcome.usage is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(("error", "category"), [
    (ProviderFailureError("upstream"), "provider_error"), (TimeoutError(), "timeout")])
async def test_real_judge_normalizes_provider_and_timeout_failures(error, category):
    outcome = await VercelSemanticJudge(RecordingProvider("", error=error),
        provider_name="vercel", model_id="explicit-judge").judge(blind_request())
    assert outcome.status == "infrastructure_failure" and outcome.error_category == category
    assert outcome.usage is None


def write_summary_run(root: Path, candidate: str) -> BenchmarkRun:
    task = BenchmarkTask(task_id="summarization-test", category="summarization",
        prompt="Summarize in at most 8 words: The bridge closes Friday.", max_output_tokens=16,
        temperature=0, evaluation_metadata={"semantic_requirements": ["bridge closes Friday"],
            "deterministic_constraints": {"max_words": 8}}, acceptable_threshold=0.8)
    model = ModelConfig(model_id="candidate-model", provider="fake", provider_model_name="fake",
        input_cost_per_1m_tokens="1", output_cost_per_1m_tokens="1", context_window=100)
    run = BenchmarkRun(dataset=BenchmarkDataset(name="fixture", version="2", tasks=(task,)),
        dataset_sha256="locked", selected_task_ids=(task.task_id,), models=(model,), configuration={})
    result = BenchmarkResult(run_id=run.run_id, request_id=str(uuid4()), task_id=task.task_id,
        model_id=model.model_id, success=True, latency_ms=5,
        response=InferenceResponse(text=candidate, model_id=model.model_id, provider="fake",
            input_tokens=4, output_tokens=4, latency_ms=5, estimated_cost_usd="0.000008"))
    directory = root / str(run.run_id)
    (directory / "results").mkdir(parents=True)
    (directory / "manifest.json").write_text(run.model_dump_json())
    (directory / "status.json").write_text('{"status":"completed"}')
    (directory / "results" / f"{result.result_id}.json").write_text(result.model_dump_json())
    return run


@pytest.mark.asyncio
async def test_summary_combination_applies_hallucination_veto_and_separate_judge_cost(tmp_path):
    run = write_summary_run(tmp_path, "The bridge closes Friday.")
    judge = FakeJudge(SemanticRubric(fact_coverage=1, factual_consistency=0.5,
        instruction_compliance=1, reason="The summary adds an unsupported implication."))
    summary = await EvaluationService(tmp_path, semantic_judge=judge).evaluate(run.run_id)
    assert summary.overall.fully_evaluated_tasks == 1
    assert summary.overall.mean_quality_score == 0
    assert summary.overall.acceptable_responses == 0
    assert summary.candidate_inference_cost_usd == Decimal("0.000008")
    assert summary.judge_calls == 1 and summary.judge_evaluation_cost_usd == 0
    assert judge.requests[0].candidate_summary == "The bridge closes Friday."


@pytest.mark.parametrize("arguments", [
    ["--semantic-judge-model", "gateway-mini"],
    ["--allow-paid-judge"],
    ["--semantic-judge-model", "gateway-mini", "--allow-paid-judge"],
    ["--summary-only", "--functional-docker"],
])
def test_cli_requires_explicit_paid_double_opt_in_and_keeps_summary_reads_passive(
        monkeypatch, arguments):
    from adaptive_llm_gateway.evaluation.__main__ import main

    monkeypatch.setattr("sys.argv", ["evaluation", "--run-id", str(uuid4()), *arguments])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
