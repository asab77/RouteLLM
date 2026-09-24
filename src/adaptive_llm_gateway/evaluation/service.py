from pathlib import Path
from uuid import UUID

from adaptive_llm_gateway.errors import EvaluationArtifactError

from .aggregation import aggregate
from .evaluators import evaluator_for
from .judge import SEMANTIC_JUDGE_VERSION, SemanticJudge, SemanticJudgeRequest
from .models import EvaluationResult, EvaluationSummary
from .normalization import strip_code_fence
from .repository import FileEvaluationRepository
from .sandbox import FunctionalEvaluationRequest, FunctionalSandbox, FunctionalTestCase


class EvaluationService:
    def __init__(self, root: Path = Path("benchmark-results"), *,
                 functional_sandbox: FunctionalSandbox | None = None,
                 semantic_judge: SemanticJudge | None = None) -> None:
        self.repository = FileEvaluationRepository(root)
        self.functional_sandbox = functional_sandbox
        self.semantic_judge = semantic_judge

    @property
    def configuration(self) -> dict:
        configuration = {"functional_sandbox": None, "semantic_judge": None}
        if self.functional_sandbox is not None:
            configuration["functional_sandbox"] = getattr(
                self.functional_sandbox, "configuration", {"type": type(self.functional_sandbox).__name__})
        if self.semantic_judge is not None:
            configuration["semantic_judge"] = getattr(
                self.semantic_judge, "configuration", {"type": type(self.semantic_judge).__name__})
        return configuration

    async def evaluate(self, run_id: UUID) -> EvaluationSummary:
        run, results = await self.repository.load_run(run_id)
        task_by_id = {task.task_id: task for task in run.dataset.tasks}
        evaluations: list[EvaluationResult] = []
        for result in results:
            try:
                task = task_by_id[result.task_id]
            except KeyError:
                raise EvaluationArtifactError(
                    f"Result references unknown task {result.task_id!r}") from None
            try:
                evaluation = evaluator_for(task).evaluate(task, result)
                if (evaluation.evaluation_status == "requires_functional_execution"
                        and self.functional_sandbox is not None and result.response is not None):
                    evaluation = await self._evaluate_functionally(task, result, evaluation)
                if (evaluation.evaluation_status == "requires_semantic_judge"
                        and self.semantic_judge is not None and result.response is not None):
                    evaluation = await self._evaluate_semantically(task, result, evaluation)
                evaluations.append(evaluation)
            except (KeyError, TypeError, ValueError) as exc:
                raise EvaluationArtifactError(
                    f"Task {task.task_id!r} has invalid evaluation metadata") from exc
        try:
            summary = aggregate(run, results, evaluations,
                                evaluation_configuration=self.configuration)
        except (KeyError, TypeError, ValueError) as exc:
            raise EvaluationArtifactError("Benchmark evaluations could not be aggregated") from exc
        await self.repository.save(run_id, evaluations, summary)
        return summary

    async def summary(self, run_id: UUID) -> EvaluationSummary:
        return await self.repository.load_summary(run_id)

    async def _evaluate_functionally(self, task, result, static: EvaluationResult) -> EvaluationResult:
        metadata = task.evaluation_metadata
        tests = metadata.get("functional_tests")
        if not isinstance(tests, list):
            raise ValueError("Functional tests must be a list")
        request = FunctionalEvaluationRequest(
            source=strip_code_fence(result.response.text),
            function_name=metadata["function_name"], parameter_count=metadata["parameters"],
            tests=tuple(FunctionalTestCase.model_validate(case) for case in tests),
            preserve_inputs="mutat" in task.prompt.casefold(),
        )
        outcome = await self.functional_sandbox.evaluate(request)
        details = {"static_validation": static.model_dump(mode="json"),
                   "functional_execution": outcome.model_dump(mode="json")}
        if outcome.execution_status == "infrastructure_failure":
            return static.model_copy(update={
                "reason": "Functional evaluation infrastructure failed; candidate quality was not scored.",
                "details": details,
            })
        return EvaluationResult(
            benchmark_result_id=result.result_id, run_id=result.run_id,
            task_id=task.task_id, model_id=result.model_id,
            evaluator_name="docker_python_functional",
            evaluator_version=outcome.evaluator_version,
            quality_score=outcome.quality_score, acceptable_threshold=1,
            acceptable=outcome.acceptable,
            reason=(f"Passed {outcome.tests_passed} of {outcome.total_tests} hidden functional tests."
                    if outcome.execution_status == "passed" else
                    f"Functional evaluation failed: {outcome.error_category}."),
            component_scores={"functional_tests": outcome.quality_score or 0},
            details=details,
        )

    async def _evaluate_semantically(self, task, result,
                                     deterministic: EvaluationResult) -> EvaluationResult:
        metadata = task.evaluation_metadata
        request = SemanticJudgeRequest(
            source_text=task.prompt, candidate_summary=result.response.text,
            semantic_requirements=tuple(metadata["semantic_requirements"]),
            output_constraints=metadata.get("deterministic_constraints", {}),
        )
        outcome = await self.semantic_judge.judge(request)
        usage = outcome.usage
        usage_fields = {
            "evaluation_call_made": True,
            "evaluation_input_tokens": usage.input_tokens if usage else 0,
            "evaluation_output_tokens": usage.output_tokens if usage else 0,
            "evaluation_latency_ms": usage.latency_ms if usage else 0,
            "evaluation_cost_usd": usage.estimated_cost_usd if usage else 0,
        }
        details = {"deterministic": deterministic.model_dump(mode="json"),
                   "semantic_judge": outcome.model_dump(mode="json")}
        if outcome.status == "infrastructure_failure" or outcome.rubric is None:
            return deterministic.model_copy(update={
                "reason": "Semantic judge infrastructure failed; candidate quality was not scored.",
                "details": details, **usage_fields,
            })
        rubric = outcome.rubric
        threshold = task.acceptable_threshold if task.acceptable_threshold is not None else 0.8
        deterministic_passed = all(score == 1 for score in deterministic.component_scores.values())
        semantic_score = (rubric.fact_coverage + rubric.factual_consistency
                          + rubric.instruction_compliance) / 3
        vetoed = not deterministic_passed or rubric.factual_consistency < 1
        quality = 0.0 if vetoed else semantic_score
        components = {**{f"deterministic_{key}": value
                         for key, value in deterministic.component_scores.items()},
                      "fact_coverage": rubric.fact_coverage,
                      "factual_consistency": rubric.factual_consistency,
                      "instruction_compliance": rubric.instruction_compliance}
        return EvaluationResult(
            benchmark_result_id=result.result_id, run_id=result.run_id,
            task_id=task.task_id, model_id=result.model_id,
            evaluator_name="semantic_summary_judge",
            evaluator_version=SEMANTIC_JUDGE_VERSION,
            quality_score=quality, acceptable_threshold=threshold,
            acceptable=quality >= threshold,
            reason=rubric.reason, component_scores=components, details=details,
            **usage_fields,
        )
