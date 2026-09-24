from collections import defaultdict
from decimal import Decimal, localcontext
from itertools import combinations

from adaptive_llm_gateway.benchmarks.models import BenchmarkResult, BenchmarkRun

from .evaluators import EVALUATION_VERSION
from .models import EvaluationResult, EvaluationSummary, MetricGroup, ModelComparison


def _known_candidate_cost(result: BenchmarkResult) -> Decimal:
    if result.response is not None:
        return result.response.estimated_cost_usd
    value = result.error_details.get("estimated_cost_usd")
    try:
        cost = Decimal(str(value))
    except Exception:
        return Decimal(0)
    return cost if cost.is_finite() and cost >= 0 else Decimal(0)


def _metric_group(items: list[tuple[BenchmarkResult, EvaluationResult]], *,
                  model_id: str | None = None, category: str | None = None) -> MetricGroup:
    count = len(items)
    fully_evaluated = [(result, evaluation) for result, evaluation in items
                       if evaluation.evaluation_status == "evaluated"]
    fully_count = len(fully_evaluated)
    successful = sum(result.success for result, _ in items)
    acceptable = sum(bool(evaluation.acceptable) for _, evaluation in fully_evaluated)
    total_cost = sum((_known_candidate_cost(result) for result, _ in items), Decimal(0))
    mean_quality = (sum(evaluation.quality_score or 0 for _, evaluation in fully_evaluated) / fully_count
                    if fully_count else None)
    average_latency = sum(result.latency_ms for result, _ in items) / count
    with localcontext() as context:
        context.prec = 50
        average_cost = total_cost / count
        cost_per_acceptable = total_cost / acceptable if acceptable else None
    return MetricGroup(
        model_id=model_id, category=category, evaluated_tasks=count,
        fully_evaluated_tasks=fully_count, incomplete_evaluations=count - fully_count,
        successful_responses=successful, acceptable_responses=acceptable,
        mean_quality_score=mean_quality, acceptable_rate=acceptable / fully_count if fully_count else None,
        total_estimated_cost_usd=total_cost, average_cost_per_task_usd=average_cost,
        cost_per_acceptable_response_usd=cost_per_acceptable,
        average_latency_ms=average_latency,
    )


def aggregate(run: BenchmarkRun, results: list[BenchmarkResult],
              evaluations: list[EvaluationResult], *,
              evaluation_configuration: dict | None = None) -> EvaluationSummary:
    if not results:
        raise ValueError("Cannot aggregate an empty benchmark run")
    evaluation_by_result = {item.benchmark_result_id: item for item in evaluations}
    if len(evaluation_by_result) != len(evaluations) or set(evaluation_by_result) != {item.result_id for item in results}:
        raise ValueError("Every benchmark result must have exactly one evaluation")
    task_categories = {task.task_id: task.category for task in run.dataset.tasks}
    paired = [(result, evaluation_by_result[result.result_id]) for result in results]
    by_model_items = defaultdict(list)
    by_category_items = defaultdict(list)
    for result, evaluation in paired:
        by_model_items[result.model_id].append((result, evaluation))
        by_category_items[(result.model_id, task_categories[result.task_id])].append((result, evaluation))
    by_model = tuple(_metric_group(items, model_id=model_id)
                     for model_id, items in sorted(by_model_items.items()))
    by_model_category = tuple(_metric_group(items, model_id=model_id, category=category)
                              for (model_id, category), items in sorted(by_category_items.items()))
    comparisons = tuple(ModelComparison(
        left_model_id=left.model_id or "", right_model_id=right.model_id or "",
        mean_quality_difference=(left.mean_quality_score - right.mean_quality_score
                                 if left.mean_quality_score is not None
                                 and right.mean_quality_score is not None else None),
        total_cost_difference_usd=left.total_estimated_cost_usd - right.total_estimated_cost_usd,
        average_latency_difference_ms=left.average_latency_ms - right.average_latency_ms,
    ) for left, right in combinations(by_model, 2))
    overall = _metric_group(paired)
    judge_evaluations = [item for item in evaluations if item.evaluation_call_made]
    return EvaluationSummary(
        run_id=run.run_id, dataset_sha256=run.dataset_sha256,
        evaluation_version=EVALUATION_VERSION, overall=overall,
        by_model=by_model, by_model_category=by_model_category, comparisons=comparisons,
        candidate_inference_cost_usd=overall.total_estimated_cost_usd,
        judge_evaluation_cost_usd=sum(
            (item.evaluation_cost_usd for item in judge_evaluations), Decimal(0)),
        judge_calls=len(judge_evaluations),
        judge_input_tokens=sum(item.evaluation_input_tokens for item in judge_evaluations),
        judge_output_tokens=sum(item.evaluation_output_tokens for item in judge_evaluations),
        judge_latency_ms=sum(item.evaluation_latency_ms for item in judge_evaluations),
        evaluation_configuration=evaluation_configuration or {},
    )
