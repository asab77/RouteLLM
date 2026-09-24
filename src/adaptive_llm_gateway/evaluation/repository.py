import asyncio
import json
import os
from pathlib import Path
from uuid import UUID, uuid4

from pydantic import ValidationError

from adaptive_llm_gateway.benchmarks.models import BenchmarkResult, BenchmarkRun
from adaptive_llm_gateway.errors import EvaluationArtifactError, EvaluationNotFoundError

from .models import EvaluationResult, EvaluationSummary


class FileEvaluationRepository:
    """Read validated benchmark runs and atomically write derived evaluations."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _directory(self, run_id: UUID) -> Path:
        return self.root / str(run_id)

    @staticmethod
    def _write(path: Path, data: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
        try:
            with temporary.open("x", encoding="utf-8") as output:
                output.write(data)
                output.flush()
                os.fsync(output.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def _load_run(self, run_id: UUID) -> tuple[BenchmarkRun, list[BenchmarkResult]]:
        directory = self._directory(run_id)
        try:
            run = BenchmarkRun.model_validate_json((directory / "manifest.json").read_bytes())
            status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise EvaluationNotFoundError(f"Benchmark run {run_id} was not found") from None
        except (ValidationError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise EvaluationArtifactError("Benchmark manifest or status is malformed") from exc
        if not isinstance(status, dict):
            raise EvaluationArtifactError("Benchmark status is malformed")
        if run.run_id != run_id:
            raise EvaluationArtifactError("Benchmark manifest run ID does not match its directory")
        if status.get("status") != "completed":
            raise EvaluationArtifactError("Only completed benchmark runs can be evaluated")
        results = []
        try:
            paths = sorted((directory / "results").glob("*.json"))
            for path in paths:
                results.append(BenchmarkResult.model_validate_json(path.read_bytes()))
        except (OSError, ValidationError, ValueError, TypeError) as exc:
            raise EvaluationArtifactError("A benchmark result is malformed") from exc
        if not results:
            raise EvaluationArtifactError("Benchmark run contains no results")
        if len({result.result_id for result in results}) != len(results):
            raise EvaluationArtifactError("Benchmark result IDs must be unique")
        selected = set(run.selected_task_ids)
        dataset_tasks = {task.task_id for task in run.dataset.tasks}
        models = {model.model_id for model in run.models}
        if not selected or len(selected) != len(run.selected_task_ids) or not selected <= dataset_tasks:
            raise EvaluationArtifactError("Benchmark manifest contains invalid selected task IDs")
        if not models or len(models) != len(run.models):
            raise EvaluationArtifactError("Benchmark manifest contains duplicate or empty model selection")
        pairs = [(result.task_id, result.model_id) for result in results]
        expected = {(task_id, model_id) for task_id in selected for model_id in models}
        if len(set(pairs)) != len(pairs) or set(pairs) != expected:
            raise EvaluationArtifactError("Benchmark results must contain exactly one result per selected task/model pair")
        if any(result.run_id != run_id for result in results):
            raise EvaluationArtifactError("Benchmark result belongs to a different run")
        return run, results

    async def load_run(self, run_id: UUID) -> tuple[BenchmarkRun, list[BenchmarkResult]]:
        return await asyncio.to_thread(self._load_run, run_id)

    async def save(self, run_id: UUID, evaluations: list[EvaluationResult], summary: EvaluationSummary) -> None:
        def persist() -> None:
            directory = self._directory(run_id) / "evaluations"
            directory.mkdir(parents=True, exist_ok=True)
            for evaluation in evaluations:
                self._write(directory / f"{evaluation.benchmark_result_id}.json",
                            evaluation.model_dump_json(indent=2))
            self._write(self._directory(run_id) / "evaluation-summary.json",
                        summary.model_dump_json(indent=2))
        await asyncio.to_thread(persist)

    def _load_summary(self, run_id: UUID) -> EvaluationSummary:
        try:
            summary = EvaluationSummary.model_validate_json(
                (self._directory(run_id) / "evaluation-summary.json").read_bytes())
        except FileNotFoundError:
            raise EvaluationNotFoundError(f"Evaluation summary for {run_id} was not found") from None
        except (ValidationError, ValueError, TypeError) as exc:
            raise EvaluationArtifactError("Evaluation summary is malformed") from exc
        if summary.run_id != run_id:
            raise EvaluationArtifactError("Evaluation summary belongs to a different run")
        return summary

    async def load_summary(self, run_id: UUID) -> EvaluationSummary:
        return await asyncio.to_thread(self._load_summary, run_id)
