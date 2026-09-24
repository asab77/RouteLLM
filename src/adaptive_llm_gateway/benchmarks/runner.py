from time import perf_counter
from uuid import uuid4

from adaptive_llm_gateway.application.service import InferenceService
from adaptive_llm_gateway.errors import ContextLimitError, GatewayError, ModelDisabledError, ProviderFailureError, ProviderUnavailableError
from adaptive_llm_gateway.providers.gateway_config import (
    CATALOG_VERIFIED_AT, PRICING_SOURCE, PRICING_VERIFIED, UPSTREAM_PROVIDERS,
)
from .features import extract_request_features
from .models import BenchmarkDataset, BenchmarkResult, BenchmarkRun
from .repository import BenchmarkRepository


class BenchmarkRunner:
    def __init__(self, service: InferenceService, repository: BenchmarkRepository,
                 *, configuration: dict | None = None) -> None:
        # Separate orchestration instance deliberately has NO production telemetry sink.
        self.service = InferenceService(service.registry, service.resolver)
        self.repository = repository
        self.configuration = configuration or {}

    async def run(self, dataset: BenchmarkDataset, model_ids: list[str], *, limit: int | None = None) -> BenchmarkRun:
        if not model_ids or len(set(model_ids)) != len(model_ids):
            raise ValueError("Select one or more distinct models")
        if limit is not None and limit <= 0:
            raise ValueError("Task limit must be positive")
        models = tuple(self.service.registry.get(model_id) for model_id in model_ids)
        for model in models:
            if not model.enabled:
                raise ModelDisabledError("Selected benchmark model is disabled")
            if not self.service.resolver.supports(model.provider):
                raise ProviderUnavailableError("Selected benchmark adapter is unavailable")
        tasks = dataset.tasks if limit is None else dataset.tasks[:limit]
        request_features = {
            task.task_id: extract_request_features(task).model_dump(mode="json") for task in tasks
        }
        frozen_models = {
            model.model_id: {
                "upstream_model_slug": model.provider_model_name,
                "upstream_provider_pin": UPSTREAM_PROVIDERS.get(model.provider_model_name),
                "service_tier": "standard",
                "region": "provider_default",
                "input_cost_per_1m_tokens": str(model.input_cost_per_1m_tokens),
                "output_cost_per_1m_tokens": str(model.output_cost_per_1m_tokens),
                "capabilities": model.capabilities.model_dump(mode="json"),
                "reasoning_effort": (model.reasoning_effort.value
                                     if model.reasoning_effort is not None else None),
            }
            for model in models
        }
        run = BenchmarkRun(dataset=dataset, dataset_sha256=dataset.sha256,
            selected_task_ids=tuple(task.task_id for task in tasks), models=models,
            configuration={"runner_version": "1", "execution": "sequential-task-then-model",
                "pricing_source": PRICING_SOURCE, "pricing_verified": PRICING_VERIFIED,
                "catalog_verified_at": CATALOG_VERIFIED_AT,
                "upstream_allowlists": UPSTREAM_PROVIDERS,
                "request_feature_schema_version": "1.0.0",
                "request_features": request_features,
                "frozen_model_configuration": frozen_models,
                **self.configuration})
        await self.repository.start(run)  # fail before paid calls if storage cannot be created
        try:
            for task in tasks:
                for model in models:
                    started = perf_counter()
                    request_id = str(uuid4())
                    try:
                        response = await self.service.generate(model.model_id, task.to_request(), request_id=request_id)
                        result = BenchmarkResult(run_id=run.run_id, request_id=request_id,
                            task_id=task.task_id, model_id=model.model_id, success=True,
                            response=response, latency_ms=response.latency_ms)
                    except (ContextLimitError, ModelDisabledError, ProviderUnavailableError, ProviderFailureError) as exc:
                        categories = {ContextLimitError: "context_limit_exceeded", ModelDisabledError: "model_disabled",
                            ProviderUnavailableError: "provider_unavailable", ProviderFailureError: "provider_failure"}
                        category = exc.category.value if isinstance(exc, GatewayError) else next(
                            value for kind, value in categories.items() if isinstance(exc, kind))
                        result = BenchmarkResult(run_id=run.run_id, request_id=request_id,
                            task_id=task.task_id, model_id=model.model_id, success=False,
                            error_category=category,
                            error_details=(exc.diagnostics if isinstance(exc, GatewayError) else {}),
                            latency_ms=(perf_counter() - started) * 1000)
                    # Storage failure aborts rather than continuing paid work with lost results.
                    await self.repository.record(result)
            await self.repository.finish(run.run_id, "completed")
        except BaseException:
            await self.repository.finish(run.run_id, "aborted")
            raise
        return run
