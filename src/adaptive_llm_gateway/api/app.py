"""ASGI entry point: uvicorn adaptive_llm_gateway.api.app:app."""

import re
from contextlib import asynccontextmanager
from collections.abc import Awaitable, Callable
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException
from starlette.responses import Response

from adaptive_llm_gateway.application.service import InferenceService
from adaptive_llm_gateway.bootstrap import create_development_service
from adaptive_llm_gateway.errors import (
    ContextLimitError,
    ModelDisabledError,
    ProviderFailureError,
    ProviderUnavailableError,
)
from adaptive_llm_gateway.registry import ModelNotFoundError
from adaptive_llm_gateway.runtime import application_service
from adaptive_llm_gateway.telemetry.query import TelemetryUnavailableError

from .routes import router
from .schemas import ErrorDetail, ErrorResponse

_REQUEST_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_ERRORS = {
    TelemetryUnavailableError: (503, "telemetry_unavailable", "Telemetry is currently unavailable."),
    ModelNotFoundError: (404, "model_not_found", "The requested model was not found."),
    ModelDisabledError: (403, "model_disabled", "The requested model is disabled."),
    ContextLimitError: (422, "context_limit_exceeded", "Input plus requested output exceeds the context window."),
    ProviderUnavailableError: (503, "provider_unavailable", "No adapter is available for this model."),
    ProviderFailureError: (502, "provider_failure", "The provider could not complete inference."),
}


def error_response(request: Request, status: int, code: str, message: str) -> JSONResponse:
    body = ErrorResponse(error=ErrorDetail(code=code, message=message),
                         request_id=request.state.request_id)
    return JSONResponse(status_code=status, content=body.model_dump(),
                        headers={"X-Request-ID": request.state.request_id})


def create_app(service: InferenceService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if service is not None:
            yield
        else:
            async with application_service() as configured_service:
                application.state.inference_service = configured_service
                yield

    app = FastAPI(
        title="Adaptive LLM Gateway",
        version="0.3.0",
        description="Phase 3: explicit model inference with PostgreSQL telemetry. Intelligent routing is deferred.",
        lifespan=lifespan,
    )
    app.state.inference_service = service if service is not None else create_development_service()

    @app.middleware("http")
    async def correlation_id(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        supplied_ids = request.headers.getlist("x-request-id")
        request.state.request_id = str(uuid4())
        if supplied_ids:
            if len(supplied_ids) != 1 or not _REQUEST_ID.fullmatch(supplied_ids[0]):
                return error_response(request, 400, "invalid_request_id",
                                      "X-Request-ID must contain 1-128 ASCII letters, digits, dots, underscores or hyphens, starting with a letter or digit.")
            request.state.request_id = supplied_ids[0]
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    async def application_error(request: Request, exc: Exception) -> JSONResponse:
        # Resolve subclasses as well as the explicitly registered error types.
        for error_type, (status, code, message) in _ERRORS.items():
            if isinstance(exc, error_type):
                return error_response(request, status, code, message)
        return error_response(request, 500, "internal_error", "An internal error occurred.")

    for error_type in _ERRORS:
        app.add_exception_handler(error_type, application_error)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Avoid echoing prompts, supplied input, or Pydantic exception context.
        return error_response(request, 422, "invalid_request", "Request body does not match the inference schema.")

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        response = error_response(request, exc.status_code, "http_error", "The HTTP request could not be completed.")
        if exc.headers:
            response.headers.update(exc.headers)
        return response

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        return error_response(request, 500, "internal_error", "An internal error occurred.")

    app.include_router(router)
    return app


app = create_app()
