from fastapi import Request

from adaptive_llm_gateway.application.service import InferenceService


async def get_service(request: Request) -> InferenceService:
    """App-local dependency; replace through create_app or dependency_overrides."""
    return request.app.state.inference_service
