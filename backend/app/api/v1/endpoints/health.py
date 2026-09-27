import time

import torch
from backend.app.core.config import Settings, get_settings
from backend.app.schemas.health import HealthResponse
from fastapi import APIRouter, Depends

router = APIRouter()
START_TIME = time.time()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service Health Check",
    description="Returns current health status, API version, and runtime telemetry.",
)
async def get_health(
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    uptime = time.time() - START_TIME
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # In Phase 1, models are not yet trained or loaded into the registry.
    models_loaded: list[str] = []

    return HealthResponse(
        status="healthy",
        project=settings.PROJECT_NAME,
        full_name=settings.PROJECT_FULL_NAME,
        version=settings.VERSION,
        environment=settings.ENVIRONMENT,
        uptime_seconds=round(uptime, 2),
        device=device,
        models_loaded=models_loaded,
        dataset_version="city_day_v1.0",
    )
