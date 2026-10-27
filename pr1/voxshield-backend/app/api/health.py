"""
app/api/health.py — VOXSHIELD Backend

GET /api/health   — basic liveness probe
GET /api/model/status — model readiness and configuration
"""

from __future__ import annotations

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.schemas.analysis import HealthResponse, ModelStatusResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get(
    "/api/health",
    response_model=HealthResponse,
    summary="API liveness probe",
    tags=["Health"],
)
async def health() -> HealthResponse:
    """
    Returns 200 OK while the API process is running.
    Does NOT check model readiness — use /api/model/status for that.
    """
    cfg = get_settings()
    return HealthResponse(version=cfg.app_version)


@router.get(
    "/api/model/status",
    response_model=ModelStatusResponse,
    summary="AASIST model readiness and configuration",
    tags=["Health"],
)
async def model_status() -> ModelStatusResponse:
    """
    Returns the current state of the AASIST model:
    - Whether it is loaded and ready for inference.
    - Which device it is running on (cpu / cuda / mps).
    - Supported input sample rate.
    - Error detail if loading failed.

    Check this endpoint before sending analysis requests.
    """
    from app.services.model_service import get_model_service
    svc = get_model_service()
    status = svc.get_status()
    return ModelStatusResponse(**status)
