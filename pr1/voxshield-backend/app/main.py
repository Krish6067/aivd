"""
app/main.py — VOXSHIELD Backend

FastAPI application factory:
  - Registers all routers.
  - Configures CORS for the frontend origin.
  - Loads the AASIST model at startup.
  - Adds safe global exception handlers (no stack traces in responses).
  - Configures request size limits.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    cfg = get_settings()

    app = FastAPI(
        title="VOXSHIELD Anti-Spoofing API",
        description=(
            "Real-time AI-generated voice detection backend powered by AASIST "
            "(Audio Anti-Spoofing using Integrated Spectro-Temporal Graph "
            "Attention Networks).\n\n"
            "**Note:** Detection results are probabilistic and must not be "
            "treated as definitive proof of fraud."
        ),
        version=cfg.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    # Allow the VOXSHIELD frontend origin.
    # In production restrict this to the exact deployed frontend URL.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Accept", "X-Request-ID"],
        expose_headers=["X-Processing-Time"],
    )

    # ── Routers ───────────────────────────────────────────────────────────────
    from app.api.health    import router as health_router
    from app.api.analysis  import router as analysis_router
    from app.api.websocket import router as ws_router

    app.include_router(health_router)
    app.include_router(analysis_router)
    app.include_router(ws_router)

    # ── Startup: load model ───────────────────────────────────────────────────
    @app.on_event("startup")
    async def _startup() -> None:
        from app.services.model_service import get_model_service
        svc = get_model_service()
        svc.load(
            weights_path=cfg.model_weights_path,
            device=cfg.resolved_device,
            max_workers=cfg.max_inference_workers,
        )
        if svc.is_ready():
            logger.info(
                "VOXSHIELD backend ready — model on %s.", cfg.resolved_device
            )
        else:
            logger.warning(
                "VOXSHIELD backend started WITHOUT a loaded model. "
                "Analysis endpoints will return 503 until weights are installed. "
                "See models/AASIST/README.md for download instructions."
            )

    # ── Shutdown: release executor ─────────────────────────────────────────────
    @app.on_event("shutdown")
    async def _shutdown() -> None:
        from app.services.model_service import get_model_service
        get_model_service().shutdown()
        logger.info("VOXSHIELD backend shut down.")

    # ── Global exception handlers (no internal details exposed) ───────────────
    @app.exception_handler(Exception)
    async def _generic_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error("Unhandled exception on %s: %s", request.url, exc, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "code":    "INTERNAL_SERVER_ERROR",
                "message": "An unexpected server error occurred.",
            },
        )

    @app.exception_handler(413)
    async def _too_large(request: Request, exc) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            content={
                "code":    "FILE_TOO_LARGE",
                "message": f"Request body exceeds the server limit.",
            },
        )

    return app


# ── Application instance ──────────────────────────────────────────────────────
app = create_app()


# ── Dev-server entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    cfg = get_settings()
    uvicorn.run(
        "app.main:app",
        host=cfg.api_host,
        port=cfg.api_port,
        reload=True,
        log_level=cfg.log_level,
    )
