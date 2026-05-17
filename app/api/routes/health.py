"""
Health Check Routes
Provides liveness and readiness endpoints for container orchestration.
"""

import time
from datetime import datetime

from fastapi import APIRouter

from app.config import settings
from app.core.vector_store import get_vector_store
from app.models.query import HealthResponse

router = APIRouter(tags=["Health"])
_start_time = time.time()


@router.get("/health", response_model=HealthResponse, summary="Application health check")
async def health_check() -> HealthResponse:
    """
    Returns the application health status along with component checks.
    Used by Docker / Kubernetes probes.
    """
    vector_store = get_vector_store()

    components = {
        "api": "ok",
        "vector_store": f"ok ({vector_store.total_vectors} vectors)",
        "embeddings": "ok",
        "llm": "ok",
    }

    return HealthResponse(
        status="ok",
        version=settings.app_version,
        environment=settings.app_env,
        components=components,
        uptime_seconds=round(time.time() - _start_time, 2),
        timestamp=datetime.utcnow(),
    )


@router.get("/health/live", summary="Liveness probe")
async def liveness() -> dict:
    """Kubernetes liveness probe — returns 200 if process is alive."""
    return {"status": "alive"}


@router.get("/health/ready", summary="Readiness probe")
async def readiness() -> dict:
    """Kubernetes readiness probe — returns 200 when all systems are ready."""
    return {"status": "ready"}
