"""
Advanced RAG Platform — FastAPI Application Entry Point
"""

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import documents, health, query
from app.config import settings
from app.utils.logger import RequestLogger, get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Run startup and shutdown logic."""
    logger.info(
        "Starting Advanced RAG Platform",
        version=settings.app_version,
        environment=settings.app_env,
    )

    # Pre-load heavy singletons so the first request isn't slow
    from app.core.embeddings import get_embedding_pipeline
    from app.core.vector_store import get_vector_store
    from app.services.rag_service import get_rag_service

    get_vector_store()
    get_rag_service()
    logger.info("All components initialized — application ready")

    yield  # ← App is running

    logger.info("Shutting down Advanced RAG Platform")


# ── App Factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Production-grade Retrieval-Augmented Generation platform with "
            "hybrid search, vector embeddings, metadata filtering, and "
            "streaming LLM responses."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ── CORS ─────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request Logging Middleware ────────────────────────────────────────────
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.perf_counter()
        client_ip = request.client.host if request.client else "unknown"
        RequestLogger.log_request(request.method, request.url.path, client_ip)

        response = await call_next(request)

        duration_ms = (time.perf_counter() - start) * 1000
        RequestLogger.log_response(
            request.method, request.url.path, response.status_code, duration_ms
        )
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
        return response

    # ── Global Exception Handler ──────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(
            "Unhandled exception",
            path=request.url.path,
            method=request.method,
            error=str(exc),
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "An unexpected error occurred. Please try again."},
        )

    # ── Routers ───────────────────────────────────────────────────────────────
    api_prefix = "/api/v1"
    app.include_router(health.router, prefix=api_prefix)
    app.include_router(query.router, prefix=api_prefix)
    app.include_router(documents.router, prefix=api_prefix)

    # ── Metrics (Prometheus) ──────────────────────────────────────────────────
    if settings.metrics_enabled:
        try:
            from prometheus_fastapi_instrumentator import Instrumentator
            Instrumentator().instrument(app).expose(app, endpoint="/metrics")
            logger.info("Prometheus metrics enabled at /metrics")
        except ImportError:
            logger.warning("prometheus-fastapi-instrumentator not installed; metrics disabled")

    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "service": settings.app_name,
            "version": settings.app_version,
            "docs": "/docs",
            "health": f"{api_prefix}/health",
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=not settings.is_production,
        workers=1 if not settings.is_production else settings.workers,
        log_level=settings.log_level.lower(),
    )
