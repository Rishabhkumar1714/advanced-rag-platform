"""
Advanced RAG Platform — FastAPI Application Entry Point
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Path to frontend folder
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Starting Advanced RAG Platform...")
    try:
        from app.core.vector_store import get_vector_store
        get_vector_store()
        logger.info("Vector store initialized")
    except Exception as e:
        logger.warning(f"Vector store init warning: {e}")
    try:
        from app.services.rag_service import get_rag_service
        get_rag_service()
        logger.info("RAG service initialized")
    except Exception as e:
        logger.warning(f"RAG service init warning: {e}")
    logger.info("Application ready!")
    yield
    logger.info("Shutting down...")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Advanced RAG Platform",
        version="1.0.0",
        description="Production-grade RAG platform with hybrid search, FAISS, and Groq LLM.",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request Logging ────────────────────────────────────────────────────────
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(f"{request.method} {request.url.path} -> {response.status_code} ({duration_ms:.1f}ms)")
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
        return response

    # ── Global Exception Handler ───────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception on {request.url.path}: {exc}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "An unexpected error occurred."},
        )

    # ── API Routes ─────────────────────────────────────────────────────────────
    from app.api.routes import documents, health, query
    api_prefix = "/api/v1"
    app.include_router(health.router, prefix=api_prefix)
    app.include_router(query.router, prefix=api_prefix)
    app.include_router(documents.router, prefix=api_prefix)

    # ── Serve Frontend at root / ───────────────────────────────────────────────
    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        index_path = os.path.join(FRONTEND_DIR, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path, media_type="text/html")
        return JSONResponse({
            "service": "Advanced RAG Platform",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/api/v1/health",
        })

    # ── Mount static assets ────────────────────────────────────────────────────
    if os.path.exists(FRONTEND_DIR):
        app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
        logger.info(f"Frontend mounted from {FRONTEND_DIR}")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
