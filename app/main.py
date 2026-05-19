"""
Advanced RAG Platform — FastAPI Application Entry Point
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


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
    from app.config import settings

    app = FastAPI(
        title="Advanced RAG Platform",
        version="1.0.0",
        description=(
            "Production-grade Retrieval-Augmented Generation platform with "
            "hybrid search, FAISS vector store, semantic embeddings, and "
            "streaming LLM responses powered by Groq."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(f"{request.method} {request.url.path} -> {response.status_code} ({duration_ms:.1f}ms)")
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
        return response

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception on {request.url.path}: {exc}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "An unexpected error occurred."},
        )

    from app.api.routes import documents, health, query
    api_prefix = "/api/v1"
    app.include_router(health.router, prefix=api_prefix)
    app.include_router(query.router, prefix=api_prefix)
    app.include_router(documents.router, prefix=api_prefix)

    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "service": "Advanced RAG Platform",
            "version": "1.0.0",
            "docs": "/docs",
            "health": f"{api_prefix}/health",
        }

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

# ── Mount static frontend ──────────────────────────────────────────────────
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os

_frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(_frontend_dir):
    app.mount("/static", StaticFiles(directory=_frontend_dir), name="static")

@app.get("/app", include_in_schema=False)
async def serve_frontend():
    index = os.path.join(_frontend_dir, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return {"error": "Frontend not found"}
