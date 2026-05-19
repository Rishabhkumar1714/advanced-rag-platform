"""
Health & Diagnostics Routes
"""

import logging
import os
import time
from datetime import datetime

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.config import settings
from app.core.vector_store import get_vector_store
from app.models.query import HealthResponse

router = APIRouter(tags=["Health"])
logger = logging.getLogger(__name__)
_start_time = time.time()


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health_check() -> HealthResponse:
    vector_store = get_vector_store()
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        environment=settings.app_env,
        components={
            "api": "ok",
            "vector_store": f"ok ({vector_store.total_vectors} vectors)",
        },
        uptime_seconds=round(time.time() - _start_time, 2),
        timestamp=datetime.utcnow(),
    )


@router.get("/health/live", summary="Liveness probe")
async def liveness() -> dict:
    return {"status": "alive"}


@router.get("/health/ready", summary="Readiness probe")
async def readiness() -> dict:
    return {"status": "ready"}


@router.get("/health/diagnose", summary="Full diagnostic — shows exactly what is failing")
async def diagnose() -> JSONResponse:
    """
    Runs each component independently and reports pass/fail.
    Open this URL in browser to see what is broken.
    """
    results = {}

    # 1. Check GROQ_API_KEY
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if groq_key:
        results["groq_api_key"] = f"SET (ends in ...{groq_key[-4:]})"
    else:
        results["groq_api_key"] = "MISSING — add GROQ_API_KEY in Render Environment Variables"

    # 2. Check fastembed
    try:
        from fastembed import TextEmbedding
        results["fastembed"] = "OK — library imported"
    except Exception as e:
        results["fastembed"] = f"FAILED: {e}"

    # 3. Check embedding pipeline
    try:
        from app.core.embeddings import get_embedding_pipeline
        pipeline = get_embedding_pipeline()
        pipeline._load_model()
        test_emb = await pipeline.embed_query("test")
        results["embedding"] = f"OK — dimension={len(test_emb)}"
    except Exception as e:
        results["embedding"] = f"FAILED: {type(e).__name__}: {e}"

    # 4. Check FAISS vector store
    try:
        from app.core.vector_store import get_vector_store
        vs = get_vector_store()
        results["vector_store"] = f"OK — {vs.total_vectors} vectors indexed"
    except Exception as e:
        results["vector_store"] = f"FAILED: {type(e).__name__}: {e}"

    # 5. Check Groq API
    try:
        if not groq_key:
            results["groq_api"] = "SKIPPED — no API key"
        else:
            from groq import AsyncGroq
            client = AsyncGroq(api_key=groq_key)
            response = await client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": "Say OK"}],
                max_tokens=5,
            )
            results["groq_api"] = f"OK — response: {response.choices[0].message.content}"
    except Exception as e:
        results["groq_api"] = f"FAILED: {type(e).__name__}: {e}"

    # Summary
    failed = [k for k, v in results.items() if "FAILED" in str(v) or "MISSING" in str(v)]
    results["overall"] = "ALL OK" if not failed else f"FAILING: {failed}"

    return JSONResponse(content=results)
