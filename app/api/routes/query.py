"""
Query Routes
"""

import logging
import traceback

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from app.api.dependencies import get_rag
from app.models.query import FeedbackRequest, QueryRequest, QueryResponse
from app.services.rag_service import RAGService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/query", tags=["Query"])


@router.post("", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def query(
    request: QueryRequest,
    rag: RAGService = Depends(get_rag),
) -> QueryResponse:
    try:
        return await rag.query(request)
    except Exception as exc:
        # Log the full traceback so we can see exactly what failed
        logger.error(f"RAG query failed: {exc}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {type(exc).__name__}: {str(exc)}"
        )


@router.post("/stream", summary="Stream a RAG query response")
async def stream_query(
    request: QueryRequest,
    rag: RAGService = Depends(get_rag),
) -> StreamingResponse:
    async def event_generator():
        try:
            async for token in rag.stream_query(request):
                yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:
            logger.error(f"Streaming failed: {exc}")
            yield f"data: [ERROR] {str(exc)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


@router.post("/feedback", status_code=status.HTTP_202_ACCEPTED)
async def submit_feedback(feedback: FeedbackRequest) -> dict:
    logger.info(f"Feedback: query_id={feedback.query_id}, rating={feedback.rating}")
    return {"accepted": True, "query_id": feedback.query_id}
