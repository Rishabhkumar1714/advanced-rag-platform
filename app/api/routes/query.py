"""
Query Routes
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from app.api.dependencies import get_rag
from app.models.query import FeedbackRequest, QueryRequest, QueryResponse
from app.services.rag_service import RAGService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/query", tags=["Query"])


@router.post("", response_model=QueryResponse, status_code=status.HTTP_200_OK, summary="Submit a RAG query")
async def query(
    request: QueryRequest,
    rag: RAGService = Depends(get_rag),
) -> QueryResponse:
    if request.stream:
        raise HTTPException(status_code=400, detail="Use /query/stream for streaming.")
    try:
        return await rag.query(request)
    except Exception as exc:
        logger.error(f"RAG query failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(exc)}")


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
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.post("/feedback", status_code=status.HTTP_202_ACCEPTED, summary="Submit feedback")
async def submit_feedback(feedback: FeedbackRequest) -> dict:
    logger.info(f"Feedback: query_id={feedback.query_id}, rating={feedback.rating}")
    return {"accepted": True, "query_id": feedback.query_id}
