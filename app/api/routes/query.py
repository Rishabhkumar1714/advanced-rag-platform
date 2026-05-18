"""
Query Routes
Handles RAG query requests — both standard and streaming.
"""

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from app.api.dependencies import AuthDep, RAGDep
from app.models.query import FeedbackRequest, QueryRequest, QueryResponse
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/query", tags=["Query"])


@router.post(
    "",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit a RAG query",
    description=(
        "Send a natural-language question. The system retrieves the most relevant "
        "document chunks and generates a grounded answer using an LLM."
    ),
)
async def query(
    request: QueryRequest,
    rag: RAGDep,
    _token: AuthDep,
) -> QueryResponse:
    if request.stream:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Use /query/stream for streaming responses.",
        )

    try:
        return await rag.query(request)
    except Exception as exc:
        logger.error("RAG query failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(exc)}",
        ) from exc


@router.post(
    "/stream",
    summary="Stream a RAG query response",
    description="Same as /query but streams the answer token-by-token via Server-Sent Events.",
)
async def stream_query(
    request: QueryRequest,
    rag: RAGDep,
    _token: AuthDep,
) -> StreamingResponse:
    async def event_generator():
        try:
            async for token in rag.stream_query(request):
                yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:
            logger.error("Streaming query failed", error=str(exc))
            yield f"data: [ERROR] {str(exc)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post(
    "/feedback",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit feedback on a query response",
)
async def submit_feedback(
    feedback: FeedbackRequest,
    _token: AuthDep,
) -> dict:
    """
    Accept user feedback for a previous query.
    In production, persist to DB and use for RLHF / quality monitoring.
    """
    logger.info(
        "Feedback received",
        query_id=feedback.query_id,
        rating=feedback.rating,
        helpful=feedback.helpful,
    )
    return {"accepted": True, "query_id": feedback.query_id}
