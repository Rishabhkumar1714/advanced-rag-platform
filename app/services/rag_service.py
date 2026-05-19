"""
RAG Orchestration Service
"""

import logging
import traceback
import uuid
from datetime import datetime
from typing import AsyncGenerator, List, Optional

from app.config import settings
from app.core.llm import get_llm_client
from app.core.retriever import get_retriever
from app.models.query import QueryRequest, QueryResponse, SourceDocument

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self):
        self._retriever = get_retriever()
        self._llm = get_llm_client()

    async def query(self, request: QueryRequest) -> QueryResponse:
        query_id = str(uuid.uuid4())
        logger.info(f"RAG query: id={query_id}, question={request.question[:60]}")

        # Step 1: Retrieve
        try:
            sources: List[SourceDocument] = await self._retriever.retrieve(
                query=request.question,
                top_k=request.top_k,
                mode=request.retrieval_mode,
                alpha=request.hybrid_alpha,
                filters=request.filters,
                rerank=request.rerank,
            )
            sources = [s for s in sources if s.score >= settings.similarity_threshold]
            logger.info(f"Retrieved {len(sources)} sources above threshold")
        except Exception as e:
            logger.error(f"Retrieval failed: {e}\n{traceback.format_exc()}")
            sources = []  # continue with empty sources rather than crash

        # Step 2: Generate
        try:
            llm_result = await self._llm.generate(
                question=request.question,
                sources=sources,
                conversation_history=request.conversation_history,
            )
        except Exception as e:
            logger.error(f"LLM generation failed: {e}\n{traceback.format_exc()}")
            raise RuntimeError(f"LLM generation failed: {type(e).__name__}: {str(e)}")

        return QueryResponse(
            query_id=query_id,
            question=request.question,
            answer=llm_result["answer"],
            sources=sources if request.include_sources else [],
            retrieval_mode=request.retrieval_mode,
            total_chunks_retrieved=len(sources),
            model_used=llm_result["model_used"],
            prompt_tokens=llm_result["prompt_tokens"],
            completion_tokens=llm_result["completion_tokens"],
            latency_ms=llm_result["latency_ms"],
            created_at=datetime.utcnow(),
        )

    async def stream_query(self, request: QueryRequest) -> AsyncGenerator[str, None]:
        try:
            sources: List[SourceDocument] = await self._retriever.retrieve(
                query=request.question, top_k=request.top_k,
                mode=request.retrieval_mode, alpha=request.hybrid_alpha,
                filters=request.filters, rerank=request.rerank,
            )
            sources = [s for s in sources if s.score >= settings.similarity_threshold]
        except Exception as e:
            logger.error(f"Retrieval failed in stream: {e}")
            sources = []

        async for token in self._llm.stream(
            question=request.question, sources=sources,
            conversation_history=request.conversation_history,
        ):
            yield token


_rag_service: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service
