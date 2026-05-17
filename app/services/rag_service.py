"""
RAG Orchestration Service
The central service that ties together retrieval, context assembly, and LLM generation.
"""

import uuid
from datetime import datetime
from typing import AsyncGenerator, Dict, List, Optional

from app.config import settings
from app.core.llm import get_llm_client
from app.core.retriever import get_retriever
from app.models.query import (
    QueryRequest,
    QueryResponse,
    RetrievalMode,
    SourceDocument,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RAGService:
    """
    Orchestrates the full RAG pipeline:
      1. Retrieve relevant chunks via hybrid search
      2. (Optional) Rerank candidates
      3. Build context-aware prompt
      4. Generate answer with LLM
      5. Return structured response with citations
    """

    def __init__(self):
        self._retriever = get_retriever()
        self._llm = get_llm_client()

    async def query(self, request: QueryRequest) -> QueryResponse:
        """Execute a full RAG query and return a structured response."""
        query_id = str(uuid.uuid4())
        logger.info(
            "RAG query started",
            query_id=query_id,
            question_preview=request.question[:80],
            mode=request.retrieval_mode,
        )

        # ── Step 1: Retrieve ─────────────────────────────────────────────────
        sources: List[SourceDocument] = await self._retriever.retrieve(
            query=request.question,
            top_k=request.top_k,
            mode=request.retrieval_mode,
            alpha=request.hybrid_alpha,
            filters=request.filters,
            rerank=request.rerank,
        )

        # ── Step 2: Filter by similarity threshold ───────────────────────────
        sources = [
            s for s in sources
            if s.score >= settings.similarity_threshold
        ]

        if not sources:
            logger.warning("No relevant sources found", query_id=query_id)

        # ── Step 3: Generate ─────────────────────────────────────────────────
        llm_result = await self._llm.generate(
            question=request.question,
            sources=sources,
            conversation_history=request.conversation_history,
        )

        response = QueryResponse(
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

        logger.info(
            "RAG query complete",
            query_id=query_id,
            sources_returned=len(sources),
            latency_ms=llm_result["latency_ms"],
        )
        return response

    async def stream_query(
        self, request: QueryRequest
    ) -> AsyncGenerator[str, None]:
        """Execute a RAG query with streaming LLM output."""
        sources: List[SourceDocument] = await self._retriever.retrieve(
            query=request.question,
            top_k=request.top_k,
            mode=request.retrieval_mode,
            alpha=request.hybrid_alpha,
            filters=request.filters,
            rerank=request.rerank,
        )

        sources = [s for s in sources if s.score >= settings.similarity_threshold]

        async for token in self._llm.stream(
            question=request.question,
            sources=sources,
            conversation_history=request.conversation_history,
        ):
            yield token


# Singleton
_rag_service: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service
