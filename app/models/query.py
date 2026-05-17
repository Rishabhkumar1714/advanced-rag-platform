"""
Query & Response Models
Pydantic schemas for RAG query requests and structured responses.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class RetrievalMode(str, Enum):
    SEMANTIC = "semantic"       # Pure vector similarity search
    KEYWORD = "keyword"         # BM25 / keyword-based
    HYBRID = "hybrid"           # Weighted combination of both


class SearchFilter(BaseModel):
    """Metadata filter to narrow the retrieval scope."""

    field: str = Field(..., description="Metadata field name to filter on")
    value: Any = Field(..., description="Filter value")
    operator: str = Field(default="eq", description="eq | ne | gt | lt | in | nin")

    @field_validator("operator")
    @classmethod
    def validate_operator(cls, v: str) -> str:
        allowed = {"eq", "ne", "gt", "lt", "gte", "lte", "in", "nin"}
        if v not in allowed:
            raise ValueError(f"operator must be one of {allowed}")
        return v


# ── Request Schemas ───────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """User query submitted to the RAG pipeline."""

    question: str = Field(
        ..., min_length=3, max_length=2048, description="Natural language question"
    )
    collection_id: Optional[str] = Field(
        None, description="Restrict retrieval to a specific collection"
    )
    retrieval_mode: RetrievalMode = Field(
        default=RetrievalMode.HYBRID, description="Retrieval strategy"
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    hybrid_alpha: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Semantic weight for hybrid search (overrides config)",
    )
    filters: List[SearchFilter] = Field(
        default_factory=list, description="Metadata filters"
    )
    rerank: bool = Field(default=True, description="Apply cross-encoder reranking")
    include_sources: bool = Field(default=True, description="Include source references")
    stream: bool = Field(default=False, description="Stream response tokens")
    conversation_history: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Previous turns for multi-turn conversation",
    )


class FeedbackRequest(BaseModel):
    """User feedback on a RAG response."""

    query_id: str
    rating: int = Field(..., ge=1, le=5, description="1-5 star rating")
    comment: Optional[str] = Field(None, max_length=1024)
    helpful: bool


# ── Response Schemas ──────────────────────────────────────────────────────────

class SourceDocument(BaseModel):
    """A retrieved document chunk with relevance metadata."""

    chunk_id: str
    document_id: str
    title: str
    content: str
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    chunk_index: int
    source: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    """Full RAG pipeline response."""

    query_id: str
    question: str
    answer: str
    sources: List[SourceDocument] = Field(default_factory=list)
    retrieval_mode: RetrievalMode
    total_chunks_retrieved: int
    model_used: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    created_at: datetime


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    components: Dict[str, str]
    uptime_seconds: float
    timestamp: datetime
