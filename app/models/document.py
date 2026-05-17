"""
Document Models
Pydantic schemas for document ingestion, storage, and retrieval.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class DocumentType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MARKDOWN = "md"
    HTML = "html"
    UNKNOWN = "unknown"


# ── Request Schemas ───────────────────────────────────────────────────────────

class DocumentUploadRequest(BaseModel):
    """Request payload for uploading a document via URL or raw text."""

    title: str = Field(..., min_length=1, max_length=512, description="Document title")
    source: Optional[str] = Field(None, description="Source URL or path")
    content: Optional[str] = Field(None, description="Raw text content to ingest")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata")
    collection_id: Optional[str] = Field(None, description="Collection/namespace to assign")

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and len(v.strip()) == 0:
            raise ValueError("content must not be empty if provided")
        return v


class DocumentUpdateRequest(BaseModel):
    """Partial update for document metadata."""

    title: Optional[str] = Field(None, max_length=512)
    metadata: Optional[Dict[str, Any]] = None
    collection_id: Optional[str] = None


# ── Response Schemas ──────────────────────────────────────────────────────────

class ChunkResponse(BaseModel):
    """Represents a single text chunk stored in the vector index."""

    chunk_id: str
    document_id: str
    content: str
    chunk_index: int
    token_count: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentResponse(BaseModel):
    """Full document representation returned by the API."""

    document_id: str
    title: str
    source: Optional[str]
    document_type: DocumentType
    status: DocumentStatus
    chunk_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    collection_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class DocumentListResponse(BaseModel):
    """Paginated list of documents."""

    items: List[DocumentResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class DocumentDeleteResponse(BaseModel):
    document_id: str
    deleted: bool
    message: str


# ── Internal Models ───────────────────────────────────────────────────────────

class DocumentChunk(BaseModel):
    """Internal representation of a text chunk with vector metadata."""

    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    content: str
    chunk_index: int
    token_count: int
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


class IngestedDocument(BaseModel):
    """Represents a document after initial processing, before vector indexing."""

    document_id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    source: Optional[str]
    document_type: DocumentType
    raw_content: str
    chunks: List[DocumentChunk] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    collection_id: Optional[str] = None
