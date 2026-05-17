"""
Document Management Routes
CRUD operations for documents plus file upload ingestion.
"""

from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status

from app.api.dependencies import AuthDep, DocServiceDep, IngestionDep
from app.config import settings
from app.models.document import (
    DocumentDeleteResponse,
    DocumentListResponse,
    DocumentResponse,
    DocumentStatus,
    DocumentUpdateRequest,
    DocumentUploadRequest,
)
from app.utils.helpers import get_file_extension
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/documents", tags=["Documents"])


# ── Upload / Ingest ───────────────────────────────────────────────────────────

@router.post(
    "/upload",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a document file",
)
async def upload_document(
    file: UploadFile = File(..., description="Document file to ingest"),
    collection_id: Optional[str] = Form(None),
    ingestion: IngestionDep = None,
    doc_service: DocServiceDep = None,
    _token: AuthDep = None,
) -> DocumentResponse:
    """
    Upload a PDF, DOCX, TXT, MD, or HTML file.
    The file is chunked, embedded, and indexed automatically.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    ext = get_file_extension(file.filename)
    if ext not in settings.allowed_extensions_list:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{ext}' not allowed. Allowed: {settings.allowed_extensions_list}",
        )

    content = await file.read()
    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {settings.max_file_size_mb} MB",
        )

    try:
        ingested_doc = await ingestion.ingest_file(
            file_content=content,
            filename=file.filename,
            collection_id=collection_id,
        )
        return doc_service.register(ingested_doc)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("File ingestion failed", filename=file.filename, error=str(exc))
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(exc)}") from exc


@router.post(
    "",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest raw text content",
)
async def create_document(
    request: DocumentUploadRequest,
    ingestion: IngestionDep = None,
    doc_service: DocServiceDep = None,
    _token: AuthDep = None,
) -> DocumentResponse:
    """Ingest raw text content without a file upload."""
    if not request.content:
        raise HTTPException(status_code=400, detail="'content' is required for text ingestion")

    try:
        ingested_doc = await ingestion.ingest_text(
            title=request.title,
            content=request.content,
            source=request.source,
            metadata=request.metadata,
            collection_id=request.collection_id,
        )
        return doc_service.register(ingested_doc)
    except Exception as exc:
        logger.error("Text ingestion failed", title=request.title, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ── Read ──────────────────────────────────────────────────────────────────────

@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List all documents",
)
async def list_documents(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    collection_id: Optional[str] = Query(default=None),
    status: Optional[DocumentStatus] = Query(default=None),
    doc_service: DocServiceDep = None,
    _token: AuthDep = None,
) -> DocumentListResponse:
    return doc_service.list_documents(
        page=page,
        page_size=page_size,
        collection_id=collection_id,
        status=status,
    )


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Get document by ID",
)
async def get_document(
    document_id: str,
    doc_service: DocServiceDep = None,
    _token: AuthDep = None,
) -> DocumentResponse:
    doc = doc_service.get(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


# ── Update ────────────────────────────────────────────────────────────────────

@router.patch(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Update document metadata",
)
async def update_document(
    document_id: str,
    update: DocumentUpdateRequest,
    doc_service: DocServiceDep = None,
    _token: AuthDep = None,
) -> DocumentResponse:
    doc = doc_service.update(document_id, update)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


# ── Delete ────────────────────────────────────────────────────────────────────

@router.delete(
    "/{document_id}",
    response_model=DocumentDeleteResponse,
    summary="Delete a document and its vectors",
)
async def delete_document(
    document_id: str,
    ingestion: IngestionDep = None,
    doc_service: DocServiceDep = None,
    _token: AuthDep = None,
) -> DocumentDeleteResponse:
    doc = doc_service.get(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    await ingestion.delete_document(document_id)
    return doc_service.delete(document_id)
