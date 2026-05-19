"""
Document Management Routes
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status

from app.api.dependencies import get_doc_service, get_ingestion_service
from app.config import settings
from app.models.document import (
    DocumentDeleteResponse, DocumentListResponse, DocumentResponse,
    DocumentStatus, DocumentUpdateRequest, DocumentUploadRequest,
)
from app.services.document_service import DocumentService
from app.services.ingestion_service import IngestionService
from app.utils.helpers import get_file_extension

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED, summary="Ingest raw text content")
async def create_document(
    request: DocumentUploadRequest,
    ingestion: IngestionService = Depends(get_ingestion_service),
    doc_service: DocumentService = Depends(get_doc_service),
) -> DocumentResponse:
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
        logger.error(f"Text ingestion failed: {request.title}: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED, summary="Upload a document file")
async def upload_document(
    file: UploadFile = File(...),
    collection_id: Optional[str] = Form(None),
    ingestion: IngestionService = Depends(get_ingestion_service),
    doc_service: DocumentService = Depends(get_doc_service),
) -> DocumentResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    ext = get_file_extension(file.filename)
    if ext not in settings.allowed_extensions_list:
        raise HTTPException(status_code=400, detail=f"File type '{ext}' not allowed.")
    content = await file.read()
    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File too large. Max: {settings.max_file_size_mb}MB")
    try:
        ingested_doc = await ingestion.ingest_file(file_content=content, filename=file.filename, collection_id=collection_id)
        return doc_service.register(ingested_doc)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error(f"File ingestion failed: {file.filename}: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("", response_model=DocumentListResponse, summary="List all documents")
async def list_documents(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    collection_id: Optional[str] = Query(default=None),
    doc_service: DocumentService = Depends(get_doc_service),
) -> DocumentListResponse:
    return doc_service.list_documents(page=page, page_size=page_size, collection_id=collection_id)


@router.get("/{document_id}", response_model=DocumentResponse, summary="Get document by ID")
async def get_document(
    document_id: str,
    doc_service: DocumentService = Depends(get_doc_service),
) -> DocumentResponse:
    doc = doc_service.get(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.patch("/{document_id}", response_model=DocumentResponse, summary="Update document metadata")
async def update_document(
    document_id: str,
    update: DocumentUpdateRequest,
    doc_service: DocumentService = Depends(get_doc_service),
) -> DocumentResponse:
    doc = doc_service.update(document_id, update)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.delete("/{document_id}", response_model=DocumentDeleteResponse, summary="Delete a document")
async def delete_document(
    document_id: str,
    ingestion: IngestionService = Depends(get_ingestion_service),
    doc_service: DocumentService = Depends(get_doc_service),
) -> DocumentDeleteResponse:
    doc = doc_service.get(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    await ingestion.delete_document(document_id)
    return doc_service.delete(document_id)
