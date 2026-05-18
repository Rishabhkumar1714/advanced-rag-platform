"""
Document Service
Manages document metadata (CRUD) backed by an in-memory store.
In production this connects to PostgreSQL via SQLAlchemy.
"""

from datetime import datetime
from typing import Dict, List, Optional

from app.models.document import (
    DocumentDeleteResponse,
    DocumentListResponse,
    DocumentResponse,
    DocumentStatus,
    DocumentType,
    DocumentUpdateRequest,
    IngestedDocument,
)
import logging

logger = logging.getLogger(__name__)


class DocumentService:
    """
    In-memory document registry.
    Replace `_store` reads/writes with async DB calls for production.
    """

    def __init__(self):
        self._store: Dict[str, DocumentResponse] = {}

    # ── Create ────────────────────────────────────────────────────────────────

    def register(self, doc: IngestedDocument) -> DocumentResponse:
        """Create a new document record after successful ingestion."""
        now = datetime.utcnow()
        record = DocumentResponse(
            document_id=doc.document_id,
            title=doc.title,
            source=doc.source,
            document_type=doc.document_type,
            status=DocumentStatus.INDEXED,
            chunk_count=len(doc.chunks),
            metadata=doc.metadata,
            collection_id=doc.collection_id,
            created_at=now,
            updated_at=now,
        )
        self._store[doc.document_id] = record
        logger.info(
            "Document registered",
            document_id=doc.document_id,
            chunks=len(doc.chunks),
        )
        return record

    # ── Read ──────────────────────────────────────────────────────────────────

    def get(self, document_id: str) -> Optional[DocumentResponse]:
        return self._store.get(document_id)

    def list_documents(
        self,
        page: int = 1,
        page_size: int = 20,
        collection_id: Optional[str] = None,
        status: Optional[DocumentStatus] = None,
    ) -> DocumentListResponse:
        items = list(self._store.values())

        if collection_id:
            items = [i for i in items if i.collection_id == collection_id]
        if status:
            items = [i for i in items if i.status == status]

        # Sort newest first
        items.sort(key=lambda x: x.created_at, reverse=True)

        total = len(items)
        start = (page - 1) * page_size
        end = start + page_size
        page_items = items[start:end]

        return DocumentListResponse(
            items=page_items,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=max(1, -(-total // page_size)),  # ceiling division
        )

    # ── Update ────────────────────────────────────────────────────────────────

    def update(
        self, document_id: str, update: DocumentUpdateRequest
    ) -> Optional[DocumentResponse]:
        record = self._store.get(document_id)
        if not record:
            return None

        update_data = update.model_dump(exclude_none=True)
        updated = record.model_copy(
            update={**update_data, "updated_at": datetime.utcnow()}
        )
        self._store[document_id] = updated
        logger.info("Document updated", document_id=document_id)
        return updated

    def set_status(self, document_id: str, status: DocumentStatus) -> None:
        record = self._store.get(document_id)
        if record:
            self._store[document_id] = record.model_copy(
                update={"status": status, "updated_at": datetime.utcnow()}
            )

    # ── Delete ────────────────────────────────────────────────────────────────

    def delete(self, document_id: str) -> DocumentDeleteResponse:
        if document_id not in self._store:
            return DocumentDeleteResponse(
                document_id=document_id,
                deleted=False,
                message="Document not found",
            )
        del self._store[document_id]
        logger.info("Document deleted from registry", document_id=document_id)
        return DocumentDeleteResponse(
            document_id=document_id,
            deleted=True,
            message="Document successfully deleted",
        )

    @property
    def total_documents(self) -> int:
        return len(self._store)


# Singleton
_doc_service: Optional[DocumentService] = None


def get_document_service() -> DocumentService:
    global _doc_service
    if _doc_service is None:
        _doc_service = DocumentService()
    return _doc_service
