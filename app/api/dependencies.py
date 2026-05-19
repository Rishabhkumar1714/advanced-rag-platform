"""
FastAPI Dependency Injection
"""

import logging

from app.services.document_service import DocumentService, get_document_service
from app.services.ingestion_service import IngestionService
from app.services.rag_service import RAGService, get_rag_service

logger = logging.getLogger(__name__)


def get_rag() -> RAGService:
    return get_rag_service()


def get_doc_service() -> DocumentService:
    return get_document_service()


def get_ingestion_service() -> IngestionService:
    return IngestionService()
