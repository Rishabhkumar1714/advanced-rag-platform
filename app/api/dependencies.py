"""
FastAPI Dependency Injection
Provides reusable dependencies for routes.
"""

from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import settings
from app.services.document_service import DocumentService, get_document_service
from app.services.ingestion_service import IngestionService
from app.services.rag_service import RAGService, get_rag_service
import logging

logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=False)


# ── Auth ──────────────────────────────────────────────────────────────────────

async def verify_api_key(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
) -> str:
    """
    Lightweight Bearer-token auth.
    In production, replace with JWT validation or an auth service.
    """
    if settings.is_production:
        if not credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing Bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        # Validate token here (e.g., decode JWT, check DB)
        # For demo, any non-empty token passes
        if not credentials.credentials:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid token",
            )
    return credentials.credentials if credentials else "dev-token"


# ── Service Dependencies ──────────────────────────────────────────────────────

def get_rag() -> RAGService:
    return get_rag_service()


def get_doc_service() -> DocumentService:
    return get_document_service()


def get_ingestion_service() -> IngestionService:
    return IngestionService()


# Type aliases for cleaner route signatures
RAGDep = Annotated[RAGService, Depends(get_rag)]
DocServiceDep = Annotated[DocumentService, Depends(get_doc_service)]
IngestionDep = Annotated[IngestionService, Depends(get_ingestion_service)]
AuthDep = Annotated[str, Depends(verify_api_key)]
