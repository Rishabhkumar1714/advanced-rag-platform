"""
API Integration Tests
Tests for health, query, and document endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch

from app.main import app

client = TestClient(app)


# ── Health ────────────────────────────────────────────────────────────────────

class TestHealthEndpoints:
    def test_liveness(self):
        response = client.get("/api/v1/health/live")
        assert response.status_code == 200
        assert response.json()["status"] == "alive"

    def test_readiness(self):
        response = client.get("/api/v1/health/ready")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"

    def test_health_check(self):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "components" in data


# ── Query ─────────────────────────────────────────────────────────────────────

class TestQueryEndpoints:
    @patch("app.services.rag_service.RAGService.query")
    def test_query_success(self, mock_query):
        from datetime import datetime
        from app.models.query import QueryResponse, RetrievalMode

        mock_query.return_value = QueryResponse(
            query_id="test-id",
            question="What is RAG?",
            answer="RAG stands for Retrieval-Augmented Generation.",
            sources=[],
            retrieval_mode=RetrievalMode.HYBRID,
            total_chunks_retrieved=0,
            model_used="gpt-4o",
            prompt_tokens=100,
            completion_tokens=50,
            latency_ms=250.0,
            created_at=datetime.utcnow(),
        )

        response = client.post(
            "/api/v1/query",
            json={"question": "What is RAG?"},
            headers={"Authorization": "Bearer test-token"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["question"] == "What is RAG?"
        assert "answer" in data

    def test_query_missing_question(self):
        response = client.post(
            "/api/v1/query",
            json={},
            headers={"Authorization": "Bearer test-token"},
        )
        assert response.status_code == 422  # Validation error

    def test_query_too_short(self):
        response = client.post(
            "/api/v1/query",
            json={"question": "Hi"},
            headers={"Authorization": "Bearer test-token"},
        )
        assert response.status_code == 422


# ── Documents ─────────────────────────────────────────────────────────────────

class TestDocumentEndpoints:
    def test_list_documents_empty(self):
        response = client.get(
            "/api/v1/documents",
            headers={"Authorization": "Bearer test-token"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data

    def test_get_nonexistent_document(self):
        response = client.get(
            "/api/v1/documents/nonexistent-id",
            headers={"Authorization": "Bearer test-token"},
        )
        assert response.status_code == 404

    def test_delete_nonexistent_document(self):
        response = client.delete(
            "/api/v1/documents/nonexistent-id",
            headers={"Authorization": "Bearer test-token"},
        )
        assert response.status_code == 404
