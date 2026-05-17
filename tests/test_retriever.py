"""
Unit Tests — Retriever and Chunker
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.chunking import (
    RecursiveCharacterChunker,
    SemanticChunker,
    SlidingWindowChunker,
    ChunkerFactory,
)
from app.core.retriever import BM25


# ── Chunking ──────────────────────────────────────────────────────────────────

SAMPLE_TEXT = """
Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval
with text generation. It first retrieves relevant documents from a knowledge base,
then uses them as context for an LLM to generate accurate responses.

RAG significantly reduces hallucinations by grounding the model's outputs in factual
retrieved content. This makes it ideal for enterprise knowledge management, customer
support, and document Q&A systems.

The core components of a RAG system are: a document ingestion pipeline, a vector store
for semantic search, and an LLM for response generation. Advanced RAG systems also
include hybrid retrieval, reranking, and metadata filtering.
""".strip()


class TestRecursiveCharacterChunker:
    def setup_method(self):
        self.chunker = RecursiveCharacterChunker(chunk_size=300, chunk_overlap=50)

    def test_produces_chunks(self):
        chunks = self.chunker.chunk(SAMPLE_TEXT)
        assert len(chunks) > 0

    def test_chunk_fields(self):
        chunks = self.chunker.chunk(SAMPLE_TEXT)
        for chunk in chunks:
            assert chunk.content
            assert chunk.chunk_index >= 0
            assert chunk.token_count > 0

    def test_chunks_cover_text(self):
        chunks = self.chunker.chunk(SAMPLE_TEXT)
        combined = " ".join(c.content for c in chunks)
        # All words from original should appear somewhere in chunks
        for word in SAMPLE_TEXT.split()[:10]:
            assert word in combined

    def test_metadata_passed_through(self):
        chunks = self.chunker.chunk(SAMPLE_TEXT, metadata={"source": "test"})
        for chunk in chunks:
            assert chunk.metadata["source"] == "test"


class TestSemanticChunker:
    def setup_method(self):
        self.chunker = SemanticChunker(max_tokens=150, overlap_sentences=1)

    def test_produces_chunks(self):
        chunks = self.chunker.chunk(SAMPLE_TEXT)
        assert len(chunks) > 0

    def test_no_empty_chunks(self):
        chunks = self.chunker.chunk(SAMPLE_TEXT)
        for chunk in chunks:
            assert chunk.content.strip()


class TestSlidingWindowChunker:
    def setup_method(self):
        self.chunker = SlidingWindowChunker(window_size=200, step_size=150)

    def test_produces_chunks(self):
        chunks = self.chunker.chunk(SAMPLE_TEXT)
        assert len(chunks) > 0

    def test_overlapping_windows(self):
        chunks = self.chunker.chunk(SAMPLE_TEXT)
        if len(chunks) > 1:
            # Consecutive windows should overlap
            end_first = chunks[0].end_char
            start_second = chunks[1].start_char
            assert start_second < end_first


class TestChunkerFactory:
    def test_creates_recursive(self):
        chunker = ChunkerFactory.create("recursive")
        assert isinstance(chunker, RecursiveCharacterChunker)

    def test_creates_semantic(self):
        chunker = ChunkerFactory.create("semantic")
        assert isinstance(chunker, SemanticChunker)

    def test_creates_sliding_window(self):
        chunker = ChunkerFactory.create("sliding_window")
        assert isinstance(chunker, SlidingWindowChunker)

    def test_raises_on_unknown(self):
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            ChunkerFactory.create("nonexistent")


# ── BM25 ──────────────────────────────────────────────────────────────────────

class TestBM25:
    def setup_method(self):
        self.bm25 = BM25()
        self.bm25.add_documents(
            chunk_ids=["c1", "c2", "c3"],
            texts=[
                "Retrieval-Augmented Generation combines retrieval with LLM generation",
                "Vector databases store high-dimensional embeddings for semantic search",
                "FAISS is an efficient library for similarity search in dense vectors",
            ],
            metadatas=[{"chunk_id": "c1"}, {"chunk_id": "c2"}, {"chunk_id": "c3"}],
        )

    def test_returns_results(self):
        results = self.bm25.search("retrieval generation", top_k=3)
        assert len(results) > 0

    def test_scores_normalized(self):
        results = self.bm25.search("vector embeddings", top_k=3)
        for _, score, _ in results:
            assert 0.0 <= score <= 1.0

    def test_relevant_chunk_ranked_first(self):
        results = self.bm25.search("FAISS similarity search", top_k=3)
        top_chunk_id = results[0][2]["chunk_id"]
        assert top_chunk_id == "c3"

    def test_empty_corpus_returns_empty(self):
        bm25 = BM25()
        results = bm25.search("any query", top_k=5)
        assert results == []
