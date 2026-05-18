"""
Hybrid Retriever
Combines semantic (FAISS) and keyword (BM25) retrieval with
optional cross-encoder reranking for maximum relevance.
"""

import math
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from app.config import settings
from app.core.embeddings import get_embedding_pipeline
from app.core.vector_store import get_vector_store
from app.models.query import RetrievalMode, SearchFilter, SourceDocument
import logging

logger = logging.getLogger(__name__)


# ── BM25 Implementation ───────────────────────────────────────────────────────

class BM25:
    """
    Okapi BM25 for keyword-based retrieval over the in-memory document corpus.
    k1 and b are standard BM25 hyperparameters.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._corpus: List[str] = []
        self._chunk_ids: List[str] = []
        self._metadatas: List[Dict[str, Any]] = []
        self._tf: List[Dict[str, float]] = []
        self._df: Dict[str, int] = defaultdict(int)
        self._idf: Dict[str, float] = {}
        self._avgdl: float = 0.0

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def add_documents(
        self,
        chunk_ids: List[str],
        texts: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        for chunk_id, text, meta in zip(chunk_ids, texts, metadatas):
            tokens = self._tokenize(text)
            tf: Dict[str, float] = defaultdict(float)
            for token in tokens:
                tf[token] += 1.0
            self._tf.append(dict(tf))
            for token in set(tokens):
                self._df[token] += 1
            self._corpus.append(text)
            self._chunk_ids.append(chunk_id)
            self._metadatas.append(meta)

        n = len(self._corpus)
        total_len = sum(len(self._tokenize(t)) for t in self._corpus)
        self._avgdl = total_len / n if n > 0 else 0.0
        self._idf = {
            term: math.log((n - df + 0.5) / (df + 0.5) + 1)
            for term, df in self._df.items()
        }

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float, Dict[str, Any]]]:
        if not self._corpus:
            return []

        query_tokens = self._tokenize(query)
        scores: List[float] = []

        for i, tf in enumerate(self._tf):
            doc_len = sum(tf.values())
            score = 0.0
            for token in query_tokens:
                if token not in tf:
                    continue
                idf = self._idf.get(token, 0.0)
                freq = tf[token]
                score += idf * (
                    freq * (self.k1 + 1)
                    / (freq + self.k1 * (1 - self.b + self.b * doc_len / self._avgdl))
                )
            scores.append(score)

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        # Normalize scores to [0, 1]
        max_score = ranked[0][1] if ranked and ranked[0][1] > 0 else 1.0
        return [
            (self._chunk_ids[i], score / max_score, self._metadatas[i])
            for i, score in ranked
            if score > 0
        ]


# ── Hybrid Retriever ──────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Combines semantic and keyword retrieval using Reciprocal Rank Fusion (RRF).
    Optionally applies cross-encoder reranking.
    """

    RRF_K = 60  # RRF constant

    def __init__(self):
        self._bm25 = BM25()
        self._embedding_pipeline = get_embedding_pipeline()
        self._vector_store = get_vector_store()

    def index_chunks(
        self,
        chunk_ids: List[str],
        texts: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """Add chunks to the BM25 index (vector store is updated separately)."""
        self._bm25.add_documents(chunk_ids, texts, metadatas)

    async def retrieve(
        self,
        query: str,
        top_k: int = settings.retrieval_top_k,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        alpha: Optional[float] = None,
        filters: Optional[List[SearchFilter]] = None,
        rerank: bool = settings.reranker_enabled,
    ) -> List[SourceDocument]:
        """
        Main retrieval method.
        Returns a list of SourceDocument sorted by relevance.
        """
        alpha = alpha if alpha is not None else settings.hybrid_search_alpha
        filter_dict = self._build_filter_dict(filters or [])

        semantic_results: List[Tuple[str, float, Dict]] = []
        keyword_results: List[Tuple[str, float, Dict]] = []

        if mode in (RetrievalMode.SEMANTIC, RetrievalMode.HYBRID):
            query_embedding = await self._embedding_pipeline.embed_query(query)
            semantic_results = self._vector_store.search(
                query_embedding, top_k=top_k * 2, filters=filter_dict
            )

        if mode in (RetrievalMode.KEYWORD, RetrievalMode.HYBRID):
            keyword_results = self._bm25.search(query, top_k=top_k * 2)
            if filter_dict:
                keyword_results = [
                    (cid, score, meta)
                    for cid, score, meta in keyword_results
                    if all(meta.get(k) == v for k, v in filter_dict.items())
                ]

        # Fuse rankings
        if mode == RetrievalMode.HYBRID:
            fused = self._rrf_fuse(semantic_results, keyword_results, alpha)
        elif mode == RetrievalMode.SEMANTIC:
            fused = semantic_results
        else:
            fused = keyword_results

        fused = fused[:top_k]

        # Optionally rerank
        if rerank and len(fused) > 1:
            fused = await self._rerank(query, fused)

        sources = [
            SourceDocument(
                chunk_id=chunk_id,
                document_id=meta.get("document_id", ""),
                title=meta.get("title", "Untitled"),
                content=meta.get("content", ""),
                score=round(score, 4),
                chunk_index=meta.get("chunk_index", 0),
                source=meta.get("source"),
                metadata={k: v for k, v in meta.items() if k not in ("content",)},
            )
            for chunk_id, score, meta in fused
        ]

        logger.info(
            "Retrieval complete",
            query_preview=query[:60],
            mode=mode,
            results=len(sources),
        )
        return sources

    def _rrf_fuse(
        self,
        semantic: List[Tuple[str, float, Dict]],
        keyword: List[Tuple[str, float, Dict]],
        alpha: float,
    ) -> List[Tuple[str, float, Dict]]:
        """
        Reciprocal Rank Fusion with alpha weighting.
        alpha=1 → pure semantic, alpha=0 → pure keyword.
        """
        rrf_scores: Dict[str, float] = defaultdict(float)
        meta_map: Dict[str, Dict] = {}

        for rank, (chunk_id, _, meta) in enumerate(semantic):
            rrf_scores[chunk_id] += alpha * (1.0 / (self.RRF_K + rank + 1))
            meta_map[chunk_id] = meta

        for rank, (chunk_id, _, meta) in enumerate(keyword):
            rrf_scores[chunk_id] += (1 - alpha) * (1.0 / (self.RRF_K + rank + 1))
            meta_map.setdefault(chunk_id, meta)

        sorted_ids = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)
        max_score = rrf_scores[sorted_ids[0]] if sorted_ids else 1.0

        return [
            (cid, rrf_scores[cid] / max_score, meta_map[cid])
            for cid in sorted_ids
        ]

    async def _rerank(
        self,
        query: str,
        candidates: List[Tuple[str, float, Dict]],
    ) -> List[Tuple[str, float, Dict]]:
        """
        Lightweight reranker using embedding cosine similarity between the
        query and each candidate chunk.  In production, swap in a cross-encoder
        (e.g. cross-encoder/ms-marco-MiniLM-L-6-v2) via sentence-transformers.
        """
        candidate_texts = [meta.get("content", "") for _, _, meta in candidates]
        all_texts = [query] + candidate_texts
        embeddings = await self._embedding_pipeline.embed_texts(all_texts)

        query_emb = embeddings[0]
        reranked: List[Tuple[str, float, Dict]] = []
        for i, (chunk_id, _, meta) in enumerate(candidates):
            cand_emb = embeddings[i + 1]
            dot = sum(a * b for a, b in zip(query_emb, cand_emb))
            norm_q = sum(a ** 2 for a in query_emb) ** 0.5
            norm_c = sum(a ** 2 for a in cand_emb) ** 0.5
            cosine = dot / (norm_q * norm_c + 1e-9)
            reranked.append((chunk_id, float(cosine), meta))

        reranked.sort(key=lambda x: x[1], reverse=True)

        top_n = settings.reranker_top_n
        logger.info("Reranking complete", top_n=top_n)
        return reranked[:top_n]

    @staticmethod
    def _build_filter_dict(filters: List[SearchFilter]) -> Dict[str, Any]:
        """Convert SearchFilter list to a simple equality dict (basic impl)."""
        return {f.field: f.value for f in filters if f.operator == "eq"}


# Singleton
_retriever: Optional[HybridRetriever] = None


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever
