"""
Hybrid Retriever - BM25 + FAISS with Reciprocal Rank Fusion
"""

import logging
import math
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from app.config import settings
from app.core.embeddings import get_embedding_pipeline
from app.core.vector_store import get_vector_store
from app.models.query import RetrievalMode, SearchFilter, SourceDocument

logger = logging.getLogger(__name__)


class BM25:
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

    def add_documents(self, chunk_ids: List[str], texts: List[str], metadatas: List[Dict[str, Any]]) -> None:
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
        self._idf = {term: math.log((n - df + 0.5) / (df + 0.5) + 1) for term, df in self._df.items()}

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
                score += idf * (freq * (self.k1 + 1) / (freq + self.k1 * (1 - self.b + self.b * doc_len / self._avgdl)))
            scores.append(score)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        max_score = ranked[0][1] if ranked and ranked[0][1] > 0 else 1.0
        return [(self._chunk_ids[i], score / max_score, self._metadatas[i]) for i, score in ranked if score > 0]


class HybridRetriever:
    RRF_K = 60

    def __init__(self):
        self._bm25 = BM25()
        self._embedding_pipeline = get_embedding_pipeline()
        self._vector_store = get_vector_store()

    def index_chunks(self, chunk_ids: List[str], texts: List[str], metadatas: List[Dict[str, Any]]) -> None:
        self._bm25.add_documents(chunk_ids, texts, metadatas)

    async def retrieve(self, query: str, top_k: int = settings.retrieval_top_k, mode: RetrievalMode = RetrievalMode.HYBRID,
                       alpha: Optional[float] = None, filters: Optional[List[SearchFilter]] = None, rerank: bool = settings.reranker_enabled) -> List[SourceDocument]:
        alpha = alpha if alpha is not None else settings.hybrid_search_alpha
        filter_dict = {f.field: f.value for f in (filters or []) if f.operator == "eq"}

        semantic_results: List[Tuple[str, float, Dict]] = []
        keyword_results: List[Tuple[str, float, Dict]] = []

        if mode in (RetrievalMode.SEMANTIC, RetrievalMode.HYBRID):
            query_embedding = await self._embedding_pipeline.embed_query(query)
            semantic_results = self._vector_store.search(query_embedding, top_k=top_k * 2, filters=filter_dict)

        if mode in (RetrievalMode.KEYWORD, RetrievalMode.HYBRID):
            keyword_results = self._bm25.search(query, top_k=top_k * 2)
            if filter_dict:
                keyword_results = [(cid, s, m) for cid, s, m in keyword_results if all(m.get(k) == v for k, v in filter_dict.items())]

        if mode == RetrievalMode.HYBRID:
            fused = self._rrf_fuse(semantic_results, keyword_results, alpha)
        elif mode == RetrievalMode.SEMANTIC:
            fused = semantic_results
        else:
            fused = keyword_results

        fused = fused[:top_k]

        if rerank and len(fused) > 1:
            fused = await self._rerank(query, fused)

        sources = [
            SourceDocument(
                chunk_id=chunk_id, document_id=meta.get("document_id", ""),
                title=meta.get("title", "Untitled"), content=meta.get("content", ""),
                score=round(score, 4), chunk_index=meta.get("chunk_index", 0),
                source=meta.get("source"), metadata={k: v for k, v in meta.items() if k != "content"},
            )
            for chunk_id, score, meta in fused
        ]
        logger.info(f"Retrieval complete: mode={mode}, results={len(sources)}")
        return sources

    def _rrf_fuse(self, semantic, keyword, alpha):
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
        return [(cid, rrf_scores[cid] / max_score, meta_map[cid]) for cid in sorted_ids]

    async def _rerank(self, query, candidates):
        candidate_texts = [meta.get("content", "") for _, _, meta in candidates]
        all_texts = [query] + candidate_texts
        embeddings = await self._embedding_pipeline.embed_texts(all_texts)
        query_emb = embeddings[0]
        reranked = []
        for i, (chunk_id, _, meta) in enumerate(candidates):
            cand_emb = embeddings[i + 1]
            dot = sum(a * b for a, b in zip(query_emb, cand_emb))
            norm_q = sum(a ** 2 for a in query_emb) ** 0.5
            norm_c = sum(a ** 2 for a in cand_emb) ** 0.5
            cosine = dot / (norm_q * norm_c + 1e-9)
            reranked.append((chunk_id, float(cosine), meta))
        reranked.sort(key=lambda x: x[1], reverse=True)
        logger.info(f"Reranking complete: top_n={settings.reranker_top_n}")
        return reranked[:settings.reranker_top_n]


_retriever: Optional[HybridRetriever] = None

def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever
