"""
Embedding Pipeline
Uses sentence-transformers locally — completely FREE, no API key needed.
Model: all-MiniLM-L6-v2 (384 dimensions, fast & accurate)
"""

import hashlib
from typing import Dict, List, Optional

from app.config import settings
from app.utils.helpers import chunk_list
from app.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingPipeline:
    """
    Local embedding pipeline using sentence-transformers.
    Runs entirely on CPU — no external API or key required.
    """

    def __init__(
        self,
        model_name: str = settings.embedding_model,
        batch_size: int = settings.embedding_batch_size,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None
        self._cache: Dict[str, List[float]] = {}

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model", model=self.model_name)
            self._model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
        return self._model

    def _cache_key(self, text: str) -> str:
        return hashlib.md5(f"{self.model_name}::{text}".encode()).hexdigest()

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts using local sentence-transformers."""
        model = self._load_model()

        results: List[Optional[List[float]]] = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        # Check cache
        for i, text in enumerate(texts):
            key = self._cache_key(text)
            if key in self._cache:
                results[i] = self._cache[key]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        # Embed uncached texts in batches
        if uncached_texts:
            all_embeddings = []
            for batch in chunk_list(uncached_texts, self.batch_size):
                embeddings = model.encode(batch, convert_to_numpy=True)
                all_embeddings.extend(embeddings.tolist())

            for idx, embedding in zip(uncached_indices, all_embeddings):
                key = self._cache_key(texts[idx])
                self._cache[key] = embedding
                results[idx] = embedding

            logger.info(
                "Embeddings generated",
                count=len(uncached_texts),
                model=self.model_name,
                cache_hits=len(texts) - len(uncached_texts),
            )

        return [r for r in results if r is not None]

    async def embed_query(self, query: str) -> List[float]:
        """Embed a single query string."""
        embeddings = await self.embed_texts([query])
        return embeddings[0]

    def clear_cache(self) -> None:
        self._cache.clear()
        logger.info("Embedding cache cleared")

    @property
    def cache_size(self) -> int:
        return len(self._cache)


# Singleton instance
_pipeline: Optional[EmbeddingPipeline] = None


def get_embedding_pipeline() -> EmbeddingPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = EmbeddingPipeline()
    return _pipeline
