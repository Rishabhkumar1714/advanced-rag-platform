"""
Embedding Pipeline
Manages OpenAI embedding generation with batching, caching, and retry logic.
"""

import asyncio
import hashlib
from typing import Dict, List, Optional

import openai
from openai import AsyncOpenAI

from app.config import settings
from app.utils.helpers import chunk_list
from app.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingPipeline:
    """
    Async embedding pipeline that batches text inputs, caches results in-memory,
    and retries on transient OpenAI errors.
    """

    def __init__(
        self,
        model: str = settings.openai_embedding_model,
        batch_size: int = settings.embedding_batch_size,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._cache: Dict[str, List[float]] = {}

    def _cache_key(self, text: str) -> str:
        return hashlib.md5(f"{self.model}::{text}".encode()).hexdigest()

    async def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Call the OpenAI embeddings API for a batch of texts with retries."""
        for attempt in range(1, self.max_retries + 1):
            try:
                response = await self._client.embeddings.create(
                    model=self.model,
                    input=texts,
                    encoding_format="float",
                )
                return [item.embedding for item in response.data]
            except openai.RateLimitError:
                wait = self.retry_delay * (2 ** (attempt - 1))
                logger.warning(
                    "Rate limit hit, retrying",
                    attempt=attempt,
                    wait_seconds=wait,
                )
                await asyncio.sleep(wait)
            except openai.APIStatusError as e:
                logger.error("OpenAI API error", status=e.status_code, message=str(e))
                raise

        raise RuntimeError(f"Embedding failed after {self.max_retries} retries")

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts.  Results for cached inputs are returned
        immediately; uncached inputs are batched and embedded.
        """
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

        # Batch-embed uncached texts
        if uncached_texts:
            all_embeddings: List[List[float]] = []
            for batch in chunk_list(uncached_texts, self.batch_size):
                embeddings = await self._embed_batch(batch)
                all_embeddings.extend(embeddings)

            for idx, embedding in zip(uncached_indices, all_embeddings):
                key = self._cache_key(texts[idx])
                self._cache[key] = embedding
                results[idx] = embedding

            logger.info(
                "Embeddings generated",
                count=len(uncached_texts),
                model=self.model,
                cache_hits=len(texts) - len(uncached_texts),
            )

        return [r for r in results if r is not None]

    async def embed_query(self, query: str) -> List[float]:
        """Embed a single query string."""
        embeddings = await self.embed_texts([query])
        return embeddings[0]

    def clear_cache(self) -> None:
        """Clear the in-memory embedding cache."""
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
