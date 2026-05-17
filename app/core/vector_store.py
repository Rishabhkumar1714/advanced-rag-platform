"""
FAISS Vector Store
Manages the FAISS index for storing and retrieving document embeddings.
Supports both flat (exact) and IVF (approximate) indexes with metadata filtering.
"""

import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.config import settings
from app.utils.helpers import ensure_directory
from app.utils.logger import get_logger

logger = get_logger(__name__)


class FAISSVectorStore:
    """
    Thread-safe FAISS vector store with:
    - Flat L2 index for exact search (default)
    - IVF index for approximate large-scale search
    - Metadata store (chunk_id → metadata)
    - Persistence (save / load from disk)
    """

    INDEX_FILE = "index.faiss"
    METADATA_FILE = "metadata.pkl"
    CONFIG_FILE = "config.json"

    def __init__(
        self,
        index_path: str = settings.faiss_index_path,
        dimension: int = settings.faiss_index_dimension,
        use_ivf: bool = False,
        n_lists: int = 100,
    ):
        self.index_path = Path(index_path)
        self.dimension = dimension
        self.use_ivf = use_ivf
        self.n_lists = n_lists

        self._index = None
        self._metadata: Dict[int, Dict[str, Any]] = {}   # faiss_id → metadata
        self._chunk_id_to_faiss: Dict[str, int] = {}     # chunk_id → faiss_id
        self._next_id: int = 0

        self._initialize_index()

    # ── Index Lifecycle ───────────────────────────────────────────────────────

    def _initialize_index(self) -> None:
        try:
            import faiss  # type: ignore
        except ImportError:
            raise ImportError("faiss-cpu is required: pip install faiss-cpu")

        if self.use_ivf:
            quantizer = faiss.IndexFlatL2(self.dimension)
            self._index = faiss.IndexIVFFlat(
                quantizer, self.dimension, self.n_lists, faiss.METRIC_L2
            )
        else:
            self._index = faiss.IndexFlatL2(self.dimension)

        logger.info(
            "FAISS index initialized",
            dimension=self.dimension,
            index_type="IVFFlat" if self.use_ivf else "FlatL2",
        )

    def save(self) -> None:
        """Persist the index and metadata to disk."""
        import faiss  # type: ignore

        ensure_directory(str(self.index_path))
        faiss.write_index(self._index, str(self.index_path / self.INDEX_FILE))
        with open(self.index_path / self.METADATA_FILE, "wb") as f:
            pickle.dump(
                {
                    "metadata": self._metadata,
                    "chunk_id_to_faiss": self._chunk_id_to_faiss,
                    "next_id": self._next_id,
                },
                f,
            )
        config = {"dimension": self.dimension, "use_ivf": self.use_ivf}
        with open(self.index_path / self.CONFIG_FILE, "w") as f:
            json.dump(config, f)

        logger.info("FAISS index saved", path=str(self.index_path), vectors=self._next_id)

    def load(self) -> bool:
        """Load index and metadata from disk. Returns True if successful."""
        import faiss  # type: ignore

        index_file = self.index_path / self.INDEX_FILE
        meta_file = self.index_path / self.METADATA_FILE

        if not index_file.exists() or not meta_file.exists():
            logger.info("No existing index found, starting fresh")
            return False

        self._index = faiss.read_index(str(index_file))
        with open(meta_file, "rb") as f:
            data = pickle.load(f)
        self._metadata = data["metadata"]
        self._chunk_id_to_faiss = data["chunk_id_to_faiss"]
        self._next_id = data["next_id"]

        logger.info(
            "FAISS index loaded",
            path=str(self.index_path),
            vectors=self._next_id,
        )
        return True

    # ── CRUD Operations ───────────────────────────────────────────────────────

    def add_embeddings(
        self,
        chunk_ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
    ) -> List[int]:
        """Add a batch of embeddings to the index."""
        if not chunk_ids:
            return []

        vectors = np.array(embeddings, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        faiss_ids = list(range(self._next_id, self._next_id + len(chunk_ids)))
        self._index.add(vectors)  # type: ignore

        for chunk_id, faiss_id, meta in zip(chunk_ids, faiss_ids, metadatas):
            self._metadata[faiss_id] = meta
            self._chunk_id_to_faiss[chunk_id] = faiss_id

        self._next_id += len(chunk_ids)
        logger.info("Embeddings added", count=len(chunk_ids), total=self._next_id)
        return faiss_ids

    def search(
        self,
        query_embedding: List[float],
        top_k: int = settings.retrieval_top_k,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search the index and return (chunk_id, score, metadata) tuples.
        Scores are normalized to [0, 1] (1 = most similar).
        """
        if self._next_id == 0:
            return []

        # Fetch more candidates when filtering
        fetch_k = top_k * 10 if filters else top_k
        fetch_k = min(fetch_k, self._next_id)

        query_vec = np.array([query_embedding], dtype=np.float32)
        distances, indices = self._index.search(query_vec, fetch_k)  # type: ignore

        results: List[Tuple[str, float, Dict[str, Any]]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            meta = self._metadata.get(int(idx), {})
            if filters and not self._match_filters(meta, filters):
                continue
            # Convert L2 distance to similarity score
            score = float(1.0 / (1.0 + dist))
            chunk_id = meta.get("chunk_id", str(idx))
            results.append((chunk_id, score, meta))
            if len(results) >= top_k:
                break

        return results

    def delete_by_document_id(self, document_id: str) -> int:
        """Remove all chunks belonging to a document. Returns count deleted."""
        to_delete = [
            (chunk_id, faiss_id)
            for chunk_id, faiss_id in self._chunk_id_to_faiss.items()
            if self._metadata.get(faiss_id, {}).get("document_id") == document_id
        ]
        for chunk_id, faiss_id in to_delete:
            self._metadata.pop(faiss_id, None)
            self._chunk_id_to_faiss.pop(chunk_id, None)

        logger.info("Chunks deleted", document_id=document_id, count=len(to_delete))
        return len(to_delete)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _match_filters(meta: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        for key, value in filters.items():
            if meta.get(key) != value:
                return False
        return True

    @property
    def total_vectors(self) -> int:
        return self._next_id

    def is_trained(self) -> bool:
        if self.use_ivf:
            return bool(self._index.is_trained)  # type: ignore
        return True


# Singleton
_vector_store: Optional[FAISSVectorStore] = None


def get_vector_store() -> FAISSVectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = FAISSVectorStore()
        _vector_store.load()
    return _vector_store
