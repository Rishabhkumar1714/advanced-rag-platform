"""
FAISS Vector Store
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.config import settings
from app.utils.helpers import ensure_directory

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    INDEX_FILE = "index.faiss"
    METADATA_FILE = "metadata.pkl"

    def __init__(self, index_path: str = settings.faiss_index_path, dimension: int = settings.faiss_index_dimension):
        self.index_path = Path(index_path)
        self.dimension = dimension
        self._index = None
        self._metadata: Dict[int, Dict[str, Any]] = {}
        self._chunk_id_to_faiss: Dict[str, int] = {}
        self._next_id: int = 0
        self._initialize_index()

    def _initialize_index(self) -> None:
        try:
            import faiss
            self._index = faiss.IndexFlatL2(self.dimension)
            logger.info(f"FAISS index initialized: dimension={self.dimension}")
        except ImportError:
            raise ImportError("faiss-cpu is required")

    def save(self) -> None:
        try:
            import faiss
            ensure_directory(str(self.index_path))
            faiss.write_index(self._index, str(self.index_path / self.INDEX_FILE))
            with open(self.index_path / self.METADATA_FILE, "wb") as f:
                pickle.dump({"metadata": self._metadata, "chunk_id_to_faiss": self._chunk_id_to_faiss, "next_id": self._next_id}, f)
            logger.info(f"FAISS index saved: vectors={self._next_id}")
        except Exception as e:
            logger.warning(f"Could not save FAISS index: {e}")

    def load(self) -> bool:
        try:
            import faiss
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
            logger.info(f"FAISS index loaded: vectors={self._next_id}")
            return True
        except Exception as e:
            logger.warning(f"Could not load FAISS index: {e}")
            return False

    def add_embeddings(self, chunk_ids: List[str], embeddings: List[List[float]], metadatas: List[Dict[str, Any]]) -> List[int]:
        if not chunk_ids:
            return []
        vectors = np.array(embeddings, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        faiss_ids = list(range(self._next_id, self._next_id + len(chunk_ids)))
        self._index.add(vectors)
        for chunk_id, faiss_id, meta in zip(chunk_ids, faiss_ids, metadatas):
            self._metadata[faiss_id] = meta
            self._chunk_id_to_faiss[chunk_id] = faiss_id
        self._next_id += len(chunk_ids)
        logger.info(f"Added {len(chunk_ids)} embeddings, total={self._next_id}")
        return faiss_ids

    def search(self, query_embedding: List[float], top_k: int = settings.retrieval_top_k, filters: Optional[Dict[str, Any]] = None) -> List[Tuple[str, float, Dict[str, Any]]]:
        if self._next_id == 0:
            return []
        fetch_k = min(top_k * 10 if filters else top_k, self._next_id)
        query_vec = np.array([query_embedding], dtype=np.float32)
        distances, indices = self._index.search(query_vec, fetch_k)
        results: List[Tuple[str, float, Dict[str, Any]]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            meta = self._metadata.get(int(idx), {})
            if filters and not self._match_filters(meta, filters):
                continue
            score = float(1.0 / (1.0 + dist))
            chunk_id = meta.get("chunk_id", str(idx))
            results.append((chunk_id, score, meta))
            if len(results) >= top_k:
                break
        return results

    def delete_by_document_id(self, document_id: str) -> int:
        to_delete = [(cid, fid) for cid, fid in self._chunk_id_to_faiss.items()
                     if self._metadata.get(fid, {}).get("document_id") == document_id]
        for chunk_id, faiss_id in to_delete:
            self._metadata.pop(faiss_id, None)
            self._chunk_id_to_faiss.pop(chunk_id, None)
        logger.info(f"Deleted {len(to_delete)} chunks for document_id={document_id}")
        return len(to_delete)

    @staticmethod
    def _match_filters(meta: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        return all(meta.get(k) == v for k, v in filters.items())

    @property
    def total_vectors(self) -> int:
        return self._next_id


_vector_store: Optional[FAISSVectorStore] = None

def get_vector_store() -> FAISSVectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = FAISSVectorStore()
        _vector_store.load()
    return _vector_store
