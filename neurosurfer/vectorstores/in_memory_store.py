import math
from typing import List, Tuple, Optional, Dict, Any
from .base import BaseVectorDB, Doc


class InMemoryVectorStore(BaseVectorDB):
    """Minimal, production-friendly baseline. Replace with FAISS, PGVecto, Chroma, Milvus, etc."""
    def __init__(self, dim: int):
        self.dim = dim
        self._docs: List[Doc] = []
        self._vecs: List[List[float]] = []

    def add_documents(self, docs: List[Doc]) -> None:
        for d in docs:
            if d.embedding is None:
                raise ValueError("Doc missing embedding")
            if len(d.embedding) != self.dim:
                raise ValueError(f"Embedding dim {len(d.embedding)} != store dim {self.dim}")
            self._docs.append(d)
            self._vecs.append(d.embedding)

    def similarity_search(
        self, 
        query_embedding: List[float],
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None
    ) -> List[Tuple[Doc, float]]:
        # cosine similarity
        def dot(a, b): return sum(x*y for x, y in zip(a, b))
        def norm(a): return math.sqrt(sum(x*x for x in a)) or 1e-9

        qn = norm(query_embedding)
        sims: List[Tuple[int, float]] = []
        for i, v in enumerate(self._vecs):
            sims.append((i, dot(query_embedding, v) / (qn * norm(v))))
        sims.sort(key=lambda x: x[1], reverse=True)
        result: List[Tuple[Doc, float]] = []
        for idx, score in sims[:top_k]:
            result.append((self._docs[idx], score))
        return result

    def count(self) -> int:
        return len(self._docs)

    def list_all_documents(
        self,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Doc]:
        return self._docs

    def delete_collection(self):
        self._docs = []
        self._vecs = []
    
    def clear_collection(self):
        self._docs = []
        self._vecs = []