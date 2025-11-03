# neurosurfer/services/rag_orchestrator.py
"""
RAG Orchestrator Service Module
================================

This module provides production-ready RAG (Retrieval-Augmented Generation) orchestration
for the Neurosurfer server. It manages the complete RAG workflow including file ingestion,
context retrieval, and relevance gating.

The RAGOrchestrator handles:
    - Thread-scoped vector collections (one per user/thread)
    - Base64 file ingestion from requests
    - Smart context retrieval with relevance scoring
    - Automatic query augmentation with retrieved context
    - Configurable similarity thresholds
    - LRU caching of vector stores for performance

Key Features:
    - Per-thread isolation: Each conversation has its own vector collection
    - Smart retrieval: Only augments queries when relevant context exists
    - Relevance gating: Filters low-quality matches
    - Flexible thresholds: Different thresholds for explicit vs implicit RAG
    - Efficient caching: Vector stores cached for reuse

Workflow:
    1. Ingest files (if provided in request)
    2. Check if thread has documents
    3. Retrieve relevant context with similarity scoring
    4. Gate context based on relevance threshold
    5. Augment query with context if relevant
    6. Return RAGResult with augmented query or original

Example:
    >>> orchestrator = RAGOrchestrator(
    ...     embedder=embedder,
    ...     persist_dir="./vector_store",
    ...     top_k=8,
    ...     min_top_sim_default=0.35
    ... )
    >>> 
    >>> result = orchestrator.apply(
    ...     actor_id=1,
    ...     thread_id=42,
    ...     user_query="What is the main topic?",
    ...     files=[{"name": "doc.pdf", "content": "base64..."}]
    ... )
    >>> 
    >>> if result.used:
    ...     print("Using RAG context")
    ...     print(result.augmented_query)
"""
from __future__ import annotations
import os, re, base64, tempfile
from functools import lru_cache
from typing import Optional, Tuple, List, Dict, Any

from neurosurfer.rag.chunker import Chunker
from neurosurfer.rag.filereader import FileReader
from neurosurfer.rag.ingestor import RAGIngestor
from neurosurfer.vectorstores.chroma import ChromaVectorStore
from neurosurfer.vectorstores.base import BaseVectorDB
from neurosurfer.models.embedders.base import BaseEmbedder

import neurosurfer.config as config


class RAGResult:
    """
    Container for RAG operation results.
    
    Holds the outcome of a RAG retrieval operation, including whether
    RAG was used, the augmented query, and metadata about the retrieval.
    
    Attributes:
        used (bool): Whether RAG context was used (passed relevance gate)
        augmented_query (str): Query with context if used=True, original if used=False
        meta (Dict[str, Any]): Metadata about retrieval (scores, reason, etc.)
    
    Example:
        >>> result = RAGResult(
        ...     used=True,
        ...     augmented_query="Query\\n\\n[CONTEXT]\\nRelevant info\\n[/CONTEXT]",
        ...     meta={"top_sim": 0.85, "chunks_retrieved": 5}
        ... )
    """
    def __init__(self, used: bool, augmented_query: str, meta: Dict[str, Any]):
        self.used = used
        self.augmented_query = augmented_query
        self.meta = meta

class RAGOrchestrator:
    """
    Production-ready RAG orchestrator with thread-scoped collections.
    
    Manages the complete RAG workflow: file ingestion, context retrieval,
    relevance gating, and query augmentation. Each user/thread combination
    gets its own isolated vector collection.
    
    The orchestrator implements smart retrieval with configurable thresholds:
    - High threshold for implicit RAG (user didn't explicitly request context)
    - Lower threshold for explicit RAG (user uploaded files or asked about them)
    - Minimum similarity filter to exclude irrelevant chunks
    
    Attributes:
        embedder (BaseEmbedder): Embedding model for vectorization
        chunker (Chunker): Document chunker
        file_reader (FileReader): File reader for various formats
        persist_dir (str): Directory for vector store persistence
        max_context_tokens (int): Maximum tokens for retrieved context
        top_k (int): Number of chunks to retrieve
        min_top_sim_default (float): Minimum similarity for implicit RAG
        min_top_sim_when_explicit (float): Minimum similarity for explicit RAG
        min_sim_to_keep (float): Minimum similarity to include a chunk
        logger: Logger instance
    
    Example:
        >>> orchestrator = RAGOrchestrator(
        ...     embedder=SentenceTransformerEmbedder(),
        ...     persist_dir="./chroma_storage",
        ...     top_k=8,
        ...     min_top_sim_default=0.35
        ... )
        >>> 
        >>> # Process request with files
        >>> result = orchestrator.apply(
        ...     actor_id=user.id,
        ...     thread_id=thread.id,
        ...     user_query="Summarize the document",
        ...     files=[{"name": "doc.pdf", "content": "base64_data", "type": "application/pdf"}]
        ... )
        >>> 
        >>> if result.used:
        ...     # RAG context was relevant
        ...     llm_response = llm.ask(result.augmented_query)
    """
    def __init__(
        self,
        *,
        embedder: BaseEmbedder,
        chunker: Optional[Chunker] = None,
        file_reader: Optional[FileReader] = None,
        persist_dir: Optional[str] = None,
        max_context_tokens: int = 1024,
        top_k: int = 8,
        min_top_sim_default: float = 0.35,
        min_top_sim_when_explicit: float = 0.15,
        min_sim_to_keep: float = 0.20,
        logger=None,
    ):
        """
        Initialize RAG orchestrator.
        
        Args:
            embedder (BaseEmbedder): Embedding model for text vectorization
            chunker (Optional[Chunker]): Document chunker. Default: Chunker()
            file_reader (Optional[FileReader]): File reader. Default: FileReader()
            persist_dir (Optional[str]): Vector store directory. Default: from config
            max_context_tokens (int): Max tokens for context. Default: 1024
            top_k (int): Number of chunks to retrieve. Default: 8
            min_top_sim_default (float): Min similarity for implicit RAG. Default: 0.35
            min_top_sim_when_explicit (float): Min similarity for explicit RAG. Default: 0.15
            min_sim_to_keep (float): Min similarity to include chunk. Default: 0.20
            logger: Logger instance. Default: None
        """
        self.embedder = embedder
        self.chunker = chunker or Chunker()
        self.file_reader = file_reader or FileReader()
        self.persist_dir = persist_dir or config.VECTOR_STORE_STORAGE_PATH
        self.max_context_tokens = max_context_tokens
        self.top_k = top_k
        self.min_top_sim_default = min_top_sim_default
        self.min_top_sim_when_explicit = min_top_sim_when_explicit
        self.min_sim_to_keep = min_sim_to_keep
        self.logger = logger

        os.makedirs(self.persist_dir, exist_ok=True)

    # ---------- public API ----------
    def apply(
        self,
        *,
        actor_id: int,
        thread_id: int,
        user_query: str,
        files: Optional[List[Dict[str, Any]]] = None,
    ) -> RAGResult:
        """
        Apply RAG to a user query.
        
        Main entry point for RAG processing. Ingests files (if provided),
        retrieves relevant context, applies relevance gating, and returns
        augmented query if context is relevant.
        
        Args:
            actor_id (int): User ID for collection scoping
            thread_id (int): Thread ID for collection scoping
            user_query (str): User's question or prompt
            files (Optional[List[Dict[str, Any]]]): Files to ingest.
                Each file dict should have: name, content (base64), type (mime)
        
        Returns:
            RAGResult: Result with used flag, augmented_query, and metadata
        
        Example:
            >>> result = orchestrator.apply(
            ...     actor_id=1,
            ...     thread_id=42,
            ...     user_query="What does the document say about AI?",
            ...     files=[{"name": "ai.pdf", "content": "...", "type": "application/pdf"}]
            ... )
            >>> if result.used:
            ...     print(f"Top similarity: {result.meta['top_sim']}")
        """
        coll = self._collection_for(actor_id, thread_id)

        if files:
            self._ingest_files(coll, files)

        if not self._collection_has_docs(coll):
            return RAGResult(False, user_query, {"reason": "no_docs"})

        ctx_text, meta = self._retrieve_context_smart(coll, user_query)
        if meta.get("used") and ctx_text:
            aug = f"{user_query}\n\n[CONTEXT]\n{ctx_text}\n[/CONTEXT]"
            return RAGResult(True, aug, meta)
        return RAGResult(False, user_query, meta)

    # ---------- internals ----------
    _ALLOWED_NAME = re.compile(r"[^a-zA-Z0-9._-]+")
    def _safe_collection_name(self, s: str) -> str:
        s = self._ALLOWED_NAME.sub("_", s).strip("._-")
        return s or "nm_default"

    def _collection_for(self, user_id: int, thread_id: int) -> str:
        """
        Generate collection name for user/thread combination.
        
        Args:
            user_id (int): User ID
            thread_id (int): Thread ID
        
        Returns:
            str: Safe collection name (e.g., "nm_u1_t42")
        """
        return self._safe_collection_name(f"nm_u{user_id}_t{thread_id}")

    @lru_cache(maxsize=512)
    def _vs(self, collection: str) -> ChromaVectorStore:
        """
        Get or create vector store for collection (cached).
        
        Uses LRU cache to reuse vector store instances for performance.
        
        Args:
            collection (str): Collection name
        
        Returns:
            ChromaVectorStore: Vector store instance
        """
        return ChromaVectorStore(collection_name=collection, persist_directory=self.persist_dir)

    def _collection_has_docs(self, collection: str) -> bool:
        """
        Check if collection has any documents.
        
        Args:
            collection (str): Collection name
        
        Returns:
            bool: True if collection has documents, False otherwise
        """
        try:
            return (self._vs(collection).count() or 0) > 0
        except Exception:
            return False

    def _ingest_files(self, collection: str, files: List[Dict[str, Any]]) -> None:
        vs = self._vs(collection)
        ing = RAGIngestor(
            embedder=self.embedder,
            vector_store=vs,
            file_reader=self.file_reader,
            chunker=self.chunker,
            batch_size=32,
        )
        tmp_paths: List[str] = []
        try:
            for idx, f in enumerate(files):
                name = f.get("name") or f"upload_{idx}.bin"
                content_b64 = f.get("content") or ""
                if not content_b64:
                    continue
                # write to tmp
                fd, tmp_path = tempfile.mkstemp(prefix="nm_up_", suffix=f"__{name}")
                with os.fdopen(fd, "wb") as out:
                    out.write(base64.b64decode(content_b64))
                tmp_paths.append(tmp_path)

                # base_id = os.path.basename(tmp_path).split("__", 1)[0]
                if name.lower().endswith(".zip"):
                    ing.add_zipfile(zip_path=tmp_path)
                else:
                    ing.add_files(paths=[tmp_path])
            if tmp_paths:
                summary = ing.build()
                if self.logger:
                    self.logger.info(f"[RAG] Ingested chunks: {summary}")
        finally:
            # clean temp files
            for p in tmp_paths:
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass

    def _sim_from_dist(self, d: Optional[float]) -> float:
        if d is None: return 0.0
        return max(-1.0, min(1.0, 1.0 - float(d)))

    _FILE_REF_HINTS = re.compile(
        r"(this (file|doc|document|pdf|zip|spreadsheet|slide|report))|"
        r"(above (file|doc|context))|"
        r"(in the (file|document|pdf))|"
        r"(according to (it|the (file|doc|pdf)))",
        flags=re.IGNORECASE,
    )
    def _explicit_file_reference(self, q: str) -> bool:
        return bool(self._FILE_REF_HINTS.search(q or ""))

    def _retrieve_candidates(self, collection: str, query: str, top_k: int) -> List[Dict[str, Any]]:
        vs = self._vs(collection)
        qvec = self.embedder.embed([query])[0]
        hits = vs.similarity_search(qvec, top_k=top_k)
        cands: List[Dict[str, Any]] = []
        for doc, dist in hits:
            cands.append({
                "id": doc.id,
                "text": doc.text,
                "metadata": getattr(doc, "metadata", None),
                "distance": float(dist or 0.0),
                "similarity": self._sim_from_dist(dist),
            })
        cands.sort(key=lambda x: x["similarity"], reverse=True)
        return cands

    def _approx_token_len(self, text: str) -> int:
        return max(1, len(text) // 4)

    def _build_context(self, candidates: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        picked: List[Dict[str, Any]] = []
        acc = 0
        for c in candidates:
            if c["similarity"] < self.min_sim_to_keep:
                continue
            t = c["text"]
            tl = self._approx_token_len(t)
            if acc + tl > self.max_context_tokens:
                rem = self.max_context_tokens - acc
                if rem > 64:
                    approx_chars = rem * 4
                    c = dict(c)
                    c["text"] = t[:approx_chars]
                    picked.append(c)
                break
            picked.append(c)
            acc += tl
        ctx = "\n\n".join(p["text"] for p in picked)
        return ctx, picked

    def _decide_use(self, query: str, candidates: List[Dict[str, Any]]) -> bool:
        if not candidates:
            return False
        top = candidates[0]["similarity"]
        explicit = self._explicit_file_reference(query)
        threshold = self.min_top_sim_when_explicit if explicit else self.min_top_sim_default
        return top >= threshold

    def _retrieve_context_smart(self, collection: str, query: str) -> Tuple[str, Dict[str, Any]]:
        cands = self._retrieve_candidates(collection, query, self.top_k)
        if not self._decide_use(query, cands):
            top_sim = cands[0]["similarity"] if cands else 0.0
            return "", {"used": False, "reason": "low_similarity", "top_similarity": top_sim,
                        "candidates": [{k:v for k,v in c.items() if k != "text"} for c in cands]}
        ctx, picked = self._build_context(cands)
        return ctx, {"used": True, "reason": "ok",
                     "top_similarity": cands[0]["similarity"] if cands else 0.0,
                     "picked_count": len(picked),
                     "candidates": [{k:v for k,v in c.items() if k != "text"} for c in cands]}
