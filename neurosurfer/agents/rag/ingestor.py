# rag_ingestor.py
"""
RAG Ingestor Module
===================

This module provides a production-grade document ingestion pipeline for RAG systems.
The RAGIngestor handles the complete workflow: reading files, chunking, embedding,
and storing in vector databases.

Key Features:
    - Multi-source support: files, directories, raw text, URLs, ZIP archives
    - Parallel processing with configurable workers
    - Batch embedding for efficiency
    - Content-based deduplication
    - Progress tracking with callbacks
    - Cancellation support
    - Automatic file type detection
    - Metadata preservation

Workflow:
    1. Add sources (files, directories, text, URLs)
    2. Read and chunk documents
    3. Generate embeddings in batches
    4. Deduplicate by content hash
    5. Upsert to vector store
    6. Report progress

Example:
    >>> from neurosurfer.rag.ingestor import RAGIngestor
    >>> from neurosurfer.models.embedders import SentenceTransformerEmbedder
    >>> from neurosurfer.vectorstores import ChromaVectorStore
    >>> 
    >>> embedder = SentenceTransformerEmbedder()
    >>> vectorstore = ChromaVectorStore(collection_name="docs")
    >>> 
    >>> ingestor = RAGIngestor(
    ...     embedder=embedder,
    ...     vector_store=vectorstore,
    ...     batch_size=64,
    ...     max_workers=4
    ... )
    >>> 
    >>> # Add files and ingest
    >>> ingestor.add_files(["./docs"])
    >>> ingestor.ingest()
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import io
import logging
import mimetypes
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
import zipfile
import tempfile
import shutil
from typing import (
    Any, Callable, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple, Union
)

from neurosurfer.models.embedders import BaseEmbedder
from neurosurfer.vectorstores.base import BaseVectorDB, Doc

from .filereader import FileReader
from .chunker import Chunker
from .constants import supported_file_types, exclude_dirs_in_code
from .url_fetcher import URLFetcher, URLFetcherConfig, URLFetchResult


ProgressCallback = Callable[[Dict[str, Any]], None]

# --------------------------
# Utility
# --------------------------
def sha256_text(s: str) -> str:
    """Generate SHA256 hash of text string."""
    h = hashlib.sha256()
    h.update(s.encode("utf-8"))
    return h.hexdigest()

def now_ts() -> float:
    """Get current timestamp."""
    return time.time()

# --------------------------
# RAG Ingestor
# --------------------------
class RAGIngestor:
    """
    Production-grade RAG document ingestor.
    
    This class orchestrates the complete document ingestion pipeline:
    reading, chunking, embedding, deduplication, and vector storage.
    
    It supports multiple input sources and provides progress tracking,
    cancellation, and parallel processing capabilities.
    
    Attributes:
        embedder (BaseEmbedder): Embedding model for generating vectors
        vs (BaseVectorDB): Vector store for document storage
        reader (FileReader): File reader for various formats
        chunker (Chunker): Document chunker
        log (logging.Logger): Logger instance
        progress_cb (Optional[ProgressCallback]): Progress callback function
        cancel_event (threading.Event): Event for cancellation
        batch_size (int): Batch size for embedding generation
        max_workers (int): Max parallel workers for processing
        deduplicate (bool): Enable content-based deduplication
        normalize_embeddings (bool): Normalize embedding vectors
    
    Example:
        >>> ingestor = RAGIngestor(
        ...     embedder=embedder,
        ...     vectorstore=vectorstore,
        ...     batch_size=64,
        ...     progress_cb=lambda p: print(f"Progress: {p['percent']:.1f}%")
        ... )
        >>> 
        >>> # Add multiple sources
        >>> ingestor.add_files(["./docs", "./code"])
        >>> ingestor.add_text("Custom content", metadata={"source": "manual"})
        >>> 
        >>> # Ingest all
        >>> stats = ingestor.ingest()
        >>> print(f"Ingested {stats['chunks_added']} chunks")
    """

    def __init__(
        self,
        *,
        embedder: BaseEmbedder,
        vectorstore: BaseVectorDB,
        file_reader: Optional[FileReader] = None,
        chunker: Optional[Chunker] = None,
        logger: Optional[logging.Logger] = None,
        progress_cb: Optional[ProgressCallback] = None,
        cancel_event: Optional[threading.Event] = None,
        batch_size: int = 64,
        max_workers: int = max(4, os.cpu_count() or 4),
        deduplicate: bool = True,
        normalize_embeddings: bool = True,
        default_metadata: Optional[Dict[str, Any]] = None,
        tmp_dir: Optional[str] = None,
    ):
        self.embedder = embedder
        self.vs = vectorstore
        self.reader = file_reader or FileReader()
        self.chunker = chunker or Chunker()
        self.logger = logger or logging.getLogger(__name__)
        self.progress_cb = progress_cb
        self.cancel_event = cancel_event or threading.Event()
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.deduplicate = deduplicate
        self.normalize_embeddings = normalize_embeddings
        self.default_md = default_metadata or {}
        self.tmp_dir = tmp_dir or "./tmp"
        os.makedirs(self.tmp_dir, exist_ok=True)

        # Initialize URL fetcher
        self.url_fetcher = URLFetcher(
            config=URLFetcherConfig(
                cache_dir=os.path.join(self.tmp_dir, "url_cache"),
            ),
            file_reader=self.reader,
            logger=self.logger,
        )
        self._queue: List[Tuple[str, str, Dict[str, Any]]] = []  # (source_id, text, metadata)
        self._seen_hashes: set[str] = set()

    def set_vectorstore(self, vectorstore: BaseVectorDB) -> None:
        self.vs = vectorstore

    # ----------------------
    # Queueing inputs
    # ----------------------
    def add_files(
        self,
        paths: Sequence[Union[str, Path]],
        include_exts: Optional[set[str]] = supported_file_types,
        *,
        root: Optional[Union[str, Path]] = None,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> "RAGIngestor":
        extra_metadata = extra_metadata or {}
        for p in paths:
            p = Path(p)
            if p.suffix.lower() not in include_exts:
                continue
            text = self.reader.read(p)
            if not text: 
                continue
            rel = str(Path(p).relative_to(root)) if root and Path(p).is_relative_to(root) else str(p)
            md = {**self.default_md, **extra_metadata, "file_path": rel, "source_type": "file"}
            self._enqueue(source_id=rel, text=text, metadata=md)
        return self

    def add_directory(
        self, 
        directory: Union[str, Path], 
        *, 
        include_exts: Optional[set[str]] = supported_file_types,
        exclude_dirs: Optional[set[str]] = None, 
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> "RAGIngestor":
        extra_metadata = extra_metadata or {}
        directory = Path(directory)
        exclude_dirs = exclude_dirs or {".git", "__pycache__", ".venv", "node_modules", "dist", "build"}
        include_exts = {e.lower() for e in (include_exts or set())}

        for root, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            for fname in files:
                path = Path(root) / fname
                if path.suffix.lower() not in include_exts:
                    continue
                text = self.reader.read(path)
                if not text:
                    continue
                rel = str(path.relative_to(directory))
                md = {**self.default_md, **extra_metadata, "file_path": rel, "source_type": "file"}
                self._enqueue(source_id=rel, text=text, metadata=md)
        return self

    def add_texts(
        self, 
        texts: Sequence[str], 
        *, 
        base_id: str = "text", 
        metadatas: Optional[Sequence[Dict[str, Any]]] = None
    ) -> "RAGIngestor":
        metadatas = metadatas or [{}] * len(texts)
        for i, (t, md) in enumerate(zip(texts, metadatas)):
            if not t:
                continue
            source_id = f"{base_id}:{i}"
            full_md = {**self.default_md, **md, "source_type": "raw_text", "ordinal": i}
            self._enqueue(source_id=source_id, text=t, metadata=full_md)
        return self

    def add_urls(
        self, 
        urls: Sequence[str], 
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> "RAGIngestor":
        """Pass a custom fetcher to keep this offline-friendly. Example fetcher can use requests/bs4."""
        extra_metadata = extra_metadata or {}
        for url in urls:
            result = None
            try:
                result = self.url_fetcher.fetch(url)
            except Exception as e:
                self.logger.warning(f"URL fetch failed: {url}: {e}")
            if not result or not result.text:
                continue
            md = {**self.default_md, **extra_metadata, "url": url, "source_type": result.source_type, "content_type": result.content_type}
            self._enqueue(source_id=url, text=result.text, metadata=md)
        return self

    def add_git_folder(
        self, repo_root: Union[str, Path], 
        *, 
        include_exts: Optional[set[str]] = None,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> "RAGIngestor":
        """Index an already-cloned repo folder (avoids adding .git, node_modules, etc.)."""
        return self.add_directory(
            repo_root,
            include_exts=include_exts,
            exclude_dirs=exclude_dirs_in_code,
            extra_metadata=extra_metadata,
        )

    def add_zipfile(
        self,
        zip_path: Union[str, Path],
        *,
        include_exts: Optional[set[str]] = supported_file_types,
        exclude_dirs: Optional[set[str]] = exclude_dirs_in_code,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> "RAGIngestor":
        """
        Extract a single .zip into a temporary folder, index its contents using add_directory,
        then delete the temp folder. Adds {'source_zip': <zip filename>} to each doc's metadata.
        """
        zip_path = Path(zip_path)
        if not zip_path.is_file() or zip_path.suffix.lower() != ".zip":
            raise ValueError(f"Invalid zip file: {zip_path}")

        extra_metadata = extra_metadata or {}
        with zipfile.ZipFile(zip_path, "r") as zf, tempfile.TemporaryDirectory(prefix="rag_zip_", dir=self.tmp_dir) as tmpdir:
            tmp_root = Path(tmpdir).resolve()
            print(tmp_root)

            # --- Safe extraction to avoid zip-slip ---
            for info in zf.infolist():
                target_path = (tmp_root / info.filename).resolve()
                if not str(target_path).startswith(str(tmp_root)):
                    continue  # skip suspicious path

                if info.is_dir():
                    target_path.mkdir(parents=True, exist_ok=True)
                    continue

                target_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info, "r") as src, open(target_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)

            # --- Index extracted folder ---
            md = {**extra_metadata, "source_zip": zip_path.name}
            self.add_directory(
                tmp_root,
                include_exts=include_exts,
                exclude_dirs=exclude_dirs,
                extra_metadata=md,
            )
        return self

    def ingest(
        self,
        sources: Optional[Union[str, Path, Iterable[Union[str, Path, str]]]] = None,
        *,
        url_fetcher: Optional[Callable[[str], Optional[str]]] = None,
        include_exts: Optional[set[str]] = supported_file_types,
        extra_metadata: Optional[Dict[str, Any]] = None,
        reset_state: bool = True,
    ) -> Dict[str, Any]:
        """
        Unified high-level ingestion wrapper.

        This method accepts a mix of paths, URLs, and raw text, routes them to the
        appropriate add_* methods, and then runs the full pipeline via `build()`.

        Supported source forms:
            - File path (string or Path)
            - Directory path (string or Path)
            - .zip archive (string or Path)
            - Git repo folder (directory containing a .git subfolder)
            - URL string (starting with http:// or https://)
            - Raw text string (everything else)

        Parameters
        ----------
        sources:
            A single source or iterable of sources. Each element can be:
            - str
            - pathlib.Path
        url_fetcher:
            Optional callable used by `add_urls`. Signature: (url: str) -> Optional[str].
        include_exts:
            Allowed file extensions for file/directory/zip ingestion. Defaults to
            `supported_file_types`.
        extra_metadata:
            Extra metadata merged into each queued document's metadata.
        reset_state:
            If True, clears previous queue and seen-hash set before processing.

        Returns
        -------
        Dict[str, Any]
            A summary dictionary, typically the result of `build()`. On error, returns:
                {
                    "status": "error",
                    "error": "<message>",
                    "unsupported": [...optional list of unsupported items...]
                }
        """
        # Normalize and guard: allow ingest() to just run build() on previously-queued data.
        if reset_state:
            self._queue.clear()
            self._seen_hashes.clear()

        if sources is None:
            if not self._queue:
                msg = "No sources provided and ingestion queue is empty."
                self.logger.warning(msg)
                return {"status": "error", "error": msg}
            try:
                return self.build()
            except Exception as e:
                self.logger.exception("Ingestion run failed")
                return {"status": "error", "error": str(e)}

        # Normalize sources into a flat list
        if isinstance(sources, (str, Path)):
            raw_items: List[Union[str, Path]] = [sources]
        elif isinstance(sources, Iterable):
            raw_items = list(sources)
        else:
            msg = (
                f"Unsupported 'sources' type: {type(sources)!r}. "
                "Expected str, Path, or an iterable of those."
            )
            self.logger.error(msg)
            return {"status": "error", "error": msg}

        extra_metadata = extra_metadata or {}
        include_exts = {e.lower() for e in (include_exts or set())}
        unsupported: List[Any] = []
        accepted_count = 0

        for src in raw_items:
            # ----------------------
            # Path-like handling
            # ----------------------
            if isinstance(src, Path):
                path = src
                is_path = True
            elif isinstance(src, str):
                stripped = src.strip()
                if not stripped:
                    # Empty string → ignore silently
                    continue

                # URL detection
                if stripped.startswith("http://") or stripped.startswith("https://"):
                    self.add_urls([stripped], extra_metadata=extra_metadata)
                    accepted_count += 1
                    continue
                
                try:
                    path = Path(stripped)
                    is_path = path.exists()
                except Exception:
                    is_path = False   # Treat as raw text
                if not is_path:
                    # Treat as raw text
                    self.add_texts(
                        [stripped],
                        base_id="text",
                        metadatas=[{**extra_metadata}],
                    )
                    accepted_count += 1
                    continue
            else:
                unsupported.append(src)
                continue

            # Existing path routing
            try:
                if path.is_dir():
                    # Git repo detection
                    if (path / ".git").is_dir():
                        self.add_git_folder(path, include_exts=include_exts, extra_metadata=extra_metadata)
                    else:
                        self.add_directory(path, include_exts=include_exts, extra_metadata=extra_metadata)
                    accepted_count += 1

                elif path.is_file():
                    # Zip archive
                    if path.suffix.lower() == ".zip":
                        self.add_zipfile(path, include_exts=include_exts, extra_metadata=extra_metadata)
                    else:
                        self.add_files([path], include_exts=include_exts, extra_metadata=extra_metadata)
                    accepted_count += 1
                else:
                    # Path object that is neither file nor dir (very rare)
                    unsupported.append(src)

            except Exception as e:
                self.logger.exception(f"Failed to enqueue source {src!r}: {e}")
                unsupported.append(src)

        # Post-validation
        if not self._queue:
            msg = "No content was queued for ingestion. All inputs were either empty, filtered out, or unsupported."
            if unsupported:
                msg += f" Unsupported items: {unsupported!r}"
            self.logger.error(msg)
            return {
                "status": "error",
                "error": msg,
                "unsupported": [str(u) for u in unsupported],
            }
        if unsupported:
            self.logger.warning("Some sources were skipped as unsupported: %r", unsupported)

        # Execute pipeline
        try:
            summary = self.build()
        except Exception as e:
            self.logger.exception("Ingestion run failed")
            return {"status": "error", "error": str(e)}

        # Attach wrapper-level info
        summary.setdefault("status", "ok")
        summary["accepted_sources"] = accepted_count
        if unsupported:
            summary["unsupported"] = [str(u) for u in unsupported]
        summary["total_docs_in_collection"] = self.vs.count()
        return summary

    def build(self) -> Dict[str, Any]:
        """
        Execute the ingestion:
          - chunk queued sources
          - dedupe by chunk hash
          - embed in batches
          - write to vector store
        Returns a summary dict.
        """
        if self.progress_cb:
            self.progress_cb({"stage": "start", "queued_sources": len(self._queue)})

        if self.cancel_event.is_set():
            return {"status": "cancelled", "queued_sources": len(self._queue)}

        # 1) Chunk concurrently
        chunk_records: List[Tuple[str, str, Dict[str, Any]]] = []  # (doc_id, chunk_text, metadata)
        total_sources = len(self._queue)
        completed = 0

        def _chunk_one(args: Tuple[str, str, Dict[str, Any]]):
            source_id, text, md = args
            return [(source_id, c, md) for c in self.chunker.chunk(text, source_id=source_id)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = [ex.submit(_chunk_one, item) for item in self._queue]
            for fut in concurrent.futures.as_completed(futures):
                if self.cancel_event.is_set():
                    break
                try:
                    result = fut.result()
                    chunk_records.extend(result)
                except Exception as e:
                    self.logger.exception(f"Chunking failed: {e}")
                completed += 1
                if self.progress_cb:
                    self.progress_cb({
                        "stage": "chunking",
                        "completed_sources": completed,
                        "total_sources": total_sources,
                        "progress_pct": int(100 * completed / max(1, total_sources))
                    })

        if self.cancel_event.is_set():
            return {"status": "cancelled", "chunked": len(chunk_records)}

        # 2) Deduplicate by chunk hash
        unique_chunks: List[Tuple[str, str, Dict[str, Any]]] = []
        for source_id, chunk_text, md in chunk_records:
            h = sha256_text(chunk_text)
            if self.deduplicate and h in self._seen_hashes:
                continue
            self._seen_hashes.add(h)
            unique_chunks.append((source_id, chunk_text, {**md, "content_hash": h}))

        if self.progress_cb:
            self.progress_cb({"stage": "dedupe", "before": len(chunk_records), "after": len(unique_chunks)})

        # 3) Batch embed
        docs_to_add: List[Doc] = []
        for i in range(0, len(unique_chunks), self.batch_size):
            if self.cancel_event.is_set():
                break
            batch = unique_chunks[i:i + self.batch_size]
            texts = [b[1] for b in batch]
            try:
                embeddings = self.embedder.embed(texts, normalize_embeddings=self.normalize_embeddings)
            except Exception as e:
                self.logger.exception(f"Embedding batch failed (idx={i}): {e}")
                continue

            for (source_id, chunk_text, md), emb in zip(batch, embeddings):
                doc_id = f"{md.get('content_hash')[:16]}:{sha256_text(source_id)[:8]}"
                docs_to_add.append(Doc(id=doc_id, text=chunk_text, embedding=emb, metadata=md))

            if self.progress_cb:
                self.progress_cb({
                    "stage": "embedding",
                    "embedded": min(i + self.batch_size, len(unique_chunks)),
                    "total": len(unique_chunks)
                })

        if self.cancel_event.is_set():
            return {"status": "cancelled", "embedded": len(docs_to_add)}

        # 4) Persist to vector store
        try:
            # Some stores prefer smaller bulks; split again if you need
            self.vs.add_documents(docs_to_add)
        except Exception as e:
            self.logger.exception(f"Vector store add_documents failed: {e}")
            raise

        if self.progress_cb:
            self.progress_cb({"stage": "done", "added": len(docs_to_add)})

        return {
            "status": "ok",
            "sources": len(self._queue),
            "chunks": len(chunk_records),
            "unique_chunks": len(unique_chunks),
            "added": len(docs_to_add),
            "finished_at": now_ts(),
        }

    # ----------------------
    # Optional retrieval (smoke test)
    # ----------------------
    def embed_query(self, text: str) -> List[float]:
        return self.embedder.embed([text], normalize_embeddings=self.normalize_embeddings)[0]

    def search(self, text: str, top_k: int = 5) -> List[Tuple[Doc, float]]:
        q = self.embed_query(text)
        return self.vs.similarity_search(q, top_k=top_k)

    # ----------------------
    # Internals
    # ----------------------
    def _enqueue(self, *, source_id: str, text: str, metadata: Dict[str, Any]) -> None:
        if self.cancel_event.is_set():
            return
        self._queue.append((source_id, text, metadata))
