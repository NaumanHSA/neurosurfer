"""Vector store backends.

``BaseVectorDB`` and ``Doc`` are always available. The concrete backends
(``ChromaVectorStore``, ``InMemoryVectorStore``) are imported lazily so that
merely importing this package — or anything that only needs ``Doc`` — does not
require the optional ``chromadb`` dependency.
"""
from .base import BaseVectorDB, Doc

__all__ = ["BaseVectorDB", "Doc", "ChromaVectorStore", "InMemoryVectorStore"]


def __getattr__(name: str):
    if name == "ChromaVectorStore":
        from .chroma import ChromaVectorStore
        return ChromaVectorStore
    if name == "InMemoryVectorStore":
        from .in_memory_store import InMemoryVectorStore
        return InMemoryVectorStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
