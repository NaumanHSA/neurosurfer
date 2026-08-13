"""Embedding backends — shared by `rag/` and anything that retrieves.

Usage::

    from neurosurfer.embeddings import get_embedder

    get_embedder("openai-compat:nomic-embed-text@http://localhost:1234/v1")
    get_embedder("intfloat/e5-small-v2")       # sentence-transformers
    get_embedder("openai:text-embedding-3-small")
    get_embedder("none")                        # → None, use lexical search

`get_embedder` returns ``None`` when embeddings are switched off or the backend
is **not configured**, so a caller can fall back to BM25. It **raises** when a
backend is configured and broken — see :mod:`neurosurfer.embeddings.base` for why
those two must stay apart.
"""

from __future__ import annotations

from .base import (
    Embedder,
    EmbedderUnavailable,
    EmbeddingBackend,
    EmbeddingError,
    NullEmbedder,
)
from .local import LocalEmbedder
from .openai_compat import OpenAICompatEmbedder, OpenAIEmbedder
from .registry import BACKENDS, get_embedder, parse_spec

#: The pre-package name for the sentence-transformers backend. `rag/agent.py`
#: and `rag/ingestor.py` import it, and it appears in released documentation.
_LocalEmbedder = LocalEmbedder

__all__ = [
    "BACKENDS",
    "Embedder",
    "EmbedderUnavailable",
    "EmbeddingBackend",
    "EmbeddingError",
    "LocalEmbedder",
    "NullEmbedder",
    "OpenAICompatEmbedder",
    "OpenAIEmbedder",
    "_LocalEmbedder",
    "get_embedder",
    "parse_spec",
]
