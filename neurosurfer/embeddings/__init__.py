"""Embedder protocol and backends — shared by rag/ and memory/.

Usage:
    from neurosurfer.embeddings import Embedder, get_embedder

The cardinal rule: ``get_embedder`` never raises — returns None on any
failure so callers can degrade to lexical/BM25 search gracefully.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..observability.logging import get_logger

log = get_logger("embeddings")


@runtime_checkable
class Embedder(Protocol):
    """Maps a list of texts to dense vectors. Implementations must be deterministic."""

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class NullEmbedder:
    """No-op backend — signals 'use BM25/lexical'. Returned when embeddings are off."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return []


def get_embedder(backend: str | None) -> Embedder | None:
    """Return an Embedder for ``backend``, or ``None`` to use BM25/lexical fallback.

    ``None``/``"none"``/``"bm25"`` → no embedder (None returned).
    Any other string attempts to load the named backend and **degrades to None
    on any error** (never raises).
    """
    name = (backend or "none").strip().lower()
    if name in ("", "none", "bm25", "null", "off"):
        return None
    try:
        if name in ("local", "sentence-transformers", "st"):
            return _LocalEmbedder()
        # Accept a model name directly (e.g. "intfloat/e5-small-v2")
        return _LocalEmbedder(model=backend)  # type: ignore[arg-type]
    except Exception as e:  # noqa: BLE001
        log.warning("embeddings backend '%s' unavailable: %s", backend, e)
        return None


class _LocalEmbedder:
    """sentence-transformers backend (optional dep). Never required at import time."""

    def __init__(self, model: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore

        self._model = SentenceTransformer(model)
        self._model_name = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return [list(map(float, v)) for v in vecs]


__all__ = ["Embedder", "NullEmbedder", "_LocalEmbedder", "get_embedder"]
