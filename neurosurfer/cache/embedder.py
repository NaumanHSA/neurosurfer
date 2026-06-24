"""CachedEmbedder — in-memory LRU cache wrapper for any Embedder."""
from __future__ import annotations

import hashlib
import json
from collections import OrderedDict

from neurosurfer.embeddings import Embedder


def _texts_key(texts: list[str]) -> str:
    return hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest()


class CachedEmbedder:
    """Wraps any Embedder with an in-memory LRU cache.

    Each unique list of texts is cached by its SHA-256 hash. When
    ``maxsize=0`` the wrapper is a transparent pass-through.

    Usage::

        embedder = _LocalEmbedder("all-MiniLM-L6-v2")
        cached = CachedEmbedder(embedder, maxsize=4096)
        vecs = cached.embed(["hello", "world"])  # first call
        vecs = cached.embed(["hello", "world"])  # cache hit
    """

    def __init__(self, embedder: Embedder, maxsize: int = 4096) -> None:
        self._embedder = embedder
        self._cache: OrderedDict[str, list[list[float]]] = OrderedDict()
        self.maxsize = maxsize

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.maxsize or not texts:
            return self._embedder.embed(texts)

        key = _texts_key(texts)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        result = self._embedder.embed(texts)
        if len(self._cache) >= self.maxsize:
            self._cache.popitem(last=False)
        self._cache[key] = result
        return result

    def clear(self) -> None:
        self._cache.clear()

    def size(self) -> int:
        return len(self._cache)
