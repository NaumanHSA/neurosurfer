"""Response and embedding caches — opt-in, off by default.

Quick start::

    from neurosurfer.cache import get_response_cache, CachedProvider, CachedEmbedder

    # LLM response cache (in-memory, 1-hour TTL)
    cache = get_response_cache("memory", ttl=3600)
    provider = CachedProvider(provider, cache=cache)

    # Or disk-backed:
    cache = get_response_cache("disk", directory="./cache", ttl=86400)

    # Embedding cache
    embedder = CachedEmbedder(embedder, maxsize=4096)

    # Disabled (transparent pass-through):
    provider = CachedProvider(provider, cache=None)
"""

from __future__ import annotations

from pathlib import Path

from .base import CacheEntry, CacheKey, ResponseCache
from .disk import DiskResponseCache
from .embedder import CachedEmbedder
from .memory import InMemoryResponseCache
from .provider import CachedProvider


def get_response_cache(
    backend: str | None = None,
    *,
    maxsize: int = 256,
    ttl: float | None = 3600.0,
    directory: str | Path = ".cache/responses",
) -> ResponseCache | None:
    """Factory for response cache backends. Returns None when disabled.

    Args:
        backend: ``"memory"`` | ``"disk"`` | ``None``/``"off"`` (disabled).
        maxsize: Maximum entries for in-memory backend (LRU eviction).
        ttl: Entry lifetime in seconds. None = never expires.
        directory: Root directory for disk backend.
    """
    name = (backend or "off").strip().lower()
    if name in ("", "off", "none", "disabled"):
        return None
    if name == "memory":
        return InMemoryResponseCache(maxsize=maxsize, ttl=ttl)
    if name == "disk":
        return DiskResponseCache(directory=directory, ttl=ttl)
    raise ValueError(f"Unknown cache backend: {backend!r}. Choose 'memory', 'disk', or None.")


__all__ = [
    "CacheKey",
    "CacheEntry",
    "ResponseCache",
    "InMemoryResponseCache",
    "DiskResponseCache",
    "CachedProvider",
    "CachedEmbedder",
    "get_response_cache",
]
