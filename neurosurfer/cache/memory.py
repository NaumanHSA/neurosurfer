"""In-memory LRU response cache with optional TTL."""
from __future__ import annotations

from collections import OrderedDict

from neurosurfer.llm.types import CanonicalResponse

from .base import CacheEntry, CacheKey, ResponseCache


class InMemoryResponseCache(ResponseCache):
    """LRU dict-backed cache. Thread-safe enough for single-process use.

    Args:
        maxsize: Maximum number of entries. Oldest (LRU) entry is evicted when
            full. 0 disables size bounding.
        ttl: Seconds before an entry is considered stale. None = never expires.
    """

    def __init__(self, maxsize: int = 256, ttl: float | None = 3600.0) -> None:
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self.maxsize = maxsize
        self.ttl = ttl

    # ------------------------------------------------------------------ #
    def get(self, key: CacheKey) -> CanonicalResponse | None:
        entry = self._store.get(key.key)
        if entry is None:
            return None
        if entry.is_expired(self.ttl):
            del self._store[key.key]
            return None
        # Bump to end (most-recently used)
        self._store.move_to_end(key.key)
        entry.hits += 1
        return entry.response

    def set(self, key: CacheKey, response: CanonicalResponse) -> None:
        if key.key in self._store:
            self._store.move_to_end(key.key)
            self._store[key.key] = CacheEntry(response=response)
            return
        if self.maxsize and len(self._store) >= self.maxsize:
            self._store.popitem(last=False)  # evict LRU (front)
        self._store[key.key] = CacheEntry(response=response)

    def clear(self) -> None:
        self._store.clear()

    def size(self) -> int:
        if self.ttl is None:
            return len(self._store)
        # Count non-expired entries
        return sum(1 for e in self._store.values() if not e.is_expired(self.ttl))
