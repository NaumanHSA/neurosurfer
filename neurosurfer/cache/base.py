"""Cache primitives: key, entry, and the ResponseCache ABC."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from neurosurfer.llm.types import CanonicalResponse


@dataclass(frozen=True)
class CacheKey:
    """Opaque stable identifier for a (model, messages, config) tuple."""
    key: str  # sha256 hex digest


@dataclass
class CacheEntry:
    response: CanonicalResponse
    created_at: float = field(default_factory=time.time)
    hits: int = 0

    def is_expired(self, ttl: float | None) -> bool:
        """Return True if TTL is set and has elapsed since creation."""
        if ttl is None:
            return False
        return time.time() - self.created_at > ttl


class ResponseCache(ABC):
    """Interface all response-cache backends must implement."""

    @abstractmethod
    def get(self, key: CacheKey) -> CanonicalResponse | None:
        """Return the cached response for *key*, or None on miss / expiry."""
        ...

    @abstractmethod
    def set(self, key: CacheKey, response: CanonicalResponse) -> None:
        """Store *response* under *key*."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Evict all entries."""
        ...

    @abstractmethod
    def size(self) -> int:
        """Return the number of live (non-expired) entries."""
        ...
