"""On-disk JSON response cache with optional TTL."""
from __future__ import annotations

import json
import time
from pathlib import Path

from neurosurfer.llm.types import CanonicalResponse

from .base import CacheEntry, CacheKey, ResponseCache
from ._serde import serialize, deserialize


class DiskResponseCache(ResponseCache):
    """JSON-file-backed cache. Each entry is one file under *directory*.

    Files are stored as ``{dir}/{key[:2]}/{key}.json`` to avoid a flat
    directory with millions of files.

    Args:
        directory: Root directory for cache files (created automatically).
        ttl: Seconds before an entry is considered stale. None = never expires.
    """

    def __init__(self, directory: str | Path, ttl: float | None = 3600.0) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl

    # ------------------------------------------------------------------ #
    def _path(self, key: CacheKey) -> Path:
        shard = key.key[:2]
        return self._dir / shard / f"{key.key}.json"

    def get(self, key: CacheKey) -> CanonicalResponse | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text())
            if self.ttl is not None and time.time() - raw["created_at"] > self.ttl:
                path.unlink(missing_ok=True)
                return None
            return deserialize(raw["response"])
        except Exception:
            return None

    def set(self, key: CacheKey, response: CanonicalResponse) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_text(json.dumps({
                "created_at": time.time(),
                "response": json.loads(serialize(response)),
            }))
        except Exception:
            pass  # disk errors must never break callers

    def clear(self) -> None:
        import shutil
        if self._dir.exists():
            shutil.rmtree(self._dir)
            self._dir.mkdir(parents=True, exist_ok=True)

    def size(self) -> int:
        count = 0
        for p in self._dir.rglob("*.json"):
            try:
                raw = json.loads(p.read_text())
                if self.ttl is None or time.time() - raw["created_at"] <= self.ttl:
                    count += 1
            except Exception:
                pass
        return count
