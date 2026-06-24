"""Long-term memory: global + per-agent facts an agent recalls across sessions.

Local-first (plain JSONL, BM25 retrieval, no mandatory cloud). The store is keyed
by ``global`` plus the active agent/task name — never a git root or working
directory — so memory works identically installed globally, in Docker, or in any
folder.
"""

from __future__ import annotations

from .models import MemoryEntry
from .store import MemoryStore

__all__ = ["MemoryEntry", "MemoryStore"]
