"""Memory data model.

A ``MemoryEntry`` is one durable fact the agent learned. It carries enough
provenance (source, session) and signal (kind, salience, uses) for retrieval to
rank it and for the curator to explain it. Entries are stored as JSONL lines.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

# global  → about the user, applies everywhere
# agent   → about one specialized agent's domain (scope_key = agent/task name)
MemoryScope = Literal["global", "agent"]

# preference → how to behave; fact → a stable truth; decision → a choice made on a
# run; glossary → a term/definition. Drives nothing enforced — used for display + light ranking.
MemoryKind = Literal["preference", "fact", "decision", "glossary"]


def _new_id(scope: str, scope_key: str, text: str) -> str:
    h = hashlib.sha1(f"{scope}|{scope_key}|{text}".encode()).hexdigest()
    return h[:12]


class MemoryEntry(BaseModel):
    scope: MemoryScope = "global"
    scope_key: str = ""  # the agent/task name for scope=agent; empty for global
    kind: MemoryKind = "fact"
    text: str
    source: str = "agent"  # "agent" | "distill" | "user" | "curator"
    session_id: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    uses: int = 0
    salience: float = 1.0
    supersedes: list[str] = Field(default_factory=list)
    id: str = ""

    def model_post_init(self, __context: object) -> None:
        if not self.id:
            self.id = _new_id(self.scope, self.scope_key, self.text)
