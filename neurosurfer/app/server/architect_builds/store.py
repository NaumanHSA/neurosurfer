"""In-memory record for one Architect build (S5).

Mirrors the workflow-run record's event-log shape so the studio can stream a
build the same way it streams a run: an append-only, ``seq``-numbered event log
plus the latest staged-graph snapshot and the terminal outcome.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

BUILD_TERMINAL = {"succeeded", "blocked", "failed", "cancelled"}


@dataclass
class BuildRecord:
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    intent: str = ""
    status: str = "running"  # running | succeeded | blocked | failed | cancelled
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    events: list[dict[str, Any]] = field(default_factory=list)
    graph: dict[str, Any] | None = None  # latest staged-graph snapshot
    workflow: str | None = None  # registered workflow name (on success)
    path: str | None = None
    error: str | None = None
    _seq: int = field(default=0, repr=False)

    def add_event(self, type: str, **fields: Any) -> dict[str, Any]:
        self._seq += 1
        evt = {"seq": self._seq, "ts": time.time(), "type": type, **fields}
        self.events.append(evt)
        self.updated_at = evt["ts"]
        return evt

    def to_dict(self, *, include_events: bool = True) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "intent": self.intent,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "graph": self.graph,
            "workflow": self.workflow,
            "path": self.path,
            "error": self.error,
        }
        if include_events:
            d["events"] = self.events
        return d
