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
    #: The interaction the build is currently parked on, if any. Mirrors the
    #: gate's state onto the record so a studio that reconnects mid-build (or
    #: polls instead of streaming) still sees what is being asked.
    pending: dict[str, Any] | None = None
    #: Structured verification outcome: criteria, per-criterion verdicts, branch
    #: cases and coverage gaps. Previously this reached the studio only as a
    #: rendered text blob in the log.
    verification: dict[str, Any] | None = None
    #: Set when this build refines an existing workflow rather than starting fresh.
    refines: str | None = None
    #: The plan this build is executing (V3 Phase 3), with each step's resolved
    #: capability. Emitted as its own event so the studio can show what was
    #: intended before any node exists — and, when a build blocks at the plan,
    #: what it blocked on.
    plan: dict[str, Any] | None = None
    #: The design review (V3 Phase 4): whether the graph answers the question it
    #: was asked, and what the reviewer would change. A judgement, not a gate —
    #: a build can succeed with issues recorded here.
    review: dict[str, Any] | None = None
    #: What the user would have to supply for a blocked build to become possible
    #: (V3 Phase 2c, surfaced in Phase 6): the servers that provide the missing
    #: capability and the credentials each wants. `error` carries the same answer
    #: as a paragraph; this is the shape a checklist can be built from.
    requirements: list[dict[str, Any]] = field(default_factory=list)
    #: Plan-to-graph coverage (V3 Phase 4b): steps explicitly dropped, and nodes no
    #: step asked for. Registration already refuses a *missing* step, so what is
    #: left here is what a green build would otherwise hide.
    coverage: dict[str, Any] | None = None
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
            "pending": self.pending,
            "verification": self.verification,
            "refines": self.refines,
            "plan": self.plan,
            "review": self.review,
            "requirements": self.requirements,
            "coverage": self.coverage,
        }
        if include_events:
            d["events"] = self.events
        return d
