"""Durable run records for the workflow execution API (Phase 2).

A :class:`RunRecord` captures everything about one workflow execution — inputs,
per-node results, an append-only event log (the live stream the UI tails), the
final outputs, timings, and status. :class:`RunStore` keeps records in memory for
the process lifetime and persists a JSON copy per run so a completed run stays
inspectable and replayable after a restart.

The event log is append-only and each event carries a monotonic ``seq``; the SSE
endpoint streams by tailing the list from a given index, so a client that connects
late still replays everything from the start. Appends happen from a worker thread
and reads from the async request handler — safe because CPython list ``append`` and
index reads are atomic under the GIL and we never mutate an existing event.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

RunStatus = Literal["running", "succeeded", "failed", "cancelled", "awaiting_input"]

# Terminal statuses — no further events will be appended once a record reaches one.
TERMINAL: frozenset[str] = frozenset({"succeeded", "failed", "cancelled"})


@dataclass
class RunRecord:
    """One workflow execution's full state (in-memory + JSON-persistable)."""

    id: str
    workflow: str
    inputs: dict[str, Any]
    status: RunStatus = "running"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    # Append-only event log (node lifecycle, logs, run start/finish).
    events: list[dict[str, Any]] = field(default_factory=list)
    # Per-node results, filled as nodes complete: node_id → {status, output, error, ...}.
    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    final: dict[str, Any] | None = None
    error: str | None = None
    trace_path: str | None = None
    # Non-serialized runtime control.
    _seq: int = field(default=0, repr=False)
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    # ── mutation ────────────────────────────────────────────────────────────
    def add_event(self, type: str, **fields: Any) -> dict[str, Any]:
        """Append an event to the log with a monotonic seq + timestamp."""
        self._seq += 1
        evt = {"seq": self._seq, "ts": time.time(), "type": type, **fields}
        self.events.append(evt)
        self.updated_at = evt["ts"]
        return evt

    def set_node(self, node_id: str, **fields: Any) -> None:
        self.nodes.setdefault(node_id, {})
        self.nodes[node_id].update(fields)

    def finish(self, status: RunStatus, *, final: dict | None = None, error: str | None = None) -> None:
        self.status = status
        self.final = final
        self.error = error
        self.updated_at = time.time()
        self.add_event("run", status=status, error=error)

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def request_cancel(self) -> None:
        self._cancel.set()

    # ── serialization ───────────────────────────────────────────────────────
    def to_dict(self, *, include_events: bool = True) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "workflow": self.workflow,
            "inputs": self.inputs,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "nodes": self.nodes,
            "final": self.final,
            "error": self.error,
            "trace_path": self.trace_path,
        }
        if include_events:
            d["events"] = self.events
        return d


class RunStore:
    """Thread-safe in-memory registry of runs, with per-run JSON persistence."""

    def __init__(self, runs_dir: Path | None = None) -> None:
        from neurosurfer.config.paths import runs_dir as default_runs_dir

        self._runs: dict[str, RunRecord] = {}
        self._lock = threading.Lock()
        self._dir = runs_dir or default_runs_dir()

    @property
    def dir(self) -> Path:
        """Directory where run records (and their traces) are persisted."""
        return self._dir

    def create(self, workflow: str, inputs: dict[str, Any]) -> RunRecord:
        run_id = uuid.uuid4().hex
        rec = RunRecord(id=run_id, workflow=workflow, inputs=dict(inputs))
        rec.add_event("run", status="running")
        with self._lock:
            self._runs[run_id] = rec
        return rec

    def get(self, run_id: str) -> RunRecord | None:
        with self._lock:
            return self._runs.get(run_id)

    def list(self) -> list[RunRecord]:
        with self._lock:
            return sorted(self._runs.values(), key=lambda r: r.created_at, reverse=True)

    def persist(self, rec: RunRecord) -> None:
        """Write the run record to ``runs_dir/<id>.json`` (best-effort)."""
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            path = self._dir / f"{rec.id}.json"
            path.write_text(
                json.dumps(rec.to_dict(), ensure_ascii=False, default=str, indent=2),
                encoding="utf-8",
            )
        except OSError:
            pass  # persistence is best-effort; the in-memory record is authoritative
