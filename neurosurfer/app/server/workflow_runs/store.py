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
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger("neurosurfer.server.runs")

RunStatus = Literal[
    "running", "succeeded", "failed", "cancelled", "awaiting_input", "interrupted"
]

# Terminal statuses — no further events will be appended once a record reaches one.
# `interrupted` is terminal too: it marks a run whose process died mid-flight, so
# it can never make progress again.
TERMINAL: frozenset[str] = frozenset(
    {"succeeded", "failed", "cancelled", "interrupted"}
)


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
    # Token usage summed over every node that called a model. Container nodes
    # already fold in their body iterations, so nothing is double-counted.
    usage: dict[str, int] | None = None
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
            "usage": self.usage,
        }
        if include_events:
            d["events"] = self.events
        return d

    def summary(self) -> dict[str, Any]:
        """A list-view row: everything a runs table needs, and nothing heavy.

        Deliberately excludes ``nodes`` and ``final`` — those carry every node's
        full output, which is fine for one record but megabytes across a few
        hundred rows. Fetch ``GET /v1/runs/{id}`` for the detail.
        """
        counts: dict[str, int] = {}
        for node in self.nodes.values():
            status = str(node.get("status") or "pending")
            counts[status] = counts.get(status, 0) + 1
        return {
            "id": self.id,
            "workflow": self.workflow,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "duration_s": max(0.0, self.updated_at - self.created_at),
            "error": self.error,
            "usage": self.usage,
            "node_counts": counts,
            "node_total": len(self.nodes),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunRecord:
        """Rebuild a record from its persisted JSON (see :meth:`RunStore._restore`).

        Only fields that were actually written come back; the runtime-control
        fields (`_seq`, `_cancel`) are deliberately fresh, since a restored run is
        finished and nothing more will be appended to it.
        """
        rec = cls(
            id=d["id"],
            workflow=d.get("workflow", "?"),
            inputs=d.get("inputs") or {},
            status=d.get("status", "succeeded"),
            created_at=float(d.get("created_at") or 0.0),
            updated_at=float(d.get("updated_at") or 0.0),
            events=list(d.get("events") or []),
            nodes=dict(d.get("nodes") or {}),
            final=d.get("final"),
            error=d.get("error"),
            trace_path=d.get("trace_path"),
            usage=d.get("usage"),
        )
        # Keep seq monotonic if anything ever does append to a restored record.
        rec._seq = max((e.get("seq", 0) for e in rec.events), default=0)
        return rec


class RunStore:
    """Thread-safe registry of runs, backed by per-run JSON on disk.

    Records are held in memory and mirrored to ``runs_dir/<id>.json``. Past runs
    are read back at construction so history survives a gateway restart — without
    that, every restart silently emptied the runs list even though the files were
    sitting on disk the whole time.
    """

    #: Cap on runs restored at startup. A long-lived gateway accumulates
    #: thousands; loading all of them would slow boot for records nobody opens.
    MAX_RESTORED = 500

    def __init__(self, runs_dir: Path | None = None, *, restore: bool = True) -> None:
        from neurosurfer.config.paths import runs_dir as default_runs_dir

        self._runs: dict[str, RunRecord] = {}
        self._lock = threading.Lock()
        self._dir = runs_dir or default_runs_dir()
        if restore:
            self._restore()

    @property
    def dir(self) -> Path:
        """Directory where run records (and their traces) are persisted."""
        return self._dir

    def _restore(self) -> None:
        """Load the most recent persisted runs back into memory (best-effort)."""
        if not self._dir.is_dir():
            return
        files = [p for p in self._dir.glob("*.json") if not p.name.endswith(".trace.json")]
        # Newest first by mtime, so the cap keeps what a user is most likely to open.
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for path in files[: self.MAX_RESTORED]:
            try:
                rec = RunRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
            except Exception:  # noqa: BLE001 - one corrupt record must not block boot
                logger.warning("skipping unreadable run record: %s", path.name)
                continue
            # A run that was mid-flight when the process died can never resume;
            # showing it as "running" forever would be a lie.
            if rec.status == "running":
                rec.status = "interrupted"
            self._runs[rec.id] = rec

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

    def finish(
        self,
        rec: RunRecord,
        status: RunStatus,
        *,
        final: dict | None = None,
        error: str | None = None,
    ) -> None:
        """Mark a run terminal **and** make it durable, atomically for readers.

        Terminal status and a persisted record have to become visible together.
        Otherwise a client that polls until the run is no longer ``running`` and
        then reads the record — exactly what a caller waiting on a run does —
        can find no file at all, because ``persist`` had not happened yet.

        Holding the store lock across both means :meth:`get` cannot hand out the
        record until it is on disk.
        """
        with self._lock:
            rec.finish(status, final=final, error=error)
            self.persist(rec)

    def persist(self, rec: RunRecord) -> None:
        """Write the run record to ``runs_dir/<id>.json`` (best-effort).

        Written to a temp file and renamed, because ``write_text`` truncates
        first: a reader (or a second persist of the same run) could otherwise
        observe a half-written file and fail to parse it. ``os.replace`` is
        atomic within a filesystem, so a reader sees either the old record or the
        new one — never a partial one.
        """
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            path = self._dir / f"{rec.id}.json"
            payload = json.dumps(rec.to_dict(), ensure_ascii=False, default=str, indent=2)
            tmp = path.with_suffix(f".json.{os.getpid()}.{threading.get_ident()}.tmp")
            tmp.write_text(payload, encoding="utf-8")
            os.replace(tmp, path)
        except OSError:
            pass  # persistence is best-effort; the in-memory record is authoritative
