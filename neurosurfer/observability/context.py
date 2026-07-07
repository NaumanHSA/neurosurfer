"""Trace identity shared across an agent run — and propagated to sub-agents.

A :class:`TraceContext` is the thin thread of identity every exporter hangs its
observations off. One is minted per agent run in
:meth:`neurosurfer.agents.base.BaseAgent._tap` and handed to every configured
:class:`~neurosurfer.observability.exporters.base.TraceExporter`.

Nesting (plan Phase 5): while a run is active its context is published in a
:class:`contextvars.ContextVar`. Anything that starts *inside* that run — a
spawned sub-agent, or a graph node's agent — reads it and builds a **child**
context that shares the ``trace_id``/``session_id`` and points ``parent_span_id``
at the enclosing run's ``span_id``. Exporters use that to render one nested trace
(parent run → sub-agent → tool) instead of many disconnected top-level traces.
contextvars propagate across ``await`` and are copied into ``asyncio.gather``
tasks, so both sequential and parallel sub-agents nest correctly.

Ids reuse :func:`neurosurfer.observability.transcript.new_run_id` (a short uuid4
hex) so the transcript, the trace backend, and any logs correlate by eye.
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass, field
from typing import Any

from neurosurfer.observability.transcript import new_run_id


def new_span_id() -> str:
    """A fresh span id (same shape as a run/trace id)."""
    return new_run_id()


@dataclass
class TraceContext:
    """Identity + metadata for one traced run.

    Attributes:
        trace_id: Stable id for the whole trace — the root the backend groups by.
            Shared by every run nested under the same top-level run.
        span_id: This run's own span id. Children nest under it.
        parent_span_id: The enclosing run's ``span_id`` (``None`` for a top-level run).
        session_id: Optional grouping across separate traces (a chat session).
        metadata: Free-form tags forwarded to the backend (agent type, model, …).
    """

    trace_id: str = field(default_factory=new_run_id)
    span_id: str = field(default_factory=new_span_id)
    parent_span_id: str | None = None
    session_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_root(self) -> bool:
        return self.parent_span_id is None

    def child(self, *, metadata: dict[str, Any] | None = None) -> TraceContext:
        """A new context nested under this run: same trace/session, fresh span,
        ``parent_span_id`` pointing at this run's ``span_id``."""
        return TraceContext(
            trace_id=self.trace_id,
            parent_span_id=self.span_id,
            session_id=self.session_id,
            metadata=metadata if metadata is not None else dict(self.metadata),
        )


# ── ambient current-run context (for sub-agent / node nesting) ───────────────
_current: contextvars.ContextVar[TraceContext | None] = contextvars.ContextVar(
    "neurosurfer_trace_context", default=None
)


def current_trace_context() -> TraceContext | None:
    """The trace context of the run currently executing, if any."""
    return _current.get()


def push_trace_context(ctx: TraceContext) -> contextvars.Token:
    """Publish *ctx* as the current run. Returns a token for :func:`pop_trace_context`."""
    return _current.set(ctx)


def pop_trace_context(token: contextvars.Token) -> None:
    """Restore the previous current run."""
    _current.reset(token)
