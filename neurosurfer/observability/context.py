"""Trace identity shared across an agent run (and, later, a workflow).

A :class:`TraceContext` is the thin thread of identity every exporter hangs its
observations off. One is minted per agent run in
:meth:`neurosurfer.agents.base.BaseAgent._tap` and handed to every configured
:class:`~neurosurfer.observability.exporters.base.TraceExporter`.

Ids reuse :func:`neurosurfer.observability.transcript.new_run_id` (a short uuid4
hex) so the transcript, the trace backend, and any logs can be correlated by eye.
``parent_span_id`` is unused for a lone agent run; it exists so a graph/workflow
run can nest each node's agent spans under the node's span (plan Phase 5).
"""

from __future__ import annotations

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
        trace_id: Stable id for the whole run — the root the backend groups by.
        session_id: Optional grouping across runs (a chat session / user thread).
        parent_span_id: Parent span to nest under (set by a workflow; ``None`` for
            a standalone agent run).
        metadata: Free-form tags forwarded to the backend (agent type, model, …).
    """

    trace_id: str = field(default_factory=new_run_id)
    session_id: str | None = None
    parent_span_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def child(self, parent_span_id: str) -> TraceContext:
        """A context sharing this trace/session but nested under *parent_span_id*."""
        return TraceContext(
            trace_id=self.trace_id,
            session_id=self.session_id,
            parent_span_id=parent_span_id,
            metadata=dict(self.metadata),
        )
