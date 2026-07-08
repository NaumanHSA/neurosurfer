"""The exporter contract every monitoring backend implements.

A :class:`TraceExporter` receives an agent run's lifecycle as a sequence of
method calls, driven by
:class:`~neurosurfer.observability.exporters.stream.EventStreamObserver` which
translates the raw agent event stream (see
``neurosurfer/agents/conversation/events.py``) into these calls.

Design notes:
    - Every method has a **no-op default**, so a backend adapter overrides only
      the hooks it cares about.
    - Every method is keyword-only with defaults, so new fields can be added
      without breaking existing exporters.
    - An exporter must **never raise into the caller** — the observer guards each
      call — but adapters should still fail soft internally.
    - Tool calls pair by ``call_id`` (the id carried on both ``ToolStarted`` and
      ``ToolFinished``); an adapter keeps its own ``call_id → span`` map.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neurosurfer.llm.types import Usage
    from neurosurfer.observability.context import TraceContext
    from neurosurfer.tools.base import ToolResult


class TraceExporter:
    """Base class for trace exporters. Subclass and override the hooks you need."""

    #: Human name used in logs / registry. Overridden by adapters.
    name: str = "base"

    def on_run_start(
        self, ctx: TraceContext, *, name: str, input: Any = None
    ) -> None:
        """A run began — open the root trace/span."""

    def on_turn(
        self,
        ctx: TraceContext,
        *,
        usage: Usage,
        model: str | None,
        stop_reason: str,
        input: Any = None,
        output: str | None = None,
    ) -> None:
        """One LLM turn completed — record a generation with token usage."""

    def on_tool_start(
        self, ctx: TraceContext, *, call_id: str, name: str, args: dict[str, Any]
    ) -> None:
        """A tool call started — open a child span keyed by *call_id*."""

    def on_tool_finish(
        self,
        ctx: TraceContext,
        *,
        call_id: str,
        name: str,
        result: ToolResult,
        is_error: bool,
    ) -> None:
        """A tool call finished — close the span opened for *call_id*."""

    def on_event(self, ctx: TraceContext, *, kind: str, **data: Any) -> None:
        """A point-in-time event on the trace (mode change, context compaction)."""

    def on_error(self, ctx: TraceContext, *, message: str) -> None:
        """The run raised — mark the trace errored."""

    def on_run_finish(
        self, ctx: TraceContext, *, status: str, output: str | None = None
    ) -> None:
        """The run ended — close the root trace/span."""

    def flush(self) -> None:
        """Push any buffered data to the backend."""

    def close(self) -> None:
        """Release resources (called once at process teardown)."""


class NullExporter(TraceExporter):
    """The zero-overhead default: does nothing. Never on the hot path when the
    registry resolves to an empty exporter list, but handy as an explicit stub."""

    name = "null"


class MemoryExporter(TraceExporter):
    """Captures every lifecycle call in-memory as ``(hook, payload)`` tuples.

    For tests and local debugging — mirrors ``MemorySpanTracer`` in
    ``neurosurfer/tracing/span.py``. Inspect :attr:`calls` after a run.
    """

    name = "memory"

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def _rec(self, hook: str, **payload: Any) -> None:
        self.calls.append((hook, payload))

    def on_run_start(self, ctx, *, name, input=None):
        self._rec(
            "run_start",
            trace_id=ctx.trace_id,
            span_id=ctx.span_id,
            parent_span_id=ctx.parent_span_id,
            session_id=ctx.session_id,
            name=name,
            input=input,
        )

    def on_turn(self, ctx, *, usage, model, stop_reason, input=None, output=None):
        self._rec(
            "turn",
            trace_id=ctx.trace_id,
            span_id=ctx.span_id,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            model=model,
            stop_reason=stop_reason,
            input=input,
            output=output,
        )

    def on_tool_start(self, ctx, *, call_id, name, args):
        self._rec("tool_start", trace_id=ctx.trace_id, call_id=call_id, name=name, args=args)

    def on_tool_finish(self, ctx, *, call_id, name, result, is_error):
        self._rec(
            "tool_finish",
            trace_id=ctx.trace_id,
            call_id=call_id,
            name=name,
            is_error=is_error,
        )

    def on_event(self, ctx, *, kind, **data):
        self._rec("event", trace_id=ctx.trace_id, kind=kind, **data)

    def on_error(self, ctx, *, message):
        self._rec("error", trace_id=ctx.trace_id, message=message)

    def on_run_finish(self, ctx, *, status, output=None):
        self._rec("run_finish", trace_id=ctx.trace_id, status=status, output=output)

    def flush(self):
        self._rec("flush")

    def close(self):
        self._rec("close")

    # ── convenience for assertions ──────────────────────────────────────────
    def hooks(self) -> list[str]:
        """The ordered list of hook names recorded (ignoring payloads)."""
        return [h for h, _ in self.calls]

    def of(self, hook: str) -> list[dict[str, Any]]:
        """All payloads recorded for a given hook name."""
        return [p for h, p in self.calls if h == hook]
