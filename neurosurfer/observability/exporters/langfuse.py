"""Langfuse exporter — the batteries-included LLM-observability backend.

Maps an agent run onto Langfuse's trace model (SDK v2, handle-based API):

    trace  (the run)                        ← on_run_start / on_run_finish
    ├─ generation  (one LLM turn)           ← on_turn   (model + token usage → cost)
    ├─ span        (one tool call)          ← on_tool_start / on_tool_finish
    └─ event       (mode change / compaction) ← on_event

Token usage on the generation is what lets Langfuse compute cost automatically.
Credentials come from ``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY`` /
``LANGFUSE_HOST`` (the SDK reads them when the ctor args are ``None``); ``LANGFUSE_HOST``
points at a self-hosted instance, defaulting to Langfuse Cloud.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neurosurfer.llm.types import Usage
    from neurosurfer.observability.context import TraceContext
    from neurosurfer.tools.base import ToolResult

from .base import TraceExporter


class LangfuseExporter(TraceExporter):
    name = "langfuse"

    def __init__(self, *, service_name: str = "neurosurfer") -> None:
        # Imported here so a base install never requires langfuse (the registry
        # turns a missing SDK into a warn+skip).
        from langfuse import Langfuse

        self._service_name = service_name
        self._client = Langfuse()  # reads LANGFUSE_* from the environment
        # trace_id → trace handle; call_id → span handle (tools in flight).
        self._traces: dict[str, Any] = {}
        self._spans: dict[str, Any] = {}

    # ── lifecycle ───────────────────────────────────────────────────────────
    def on_run_start(self, ctx: TraceContext, *, name: str, input: Any = None) -> None:
        self._traces[ctx.trace_id] = self._client.trace(
            id=ctx.trace_id,
            name=name,
            session_id=ctx.session_id,
            input=input,
            metadata={"service": self._service_name, **ctx.metadata},
        )

    def on_turn(self, ctx, *, usage: Usage, model, stop_reason, output=None) -> None:
        trace = self._traces.get(ctx.trace_id)
        if trace is None:
            return
        trace.generation(
            name="llm-turn",
            model=model,
            usage={
                "input": int(usage.input_tokens),
                "output": int(usage.output_tokens),
                "unit": "TOKENS",
            },
            output=output,
            metadata={"stop_reason": stop_reason},
        )

    def on_tool_start(self, ctx, *, call_id, name, args) -> None:
        trace = self._traces.get(ctx.trace_id)
        if trace is None:
            return
        self._spans[call_id] = trace.span(name=f"tool:{name}", input=args)

    def on_tool_finish(self, ctx, *, call_id, name, result: ToolResult, is_error) -> None:
        span = self._spans.pop(call_id, None)
        if span is None:
            return
        span.end(
            output=getattr(result, "content", None),
            level="ERROR" if is_error else "DEFAULT",
            status_message="tool error" if is_error else None,
        )

    def on_event(self, ctx, *, kind, **data) -> None:
        trace = self._traces.get(ctx.trace_id)
        if trace is None:
            return
        trace.event(name=kind, metadata=data)

    def on_error(self, ctx, *, message) -> None:
        trace = self._traces.get(ctx.trace_id)
        if trace is None:
            return
        trace.update(output={"error": message})

    def on_run_finish(self, ctx, *, status, output=None) -> None:
        trace = self._traces.pop(ctx.trace_id, None)
        if trace is None:
            return
        # Close any tool spans left dangling by an early exit.
        for call_id in [cid for cid, sp in self._spans.items() if sp is not None]:
            try:
                self._spans.pop(call_id).end()
            except Exception:  # noqa: BLE001
                pass
        trace.update(output=output, metadata={"status": status})

    def flush(self) -> None:
        try:
            self._client.flush()
        except Exception:  # noqa: BLE001
            pass

    def close(self) -> None:
        self.flush()
