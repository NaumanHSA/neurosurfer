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
        # span_id → (kind, handle): the run's own handle. kind is "trace" (root)
        # or "span" (a sub-agent nested under its parent). call_id → tool span.
        self._runs: dict[str, tuple[str, Any]] = {}
        self._spans: dict[str, Any] = {}

    def _handle(self, ctx: TraceContext):
        entry = self._runs.get(ctx.span_id)
        return entry[1] if entry else None

    # ── lifecycle ───────────────────────────────────────────────────────────
    def on_run_start(self, ctx: TraceContext, *, name: str, input: Any = None) -> None:
        parent = self._runs.get(ctx.parent_span_id) if ctx.parent_span_id else None
        if parent is not None:
            # Nested run (sub-agent / node): a span under the parent's handle,
            # sharing the same trace.
            handle = parent[1].span(
                id=ctx.span_id,
                name=name,
                input=input,
                metadata={"service": self._service_name, **ctx.metadata},
            )
            self._runs[ctx.span_id] = ("span", handle)
        else:
            handle = self._client.trace(
                id=ctx.trace_id,
                name=name,
                session_id=ctx.session_id,
                input=input,
                metadata={"service": self._service_name, **ctx.metadata},
            )
            self._runs[ctx.span_id] = ("trace", handle)

    def on_turn(self, ctx, *, usage: Usage, model, stop_reason, input=None, output=None) -> None:
        handle = self._handle(ctx)
        if handle is None:
            return
        # We report *tokens only* — cost is the backend's job, computed from its own
        # per-model price table. Send both usage shapes for broad server compat: the
        # legacy `usage` (Langfuse ≤ v2 servers) and `usage_details` (v3-native, whose
        # keys line up with a model's `prices` map so it prices correctly). Verified
        # the two together do NOT double-count. Cache tokens surface when present.
        usage_details = {
            "input": int(usage.input_tokens),
            "output": int(usage.output_tokens),
        }
        if usage.cache_read_input_tokens:
            usage_details["cache_read"] = int(usage.cache_read_input_tokens)
        if usage.cache_creation_input_tokens:
            usage_details["cache_write"] = int(usage.cache_creation_input_tokens)
        handle.generation(
            name="llm-turn",
            model=model,
            input=input,
            usage={
                "input": int(usage.input_tokens),
                "output": int(usage.output_tokens),
                "unit": "TOKENS",
            },
            usage_details=usage_details,
            output=output,
            metadata={"stop_reason": stop_reason},
        )

    def on_tool_start(self, ctx, *, call_id, name, args) -> None:
        handle = self._handle(ctx)
        if handle is None:
            return
        self._spans[call_id] = handle.span(name=f"tool:{name}", input=args)

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
        handle = self._handle(ctx)
        if handle is None:
            return
        handle.event(name=kind, metadata=data)

    def on_error(self, ctx, *, message) -> None:
        handle = self._handle(ctx)
        if handle is not None:
            handle.update(output={"error": message})

    def on_run_finish(self, ctx, *, status, output=None) -> None:
        entry = self._runs.pop(ctx.span_id, None)
        if entry is None:
            return
        kind, handle = entry
        if kind == "span":
            handle.end(output=output, metadata={"status": status})
        else:
            handle.update(output=output, metadata={"status": status})

    def flush(self) -> None:
        try:
            self._client.flush()
        except Exception:  # noqa: BLE001
            pass

    def close(self) -> None:
        self.flush()
