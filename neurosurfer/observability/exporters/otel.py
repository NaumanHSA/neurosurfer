"""OpenTelemetry exporter — the vendor-neutral substrate.

Emits GenAI-semantic-convention spans over OTLP, so any OTel backend ingests a
neurosurfer run: Arize Phoenix, Grafana Tempo, Datadog, Honeycomb, or Langfuse's
own OTLP endpoint. Configuration is the standard OTel environment
(``OTEL_EXPORTER_OTLP_ENDPOINT`` etc.) — the OTLP exporter reads it directly.

Span shape (one trace per agent run):

    <AgentType>.run                 (root, gen_ai.operation.name=agent)
    ├─ llm.turn                     (gen_ai.operation.name=chat + gen_ai.usage.*)
    ├─ tool.<name>                  (gen_ai.tool.name, args, result)
    └─ …

We build our **own** ``TracerProvider`` rather than touching the global one, so a
host app that already configured OpenTelemetry is never disturbed; spans are
parented manually (``set_span_in_context``) because the run's lifecycle crosses
async ``await`` boundaries where OTel's implicit context vars don't hold.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neurosurfer.llm.types import Usage
    from neurosurfer.observability.context import TraceContext
    from neurosurfer.tools.base import ToolResult

from .base import TraceExporter


def _short(value: Any, limit: int = 2000) -> str:
    try:
        s = value if isinstance(value, str) else json.dumps(value, default=str)
    except Exception:  # noqa: BLE001
        s = str(value)
    return s if len(s) <= limit else s[: limit - 1] + "…"


class OtelExporter(TraceExporter):
    name = "otel"

    def __init__(self, *, service_name: str = "neurosurfer") -> None:
        # Imported here so a base install without the `observability` extra never
        # pays for OpenTelemetry (the registry turns ImportError into a warn+skip).
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        self._trace_api = __import__("opentelemetry.trace", fromlist=["trace"])
        provider = TracerProvider(
            resource=Resource.create({"service.name": service_name})
        )
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        self._provider = provider
        self._tracer = provider.get_tracer("neurosurfer")
        # per-run state: span_id → {"root": Span, "ctx": Context, "tools": {call_id: Span}}
        self._runs: dict[str, dict[str, Any]] = {}

    # ── lifecycle ───────────────────────────────────────────────────────────
    def on_run_start(self, ctx: TraceContext, *, name: str, input: Any = None) -> None:
        # Nest under the enclosing run's span (sub-agent / node), if any.
        parent_run = self._runs.get(ctx.parent_span_id) if ctx.parent_span_id else None
        parent_ctx = parent_run["ctx"] if parent_run else None
        root = self._tracer.start_span(
            name,
            context=parent_ctx,
            attributes={
                "gen_ai.operation.name": "agent",
                "gen_ai.system": str(ctx.metadata.get("provider", "")),
                "gen_ai.request.model": str(ctx.metadata.get("model") or ""),
                "neurosurfer.trace_id": ctx.trace_id,
                **({"gen_ai.prompt": _short(input)} if input else {}),
            },
        )
        child_ctx = self._trace_api.set_span_in_context(root)
        self._runs[ctx.span_id] = {"root": root, "ctx": child_ctx, "tools": {}}

    def on_turn(
        self, ctx, *, usage: Usage, model, stop_reason, output=None
    ) -> None:
        run = self._runs.get(ctx.span_id)
        if run is None:
            return
        span = self._tracer.start_span(
            "llm.turn",
            context=run["ctx"],
            attributes={
                "gen_ai.operation.name": "chat",
                "gen_ai.request.model": str(model or ""),
                "gen_ai.response.finish_reason": str(stop_reason),
                "gen_ai.usage.input_tokens": int(usage.input_tokens),
                "gen_ai.usage.output_tokens": int(usage.output_tokens),
                **({"gen_ai.completion": _short(output)} if output else {}),
            },
        )
        span.end()

    def on_tool_start(self, ctx, *, call_id, name, args) -> None:
        run = self._runs.get(ctx.span_id)
        if run is None:
            return
        span = self._tracer.start_span(
            f"tool.{name}",
            context=run["ctx"],
            attributes={
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": name,
                "gen_ai.tool.call.id": call_id,
                "gen_ai.tool.arguments": _short(args),
            },
        )
        run["tools"][call_id] = span

    def on_tool_finish(self, ctx, *, call_id, name, result: ToolResult, is_error) -> None:
        run = self._runs.get(ctx.span_id)
        if run is None:
            return
        span = run["tools"].pop(call_id, None)
        if span is None:
            return
        span.set_attribute("gen_ai.tool.result", _short(getattr(result, "content", "")))
        if is_error:
            from opentelemetry.trace import Status, StatusCode

            span.set_status(Status(StatusCode.ERROR))
        span.end()

    def on_event(self, ctx, *, kind, **data) -> None:
        run = self._runs.get(ctx.span_id)
        if run is None:
            return
        run["root"].add_event(kind, attributes={k: _short(v) for k, v in data.items()})

    def on_error(self, ctx, *, message) -> None:
        run = self._runs.get(ctx.span_id)
        if run is None:
            return
        from opentelemetry.trace import Status, StatusCode

        run["root"].set_status(Status(StatusCode.ERROR, message))
        run["root"].record_exception(RuntimeError(message))

    def on_run_finish(self, ctx, *, status, output=None) -> None:
        run = self._runs.pop(ctx.span_id, None)
        if run is None:
            return
        # End any tool spans left dangling by an early exit.
        for span in run["tools"].values():
            span.end()
        root = run["root"]
        if output:
            root.set_attribute("gen_ai.completion", _short(output))
        root.set_attribute("neurosurfer.status", str(status))
        root.end()

    def flush(self) -> None:
        try:
            self._provider.force_flush()
        except Exception:  # noqa: BLE001
            pass

    def close(self) -> None:
        try:
            self._provider.shutdown()
        except Exception:  # noqa: BLE001
            pass
