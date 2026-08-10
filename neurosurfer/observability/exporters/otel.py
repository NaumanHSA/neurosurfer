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
import logging
from time import perf_counter
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neurosurfer.llm.types import Usage
    from neurosurfer.observability.context import TraceContext
    from neurosurfer.tools.base import ToolResult

from .base import TraceExporter

logger = logging.getLogger("neurosurfer.observability")


def _short(value: Any, limit: int = 2000) -> str:
    try:
        s = value if isinstance(value, str) else json.dumps(value, default=str)
    except Exception:  # noqa: BLE001
        s = str(value)
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _root_cause(exc: BaseException) -> str:
    """The innermost exception, named. What a caller can act on.

    A refused connection arrives wrapped four deep — ``requests.ConnectionError``
    over ``MaxRetryError`` over ``NewConnectionError`` over the ``OSError`` — and
    only the last one says *why*. The chain mixes ``__cause__`` (``raise … from``)
    with ``__context__`` (a bare ``raise`` inside ``except``), so both are followed.
    """
    seen: set[int] = {id(exc)}
    root = exc
    while (nxt := root.__cause__ or root.__context__) is not None and id(nxt) not in seen:
        seen.add(id(nxt))
        root = nxt
    return f"{type(root).__name__}: {root}"


class _QuietSpanExporter:
    """Wraps the OTLP exporter so an absent collector costs one warning, once.

    Two problems, both of which only appear when nothing is listening.

    **It was loud.** ``OTLPSpanExporter.export`` lets a transport error
    propagate — it retries a ``ConnectionError`` once and then re-raises — and
    ``BatchSpanProcessor`` turns that into ``logger.exception``, a full
    traceback *per batch*. A graph run emits a span per node, so the run's own
    output vanished under stacks that name no neurosurfer frame.

    **It was slow, which was the worse half.** ``force_flush`` is a *blocking*
    export on the calling thread, and the exporters are flushed at every run
    finish. On Windows a connect to a closed port is not refused instantly the
    way it is on Linux — the stack retries the SYN for ~2s. ``localhost``
    resolves to both ``::1`` and ``127.0.0.1``, so one attempt costs ~4s, and
    the exporter's blind retry doubles it: **~8.2s per flush, measured**, on a
    machine where the same misconfiguration costs ~0ms on Linux. Tutorial 03's
    router cell took 74s against a model that answered in under two.

    So the first failure **disables the exporter for the session**. Continuing
    to try was the original choice — on the theory that a collector might come
    up later — and 8.2s of dead air per node is far too much to pay for that
    chance. A run is not the place to keep probing a socket nobody answered.

    Deliberately not a ``SpanExporter`` subclass: that would need OpenTelemetry
    imported at module level, and this module keeps the SDK out of a base
    install's import path. ``BatchSpanProcessor`` duck-types its exporter —
    ``export``, ``force_flush``, ``shutdown``, nothing else.
    """

    def __init__(self, inner: Any, failure_result: Any) -> None:
        self._inner = inner
        self._failure = failure_result
        self._dead = False

    def _disable(self, exc: BaseException, seconds: float) -> None:
        if self._dead:
            return
        self._dead = True
        logger.warning(
            "Trace exporter 'otel': the OTLP collector at %s is not reachable (%s). "
            "That attempt blocked the run for %.1fs, so tracing is now disabled for "
            "this session and later spans are dropped without touching the network. "
            "Point OTEL_EXPORTER_OTLP_ENDPOINT at a running collector, or set "
            "NEUROSURFER_EXPORTERS=none to keep it off from the start.",
            getattr(self._inner, "_endpoint", "the configured endpoint"),
            _root_cause(exc),
            seconds,
        )

    def _attempt(self, call: Any, on_dead: Any) -> Any:
        if self._dead:
            return on_dead
        started = perf_counter()
        try:
            return call()
        except Exception as e:  # noqa: BLE001 — a dead collector must never break a run
            self._disable(e, perf_counter() - started)
            return on_dead

    def export(self, spans: Any) -> Any:
        return self._attempt(lambda: self._inner.export(spans), self._failure)

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return self._attempt(lambda: self._inner.force_flush(timeout_millis), False)

    def shutdown(self) -> None:
        self._attempt(self._inner.shutdown, None)


class OtelExporter(TraceExporter):
    name = "otel"

    def __init__(self, *, service_name: str = "neurosurfer") -> None:
        # Imported here so a base install without the `observability` extra never
        # pays for OpenTelemetry (the registry turns ImportError into a warn+skip).
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExportResult

        self._trace_api = __import__("opentelemetry.trace", fromlist=["trace"])
        provider = TracerProvider(
            resource=Resource.create({"service.name": service_name})
        )
        # Wrapped, so an absent collector is one warning rather than a traceback
        # per batch — see :class:`_QuietSpanExporter`.
        exporter = _QuietSpanExporter(OTLPSpanExporter(), SpanExportResult.FAILURE)
        provider.add_span_processor(BatchSpanProcessor(exporter))
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
        self, ctx, *, usage: Usage, model, stop_reason, input=None, output=None
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
