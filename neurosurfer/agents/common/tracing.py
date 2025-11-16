from __future__ import annotations

from typing import Any, Dict, Optional, Union, Callable, List
import time
import logging
from pydantic import BaseModel, Field


# -----------------------------------------------------------------------------
# Low-level span tracers (your existing stuff, just slightly refactored)
# -----------------------------------------------------------------------------

default_logger = logging.getLogger("neurosurfer.graph")


class SpanTracer:
    """
    Low-level span tracer interface.

    Used internally by WorkflowTracer to optionally print/hook human-readable
    trace lines. This is your old `Tracer` interface, renamed to avoid confusion
    with the new high-level WorkflowTracer.
    """
    def span(self, name: str, attrs: Optional[Dict[str, Any]] = None):
        raise NotImplementedError


class ConsoleTracer(SpanTracer):
    """Writes spans to stdout (print)."""

    def span(self, name: str, attrs: Optional[Dict[str, Any]] = None):
        return _Span(name, attrs, sink=print)


try:
    from rich.console import Console

    class RichTracer(SpanTracer):
        """
        Tracer that writes spans using `rich`. Supports either:
        - dict attrs (printed as key=value)
        - str attrs (printed as a message)
        """

        def __init__(self, console: Optional[Console] = None):
            self.console = console or Console()

        def span(
            self,
            name: str,
            attrs: Optional[Union[Dict[str, Any], str]] = None,
        ):
            def sink(msg: str):
                # no extra blank line spam
                self.console.print(f"[dim]{msg}[/dim]", end="\n")
            return _Span(name, attrs, sink=sink)

except Exception:  # pragma: no cover
    default_logger.warning("Rich not installed; falling back to ConsoleTracer")
    RichTracer = ConsoleTracer  # type: ignore[misc]


class LoggerTracer(SpanTracer):
    """Writes spans to Python logging (INFO)."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or default_logger

    def span(self, name: str, attrs: Optional[Dict[str, Any]] = None):
        return _Span(
            name, attrs, sink=lambda *args, **kw: self.logger.info(args[0])
        )


class MemorySpanTracer(SpanTracer):
    """
    Collects spans in memory (mostly for debugging / tests).
    """
    def __init__(self):
        self.events: List[Dict[str, Any]] = []

    def span(self, name: str, attrs: Optional[Dict[str, Any]] = None):
        tracer = self

        class _MemSpan(_Span):
            def __enter__(self):
                self.t0 = time.time()
                tracer.events.append(
                    {"name": self.name, "attrs": self.attrs, "ts": self.t0, "phase": "start"}
                )
                return self

            def __exit__(self, exc_type, exc, tb):
                dt = time.time() - self.t0
                tracer.events.append(
                    {
                        "name": self.name,
                        "attrs": self.attrs,
                        "ts": time.time(),
                        "phase": "end",
                        "ms": int(dt * 1000),
                        "error": bool(exc_type),
                    }
                )

        return _MemSpan(name, attrs, sink=lambda *_: None)


class _Span:
    """
    Simple span context manager with a pluggable sink.

    - attrs can be:
        * dict  -> formatted as key=value
        * str   -> treated as a human-readable message
        * None  -> no suffix
    """

    def __init__(self, name: str, attrs=None, sink=print):
        self.name = name
        self.attrs = attrs
        self._sink = sink

    def _format_suffix(self) -> str:
        if isinstance(self.attrs, dict):
            if not self.attrs:
                return ""
            kv = " ".join(f"{k}={v!r}" for k, v in self.attrs.items())
            return " " + kv
        elif isinstance(self.attrs, str):
            return " — " + self.attrs
        else:
            return ""

    def __enter__(self):
        self.t0 = time.time()
        suffix = self._format_suffix()
        self._sink(f"[trace] ▶ {self.name}{suffix}")
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.time() - self.t0
        suffix = self._format_suffix()
        self._sink(
            f"[trace] ◀ {self.name}{suffix} took {dt:.3f}s; error={bool(exc_type)}"
        )


class NullSpanTracer(SpanTracer):
    """No-op span tracer."""

    def span(self, name: str, attrs=None):
        return _Span(name, attrs, sink=lambda *_: None)


# -----------------------------------------------------------------------------
# High-level unified tracing (simple .step API + Pydantic results)
# -----------------------------------------------------------------------------
class TraceStep(BaseModel):
    """
    One traced step in a workflow.

    Flexible enough to cover:
      - LLM calls
      - tool calls
      - any custom step
    """

    step_id: int
    kind: str                   # e.g. "llm", "tool", "graph", "other"
    label: Optional[str] = None # free-form, e.g. "manager.compose"
    node_id: Optional[str] = None
    agent_id: Optional[str] = None

    started_at: float
    duration_ms: int

    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    meta: Dict[str, Any] = Field(default_factory=dict)

    ok: bool
    error: Optional[str] = None


class TraceResult(BaseModel):
    """
    Collection of steps + optional metadata for the whole run.
    """

    steps: List[TraceStep] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)

    def summary(self) -> str:
        ok_count = sum(1 for s in self.steps if s.ok)
        err_count = sum(1 for s in self.steps if not s.ok)
        return (
            f"TraceResult(steps={len(self.steps)}, ok={ok_count}, "
            f"errors={err_count})"
        )


class WorkflowTracer:
    """
    Unified, simple tracing class.

    - Construct once per agent / graph run.
    - Use `with tracer.step(...):` around any operation you want to trace.
    - At the end, read `tracer.results` and attach it to your AgentResult.

    Design goals:
      - Zero change in calling code when tracing is disabled.
      - Minimal boilerplate at call site.
      - Flexible `add(...)` method to store whatever you want.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        log_steps: bool = False,
        span_tracer: Optional[SpanTracer] = None,
        meta: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Args:
            enabled:
                If False, `step(...)` returns a no-op context manager. Your
                code stays the same, but nothing is recorded.
            log_steps:
                If True, each step prints human-readable spans via `span_tracer`.
            span_tracer:
                Low-level span tracer used when log_steps=True. Defaults to
                ConsoleTracer if log_steps=True, otherwise NullSpanTracer.
            meta:
                Optional metadata for the whole run (e.g. graph_name, run_id, etc.).
            logger:
                Optional logger to use for internal warnings (if any).
        """
        self.enabled = bool(enabled)
        self.log_steps = bool(log_steps)
        self.logger = logger or default_logger

        if span_tracer is not None:
            self._span_tracer = span_tracer
        else:
            self._span_tracer = ConsoleTracer() if self.log_steps else NullSpanTracer()

        self._steps: List[TraceStep] = []
        self._meta: Dict[str, Any] = dict(meta or {})
        self._counter: int = 0

    # ----------------------------
    # Public API
    # ----------------------------

    @property
    def meta(self) -> Dict[str, Any]:
        return self._meta

    def set_meta(self, **kwargs: Any) -> None:
        """
        Update global metadata for this tracer. Example:

            tracer.set_meta(graph_name="blog_flow", run_id="123")
        """
        self._meta.update(kwargs)

    @property
    def results(self) -> TraceResult:
        """
        Structured tracing results for this run. Attach this to Agent results.
        """
        return TraceResult(steps=list(self._steps), meta=dict(self._meta))

    def step(
        self,
        *,
        kind: str,
        label: Optional[str] = None,
        inputs: Optional[Dict[str, Any]] = None,
        node_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ):
        """
        Create a traced step.

        Usage:

            with tracer.step(kind="llm", label="structured_call", inputs=inp) as t:
                res = self.llm.ask(**inp)
                res_norm = normalize(res)
                t.add(output=res_norm, raw=res)

        The context manager will automatically:
          - assign step_id
          - capture start time
          - measure duration_ms
          - flag ok/error based on exceptions
          - store inputs + any outputs/meta you add via `t.add(...)`
        """
        if not self.enabled:
            return _NoOpStepContext()

        self._counter += 1
        step_id = self._counter

        return _TraceStepContext(
            tracer=self,
            step_id=step_id,
            kind=kind,
            label=label,
            inputs=inputs or {},
            node_id=node_id,
            agent_id=agent_id,
            meta=meta or {},
        )

    # ----------------------------
    # Internal (used by _TraceStepContext)
    # ----------------------------

    def _record_step(self, raw: Dict[str, Any]) -> None:
        """
        Convert raw dict from _TraceStepContext into a TraceStep and store it.
        """
        try:
            step = TraceStep(**raw)
            self._steps.append(step)
        except Exception as e:
            # Don't break user code if tracing data is weird; just log it.
            self.logger.warning("Failed to record trace step: %s", e)

    def _span(self, name: str, attrs: Dict[str, Any]):
        """
        Internal helper to create a low-level span (or a no-op if log_steps=False).
        """
        if not self.log_steps:
            return NullSpanTracer().span(name, attrs)
        return self._span_tracer.span(name, attrs)


class _TraceStepContext:
    """
    Context manager returned by WorkflowTracer.step().

    - On __enter__: starts timing, optionally logs start span, returns itself.
    - On __exit__: finalizes TraceStep, logs end span, and records step.
    - Provides .add(...) to attach outputs / extra fields.
    """

    def __init__(
        self,
        *,
        tracer: WorkflowTracer,
        step_id: int,
        kind: str,
        label: Optional[str],
        inputs: Dict[str, Any],
        node_id: Optional[str],
        agent_id: Optional[str],
        meta: Dict[str, Any],
    ) -> None:
        self._tracer = tracer
        self._data: Dict[str, Any] = {
            "step_id": step_id,
            "kind": kind,
            "label": label,
            "node_id": node_id,
            "agent_id": agent_id,
            "inputs": inputs,
            "outputs": {},
            "meta": meta,
            "started_at": 0.0,       # set in __enter__
            "duration_ms": 0,        # set in __exit__
            "ok": True,
            "error": None,
        }
        self._span_cm = None

    def __enter__(self) -> "_TraceStepContext":
        self._data["started_at"] = time.time()
        self._span_cm = self._tracer._span(
            name=f"Step Number: {self._data['step_id']} | step.{self._data['kind']}",
            attrs={
                "label": self._data["label"],
                "node_id": self._data["node_id"],
                "agent_id": self._data["agent_id"],
            },
        )
        self._span_cm.__enter__()
        return self

    def add(self, **kwargs: Any) -> None:
        """
        Add arbitrary fields to the `outputs` or `meta` or wherever.

        Common pattern:
            t.add(output=res_norm)
            t.add(system_prompt=sys_prompt, user_prompt=user_prompt)
        """
        outputs: dict = self._data.setdefault("outputs", {})
        outputs.update(kwargs)

    def __exit__(self, exc_type, exc, tb) -> bool:
        end = time.time()
        self._data["duration_ms"] = int((end - self._data["started_at"]) * 1000)
        if exc_type is not None:
            self._data["ok"] = False
            self._data["error"] = repr(exc)

        # end span
        self._span_cm.__exit__(exc_type, exc, tb)  # type: ignore[union-attr]

        # record step on tracer
        self._tracer._record_step(self._data)

        # do NOT suppress exceptions
        return False


class _NoOpStepContext:
    """
    No-op context manager returned when tracing is disabled.

    - __enter__ returns self
    - .add(...) does nothing
    - __exit__ does nothing
    """

    def __enter__(self) -> "_NoOpStepContext":
        return self

    def add(self, **kwargs: Any) -> None:
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False
