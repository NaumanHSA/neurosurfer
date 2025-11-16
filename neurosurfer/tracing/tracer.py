# neurosurfer/tracing/workflow.py
from __future__ import annotations

from typing import Any, Dict, Optional, List, Literal
import time
import logging

from pydantic import BaseModel

from .models import TraceStep, TraceResult
from .span import SpanTracer, RichTracer, NullSpanTracer


logger = logging.getLogger("neurosurfer.tracing.tracer")
RICH_LOG_TYPES_MAPPING = {
    "info": "[bold green]INFO: {message}[/bold green]",
    "warning": "[bold yellow]WARNING: {message}[/bold yellow]",
    "error": "[bold red]ERROR: {message}[/bold red]",
    "debug": "[bold blue]DEBUG: {message}[/bold blue]",
}

class TracerConfig(BaseModel):
    """
    Configuration options for Tracer.

    Attributes:
        enabled:
            If False, `step(...)` becomes a no-op. Your code can always call it.
        log_steps:
            If True, each step prints human-readable spans via `span_tracer`.
        max_output_preview_chars:
            When you store large outputs in `t.add(...)`, you may choose to
            also store a shortened preview in `outputs["preview"]`.
            This class itself doesn't enforce truncation — it's up to your
            usage — but the parameter is here for convenience / future use.
    """

    enabled: bool = True
    log_steps: bool = False
    max_output_preview_chars: int = 4000


class Tracer:
    """
    Unified, simple tracing class for agents / graphs.

    Usage example inside an Agent:

        class Agent:
            def __init__(..., enable_tracing: bool = False, log_tracing_steps: bool = False):
                self.tracer = Tracer(
                    config=TracerConfig(
                        enabled=enable_tracing,
                        log_steps=log_tracing_steps,
                    ),
                    span_tracer=RichTracer() if log_tracing_steps else NullSpanTracer(),
                    meta={"agent_type": "generic_agent"},
                )

            def run(...):
                inputs = {...}
                with self.tracer.step(
                    kind="llm",
                    label="agent.llm.ask",
                    inputs=inputs,
                    node_id=context.get("node_id"),
                    agent_id="main_agent",
                ) as t:
                    res = self.llm.ask(**inputs)
                    norm = normalize_tool_observation(res)
                    t.add(output=norm, raw_response=res)

                return AgentResult(
                    ...,
                    traces=self.tracer.results,
                )

    Key properties:
      - If `config.enabled=False`, `step(...)` returns a no-op context manager
        and no steps are recorded — your calling code doesn't change.
      - Each step has:
          * auto-incremented step_id
          * timing (started_at, duration_ms)
          * ok/error (based on exceptions)
          * inputs/outputs/meta dicts
      - `results` returns a `TraceResult` Pydantic model.
    """

    def __init__(
        self,
        *,
        config: Optional[TracerConfig] = None,
        span_tracer: Optional[SpanTracer] = None,
        meta: Optional[Dict[str, Any]] = None,
        logger_: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config or TracerConfig()
        self.logger = logger_ or logger

        if span_tracer is not None:
            self._span_tracer = span_tracer
        else:
            self._span_tracer = RichTracer() if self.config.log_steps else NullSpanTracer()

        self._steps: list[TraceStep] = []
        self._meta: Dict[str, Any] = dict(meta or {})
        self._counter: int = 0

        # NEW: track nesting depth of step contexts
        self._depth: int = 0


    # ------------------------------------------------------------------
    # Global metadata
    # ------------------------------------------------------------------
    @property
    def meta(self) -> Dict[str, Any]:
        """
        Metadata attached to the whole trace run (e.g. graph_name, run_id).
        """
        return self._meta

    def set_meta(self, **kwargs: Any) -> None:
        """
        Update global metadata for this tracer.
        """
        self._meta.update(kwargs)

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    @property
    def results(self) -> TraceResult:
        """
        Structured tracing results for this run.
        """
        return TraceResult(steps=list(self._steps), meta=dict(self._meta))

    # ------------------------------------------------------------------
    # Public API: .step(...)
    # ------------------------------------------------------------------

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
        Create a traced step context.

        Example:
            with tracer.step(kind="llm", label="agent.llm.ask", inputs=inp) as t:
                res = self.llm.ask(**inp)
                norm = normalize(res)
                t.add(output=norm, raw=res)

        If tracing is disabled (config.enabled=False), this returns a
        no-op context manager and records nothing.
        """
        if not self.config.enabled:
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record_step(self, raw: Dict[str, Any]) -> None:
        """
        Convert raw dict from _TraceStepContext into a TraceStep and store it.
        """
        try:
            step = TraceStep(**raw)
            self._steps.append(step)
        except Exception as e:  # pragma: no cover - defensive
            self.logger.warning("Failed to record trace step: %s", e)
    
    def _span(self, name: str, attrs: Dict[str, Any]):
        """
        Internal helper to create a low-level span (or a no-op if log_steps=False).

        Here we inject `_level` automatically based on current self._depth.
        """
        # make a shallow copy so we don't mutate caller's dict
        attrs = dict(attrs)
        # depth 1 (outermost) -> level 0 indent
        level = max(self._depth - 1, 0)
        attrs["_level"] = level

        if not self.config.log_steps:
            return NullSpanTracer().span(name, attrs)
        return self._span_tracer.span(name, attrs)

    def _log_line(self, *, step_id: int, indent_level: int, message: str, type: Optional[str] = None) -> None:
        """
        Emit a single inline log line for a step, at the correct indentation.
        Respects `log_steps` and uses the underlying span backend (rich/logger/print).
        """
        if not self.config.log_steps:
            return

        indent = " " * (indent_level * 4)
        line = f"{indent}    {RICH_LOG_TYPES_MAPPING[type].format(message=message)}"
        st = self._span_tracer
        try:
            from rich.console import Console
            # RichTracer-style
            console: Optional[Console] = getattr(st, "console", None)
            if console is not None:
                console.print(line)
                return

            # LoggerTracer-style
            logger_obj = getattr(st, "logger", None)
            if logger_obj is not None:
                logger_obj.info(line)
                return

            # Fallback: plain print
            print(line)
        except Exception:
            # Last-resort fallback, shouldn't really happen
            print(line)
            
    def reset(self):
        self._steps = []
        self._counter = 0
        self._depth = 0
    

class _TraceStepContext:
    """
    Context manager returned by Tracer.step().

    - On __enter__:
        * records start time
        * optionally logs a span "▶ step.<kind>"
    - On __exit__:
        * sets duration_ms
        * sets ok/error
        * records a TraceStep via Tracer._record_step()
        * does NOT suppress exceptions
    """

    def __init__(
        self,
        *,
        tracer: Tracer,
        step_id: int,
        kind: str,
        label: Optional[str],
        inputs: Dict[str, Any],
        node_id: Optional[str],
        agent_id: Optional[str],
        meta: Dict[str, Any],
    ) -> None:
        self._tracer = tracer
        self._step_data: Dict[str, Any] = {
            "step_id": step_id,
            "kind": kind,
            "label": label,
            "node_id": node_id,
            "agent_id": agent_id,
            "inputs": inputs,
            "outputs": {},
            "meta": meta,
            "started_at": 0.0,
            "duration_ms": 0,
            "ok": True,
            "error": None,
            "logs": [],
        }
        self._span_cm = None

    def __enter__(self) -> "_TraceStepContext":
        # increase nesting depth for this tracer
        self._tracer._depth += 1

        self._step_data["started_at"] = time.time()
        self._span_cm = self._tracer._span(
            name=f"step.{self._step_data['kind']}",
            attrs={
                "step_id": self._step_data["step_id"],
                "label": self._step_data["label"],
                "node_id": self._step_data["node_id"],
                "agent_id": self._step_data["agent_id"],
            },
        )
        self._span_cm.__enter__()
        return self

    def set_error(self, error: str) -> None:
        self._step_data["error"] = error
        self._step_data["ok"] = False

    def add_meta(self, **kwargs: Dict[str, Any]) -> None:
        self._step_data["meta"].update(kwargs)
    
    def outputs(self, **kwargs: Any) -> None:
        """
        Add arbitrary key/value pairs to `outputs`.

        Typical usage:
            t.outputs(output=normalized_result)
            t.outputs(system_prompt=sys, user_prompt=usr)
        """
        outputs: dict = self._step_data.setdefault("outputs", {})
        outputs.update(kwargs)

    def inputs(self, **kwargs: Any) -> None:
        """
        Add arbitrary key/value pairs to `inputs`.

        Typical usage:
            t.inputs(output=normalized_result)
            t.inputs(system_prompt=sys, user_prompt=usr)
        """
        inputs: dict = self._step_data.setdefault("inputs", {})
        inputs.update(kwargs)

    def log(self, message: str, type: Literal["info", "warning", "error", "debug"] = "info", **data: Any) -> None:
        """
        Add an internal log line to this step.

        - Stored in the structured trace result (step.logs)
        - Printed at the same indentation level as the step spans
          when `log_steps=True`.
        """
        ts = time.time()

        # store in structured trace
        logs: list = self._step_data.setdefault("logs", [])
        logs.append(
            {
                "ts": ts,
                "message": message,
                "data": data or {},
                "type": type,
            }
        )

        # print as a log line (aligned with this step)
        indent_level = max(self._tracer._depth - 1, 0)
        self._tracer._log_line(
            step_id=self._step_data["step_id"],
            indent_level=indent_level,
            message=message,
            type=type,
        )

    def start(self) -> "_TraceStepContext":
        self.__enter__()
        return self

    def close(self) -> None:
        self.__exit__(None, None, None)

    def __exit__(self, exc_type, exc, tb) -> bool:
        end = time.time()
        self._step_data["duration_ms"] = int(
            (end - self._step_data["started_at"]) * 1000
        )
        if exc_type is not None:
            self._step_data["ok"] = False
            self._step_data["error"] = repr(exc)

        # end span
        self._span_cm.__exit__(exc_type, exc, tb)

        # decrease nesting depth
        self._tracer._depth -= 1

        # record step
        self._tracer._record_step(self._step_data)

        # do not suppress exceptions
        return False


class _NoOpStepContext:
    """
    No-op context manager returned when tracing is disabled.

    - __enter__ returns self
    - .add(...) does nothing
    - __exit__ does nothing and does NOT suppress exceptions
    """

    def __enter__(self) -> "_NoOpStepContext":
        return self

    def add(self, **kwargs: Any) -> None:
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False
    
    def log(self, message: str, **data: Any) -> None:
        return None
