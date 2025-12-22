# neurosurfer/tracing/workflow.py
from __future__ import annotations

from typing import Any, Dict, Optional, List, Literal
import time
import logging

from pydantic import BaseModel

from .models import TraceStep, TraceResult
from .span import SpanTracer, RichTracer, NullSpanTracer
from .step_context import TraceStepContext
from .render import render_trace_result
from .config import TracerConfig, RICH_LOG_TYPES_MAPPING

logger = logging.getLogger("neurosurfer.tracing.tracer")


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
        depth: int = 0,
        logger_: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config or TracerConfig()
        self.logger = logger_ or logger

        if span_tracer is not None:
            self._span_tracer = span_tracer
        else:
            self._span_tracer = RichTracer() if self.config.log_steps else NullSpanTracer()

        self._meta: Dict[str, Any] = dict(meta or {})
        self._result = TraceResult(meta=self._meta)  # single shared instance

        self._counter: int = 0
        self._depth: int = depth
        self._stream_started: set[int] = set()


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
        self._result.steps = sorted(self._result.steps, key=lambda s: s.step_id)
        return self._result

    # render traces as string or markdown for prettier printing
    def render(self, format: Literal["text", "markdown"] = "text") -> str:
        """
        Render the trace result as a string.
        """
        if format not in ["text", "markdown"]:
            format = "text"
        return render_trace_result(self.results, cfg=self.config, format=format)

    # ------------------------------------------------------------------
    # Public API: tracer(...)
    # ------------------------------------------------------------------
    def __call__(
        self,
        *,
        kind: str,
        start_message: Optional[str] = None,
        end_message: Optional[str] = None,
        label: Optional[str] = None,
        inputs: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> TraceStepContext:
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

        start_message = self._format_message((start_message or f"Starting {kind}...").strip("\\"), tag=f"[{agent_id}]")
        end_message = self._format_message((end_message or f"Completed {kind}!").strip("\\"), tag=f"[{agent_id}]")

        return TraceStepContext(
            tracer=self,
            step_id=step_id,
            kind=kind,
            start_message=start_message,
            end_message=end_message,
            label=label,
            inputs=inputs or {},
            agent_id=agent_id,
            meta=meta or {},
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _record_step(self, raw: Dict[str, Any]) -> None:
        """
        Convert raw dict from TraceStepContext into a TraceStep and store it.
        """
        try:
            step = TraceStep(**raw)
            self._result.steps.append(step)
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

    def _log_line(
        self,
        *,
        step_id: int,
        indent_level: int,
        message: str,
        type: Optional[str] = None,
        type_keyword: bool = True,
        stream: bool = False,
    ) -> None:
        """
        Emit a single inline log line for a step, at the correct indentation.
        Respects `log_steps` and uses the underlying span backend (rich/logger/print).

        - Non-stream (default): line-based logging with indentation and per-type formatting.
        - Stream=True: append chunks inline on a single indented line for this step_id,
        keeping indentation and colors even when message contains '\\n'.
        """
        if not self.config.log_steps:
            return
        
        if type not in RICH_LOG_TYPES_MAPPING:
            type = "neutral"

        prefix = " " * (indent_level * 4)
        # prefix = indent + "    "

        # ---------- STREAMING MODE ----------
        if stream:
            # lazy init state for streaming steps
            if not hasattr(self, "_stream_started"):
                self._stream_started: set[int] = set()
            if not hasattr(self, "_stream_at_line_start"):
                # True means "next char will be at the beginning of a logical line"
                self._stream_at_line_start: Dict[int, bool] = {}

            # pick formatter (colors / style)
            fmt = RICH_LOG_TYPES_MAPPING.get(type or "info", "{message}")
            if not type_keyword:
                # strip "DEBUG: " / "ERROR: " etc if present in format
                token = f"{(type or '').upper()}: "
                fmt = fmt.replace(token, "")

            # Try to use Rich console if available
            st = self._span_tracer
            console = None
            try:
                from rich.console import Console  # type: ignore
                console = getattr(st, "console", None)
                if not isinstance(console, Console):
                    console = None
            except Exception:
                console = None

            # split by lines but keep line breaks so we can re-indent after '\n'
            pieces = message.splitlines(keepends=True) or [""]

            # Initialize per-step state
            if step_id not in self._stream_started:
                self._stream_started.add(step_id)
                self._stream_at_line_start[step_id] = True

            def _write(text: str, end: str = ""):
                # Unified low-level writer for streaming
                if console is not None:
                    console.print(text, end=end, soft_wrap=False)
                else:
                    # don't use logger here; it always adds newlines
                    print(text, end=end, flush=True)

            # Walk through all pieces, respecting '\n' and line starts
            # pieces.append("\n\n")
            for piece in pieces:
                if piece == "":
                    continue

                has_newline = piece.endswith("\n")
                chunk = piece[:-1] if has_newline else piece

                # At the beginning of a logical line? Then print indent first.
                if self._stream_at_line_start.get(step_id, True):
                    _write(prefix)
                    self._stream_at_line_start[step_id] = False

                # Apply formatting (colors etc.)
                formatted = fmt.format(message=chunk)
                _write(formatted, end="")

                if has_newline:
                    # Finish this line and mark that next char is a new line
                    _write("\n")
                    self._stream_at_line_start[step_id] = True
            return

        # ---------- NON-STREAM MODE (line-based) ----------
        raw_lines = message.splitlines() or [""]
        fmt = RICH_LOG_TYPES_MAPPING.get(type or "info", "{message}")
        if not type_keyword:
            token = f"{(type or '').upper()}: "
            fmt = fmt.replace(token, "")

        formatted_lines = [
            prefix + fmt.format(message=line)
            for line in raw_lines
        ]

        st = self._span_tracer
        try:
            from rich.console import Console  # type: ignore
            console: Optional[Console] = getattr(st, "console", None)
            if console is not None:
                for line in formatted_lines:
                    console.print(line)
                return

            logger_obj = getattr(st, "logger", None)
            if logger_obj is not None:
                for line in formatted_lines:
                    logger_obj.info(line)
                return

            for line in formatted_lines:
                print(line)
        except Exception:
            for line in formatted_lines:
                print(line)

    def reset(self):
        self._result = TraceResult(meta=self._meta)  # single shared instance
        self._counter = 0
        self._depth = 0

    def _format_message(self, msg: str, tag: str) -> str:
        """
        If msg starts with a newline (optionally after leading spaces/tabs),
        return '\n {tag} {msg_lstripped}'.
        Otherwise return '{tag} {msg}'.

        Examples:
        "\nHi"      -> "\n [added] Hi"
        "Hi"        -> "[added] Hi"
        "  \nHi"    -> "\n [added] Hi"
        """
        # Detect a leading newline possibly after leading spaces/tabs
        i = 0
        while i < len(msg) and msg[i] in (" ", "\t"):
            i += 1
        has_leading_newline = i < len(msg) and msg[i] == "\n"

        if has_leading_newline:
            # Keep exactly one leading newline for the added prefix, then strip whitespace/newlines before content
            content = msg.lstrip(" \t\n")
            return f"\n\{tag} {content}" if content else f"\n\{tag}"
        else:
            return f"\{tag} {msg}"

