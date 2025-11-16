# neurosurfer/tracing/span.py
from __future__ import annotations

from typing import Any, Dict, Optional, Union, Callable, List
import time
import logging

default_logger = logging.getLogger("neurosurfer.tracing")


class SpanTracer:
    """
    Low-level span tracer interface.

    This is deliberately minimal: it just exposes a `span(name, attrs)` method
    that returns a context manager. Span tracers are used by higher-level
    tracing utilities (e.g., Tracer) to emit human-readable logs.
    """

    def span(self, name: str, attrs: Optional[Dict[str, Any]] = None):
        raise NotImplementedError


class ConsoleTracer(SpanTracer):
    """
    Span tracer that writes to stdout using `print`.

    Example line:
        [trace] ▶ step.llm step_id=1 label='agent.llm.ask'
    """

    def span(self, name: str, attrs: Optional[Dict[str, Any]] = None):
        return _Span(name, attrs, sink=print)


try:
    from rich.console import Console
    class RichTracer(SpanTracer):
        """
        Span tracer that writes using `rich`.

        - `attrs` can be a dict (printed as key=value pairs)
        - or a str (printed as a message)
        """

        def __init__(self, console: Optional[Console] = None):
            self.console = console or Console(force_jupyter=False, force_terminal=True, width=300)

        def span(
            self,
            name: str,
            attrs: Optional[Union[Dict[str, Any], str]] = None,
        ):
            def sink(msg: str):
                # Avoid extra blank lines
                self.console.print(f"[dim]{msg}[/dim]", end="\n")

            return _Span(name, attrs, sink=sink)

except Exception:  # pragma: no cover - optional dependency
    default_logger.warning("Rich not installed; RichTracer will not be available")
    RichTracer = ConsoleTracer  # type: ignore[misc]


class LoggerTracer(SpanTracer):
    """
    Span tracer that writes to a Python `logging.Logger`.

    Each span line is logged at INFO level.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or default_logger

    def span(self, name: str, attrs: Optional[Dict[str, Any]] = None):
        return _Span(
            name,
            attrs,
            sink=lambda *args, **kw: self.logger.info(args[0]),
        )


class MemorySpanTracer(SpanTracer):
    """
    Span tracer that captures span events in-memory.

    Mostly useful for tests or debugging.
    """

    def __init__(self):
        self.events: List[Dict[str, Any]] = []

    def span(self, name: str, attrs: Optional[Dict[str, Any]] = None):
        tracer = self

        class _MemSpan(_Span):
            def __enter__(self):
                self.t0 = time.time()
                tracer.events.append(
                    {
                        "name": self.name,
                        "attrs": self.attrs,
                        "ts": self.t0,
                        "phase": "start",
                    }
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


class NullSpanTracer(SpanTracer):
    """
    No-op span tracer.

    All spans are ignored; useful when you want structured tracing
    but no human-readable logs.
    """

    def span(self, name: str, attrs: Optional[Dict[str, Any]] = None):
        return _Span(name, attrs, sink=lambda *_: None)


class _Span:
    """
    Simple span context manager with a pluggable sink.

    Semantics:
      - On __enter__: logs "▶ ..."
      - On __exit__: logs "◀ ... took X.XXXs; error=<bool>"

    `attrs` may be:
      - dict -> formatted as "key=value key2=value2"
      - str  -> appended as a free-form message
      - None -> nothing extra

    Special dict keys:
      - step_id: int, printed as [trace:<id>]
      - _level: int, nesting depth (0 = root, 1 = child, ...)
    """

    def __init__(
        self,
        name: str,
        attrs: Optional[Union[Dict[str, Any], str]] = None,
        sink: Callable[[str], None] = print,
    ):
        self.name = name
        self.attrs = attrs
        self.indent_chars = 4
        self._sink = sink

        self.step_id: Optional[int] = None
        self.level: int = 0

        # Only dict attrs get special handling
        if isinstance(self.attrs, dict):
            attrs_dict = self.attrs

            step_id = attrs_dict.pop("step_id", None)
            if step_id is not None:
                try:
                    self.step_id = int(step_id)
                except Exception:
                    self.step_id = None

            lvl = attrs_dict.pop("_level", None)
            if lvl is not None:
                try:
                    self.level = int(lvl)
                except Exception:
                    self.level = 0

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
        indent = self.level * self.indent_chars
        name_tag = f"\\[{self.step_id}]\\[{self.name}]" if self.step_id is not None else f"\\[{self.name}]"
        self._sink(f"{' ' * indent} ▶ {name_tag}{suffix}")
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.time() - self.t0
        suffix = self._format_suffix()
        indent = self.level * self.indent_chars
        name_tag = f"\\[{self.step_id}]\\[{self.name}]" if self.step_id is not None else f"\\[{self.name}]"
        self._sink(
            f"{' ' * indent} ◀ {name_tag}{suffix} took {dt:.3f}s; error={bool(exc_type)}"
        )