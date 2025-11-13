# neurosurfer/agents/graph/tracing.py
from typing import Any, Dict, Optional, Union
import time
import logging

default_logger = logging.getLogger("neurosurfer.graph")

class Tracer:
    def span(self, name: str, attrs: Optional[Dict[str, Any]] = None):
        raise NotImplementedError

class ConsoleTracer(Tracer):
    """Writes spans to stdout (print)."""
    def span(self, name: str, attrs: Optional[Dict[str, Any]] = None):
        return _Span(name, attrs, sink=print)

try:
    from rich.console import Console
    class RichTracer(Tracer):
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
                self.console.print(f"[dim]{msg}[/dim]", end="\n")
            return _Span(name, attrs, sink=sink)

except Exception:
    default_logger.warning("Rich not installed; falling back to ConsoleTracer")
    RichTracer = ConsoleTracer  # fall back

class LoggerTracer(Tracer):
    """Writes spans to Python logging (INFO)."""
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or default_logger
    def span(self, name: str, attrs: Optional[Dict[str, Any]] = None):
        return _Span(name, attrs, sink=lambda *args, **kw: self.logger.info(args[0]))

class MemoryTracer(Tracer):
    def __init__(self):
        self.events = []  # list of dicts
    def span(self, name: str, attrs: Optional[Dict[str, Any]] = None):
        tracer = self
        class _MemSpan(_Span):
            def __enter__(self):
                self.t0 = time.time()
                tracer.events.append({"name": self.name, "attrs": self.attrs, "ts": self.t0, "phase": "start"})
                return self
            def __exit__(self, exc_type, exc, tb):
                dt = time.time() - self.t0
                tracer.events.append({"name": self.name, "attrs": self.attrs, "ts": time.time(), "phase": "end", "ms": int(dt*1000), "error": bool(exc)})
        return _MemSpan(name, attrs, sink=lambda *_: None)

class _Span:
    def __init__(self, name: str, attrs=None, sink=print):
        self.name = name
        self.attrs = attrs
        self._sink = sink

    def _format_suffix(self) -> str:
        if isinstance(self.attrs, dict):
            if not self.attrs:
                return ""
            # compact: key=value key2=value2
            kv = " ".join(f"{k}={v!r}" for k, v in self.attrs.items())
            return " " + kv
        elif isinstance(self.attrs, str):
            # simple human message
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
        self._sink(f"[trace] ◀ {self.name}{suffix} took {dt:.3f}s; error={bool(exc)}")
        
class NullTracer(Tracer):
    def span(self, name: str, attrs=None):
        return _Span(name, attrs, sink=lambda *_: None)

