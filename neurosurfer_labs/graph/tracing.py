# neurosurfer/agents/graph/tracing.py
from typing import Any, Dict, Optional
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
        def __init__(self):
            self.console = Console()
        def span(self, name: str, attrs: Optional[Dict[str, Any]] = None):
            def sink(msg):
                self.console.print(f"[dim]{msg}[/dim]")
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
        self.attrs = attrs or {}
        self._sink = sink
    def __enter__(self):
        self.t0 = time.time()
        self._sink(f"[trace] ▶ {self.name} {self.attrs}")
        return self
    def __exit__(self, exc_type, exc, tb):
        dt = time.time() - self.t0
        self._sink(f"[trace] ◀ {self.name} took {dt:.3f}s; error={bool(exc)}")

class NullTracer(Tracer):
    def span(self, name: str, attrs=None):
        return _Span(name, attrs, sink=lambda *_: None)

