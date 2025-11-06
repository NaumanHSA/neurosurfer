# neurosurfer/agents/graph/tracing.py
from typing import Any, Dict, Optional
import time

class Tracer:
    def span(self, name: str, attrs: Optional[Dict[str, Any]] = None):
        raise NotImplementedError

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
