# neurosurfer/tracing/__init__.py
from .span import (
    SpanTracer,
    ConsoleTracer,
    LoggerTracer,
    MemorySpanTracer,
    NullSpanTracer,
    RichTracer,     # RichTracer becomes ConsoleTracer if `rich` package is missing
)
from .models import TraceStep, TraceResult
from .tracer import Tracer, TracerConfig, TraceStepContext

__all__ = [
    "SpanTracer",
    "ConsoleTracer",
    "LoggerTracer",
    "MemorySpanTracer",
    "NullSpanTracer",
    "RichTracer",
    "TraceStep",
    "TraceResult",
    "Tracer",
    "TracerConfig",
    "TraceStepContext",
]
