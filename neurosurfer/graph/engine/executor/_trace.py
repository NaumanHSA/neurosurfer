"""Trace helpers shared by every node runner.

These live apart from `core` for one structural reason: the runners import them,
`core` imports the runners, and a helper left in `core` would make that a cycle.
Nothing here knows what a node is.
"""

from __future__ import annotations

from typing import Any

_TRACE_TEXT_LIMIT = 4000


def _trace_text(value: Any) -> Any:
    """JSON-safe, length-capped copy of a value for the trace.

    A trace is a debugging artifact, not a second store of every payload: a long
    map output would otherwise be duplicated in full on disk for every run.
    """
    if value is None or isinstance(value, (bool, int, float)):
        return value
    text = value if isinstance(value, str) else repr(value)
    return text if len(text) <= _TRACE_TEXT_LIMIT else text[:_TRACE_TEXT_LIMIT] + "…"


def _trace_step(tracer, **kwargs):
    """Open a trace step, or a no-op context when there's no tracer.

    Keeps call sites free of `if self.tracer is not None` noise; the Tracer's own
    disabled path already returns a no-op, this covers `tracer=None` too.
    """
    if tracer is None:
        from contextlib import nullcontext

        return nullcontext(None)
    return tracer(**kwargs)
