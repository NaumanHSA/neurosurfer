"""Open a trace span around a run that has no event stream of its own.

Agent runs get their trace for free: :meth:`neurosurfer.agents.base.BaseAgent._tap`
wraps the event stream in a :class:`~.exporters.stream.TraceStreamObserver` that
opens a trace, publishes an ambient :class:`~.context.TraceContext`, and closes it.

Two things in the **graph executor** have no such stream and need a manual span:

  * the **workflow** as a whole — otherwise each node opens its own disconnected
    top-level trace; and
  * each **node** — so non-agent nodes (function / tool) are visible at all, and an
    agent node nests under its *node* row rather than directly under the workflow.

:func:`traced_run` covers both. It opens a span (``on_run_start`` → ``on_run_finish``),
publishes the ambient ``TraceContext`` so anything starting inside nests under it,
and — when it itself runs inside another traced run — nests under *that* run's span
(node under workflow, agent under node) instead of opening a new top-level trace.

It is **fail-soft** (an exporter can never break the run) and **zero-overhead** when
no exporter is configured (it yields ``None`` immediately).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from neurosurfer.observability.context import (
    TraceContext,
    current_trace_context,
    pop_trace_context,
    push_trace_context,
)
from neurosurfer.observability.exporters import get_active_exporters

logger = logging.getLogger("neurosurfer.observability")


class RunSpan:
    """Handle yielded by :func:`traced_run` for setting the span's outcome.

    A node runner catches its own errors and *returns* them on the result rather
    than raising, so the ``with`` block can't detect failure by exception. Call
    :meth:`error` (or set :attr:`output`) to mark the span before it closes.
    """

    def __init__(self, ctx: TraceContext, fan: Any) -> None:
        self.ctx = ctx
        self.status = "completed"
        self.output: Any = None
        self._fan = fan

    def error(self, message: str) -> None:
        self.status = "error"
        self._fan("on_error", message=message)


@contextmanager
def traced_run(
    name: str,
    *,
    metadata: dict[str, Any] | None = None,
    session_id: str | None = None,
    input: Any = None,
    flush: bool = True,
) -> Iterator[RunSpan | None]:
    """Open a trace span for a non-agent run (a workflow, or one graph node).

    Yields a :class:`RunSpan` handle (or ``None`` when no exporter is active).
    Anything that starts inside the ``with`` block reads the published context and
    nests under this span.

    ``flush`` controls whether exporters are flushed when the span closes — leave
    it on for the outermost (workflow) span; pass ``flush=False`` for the many
    per-node spans, whose data the workflow-level flush already pushes.
    """
    exporters = get_active_exporters()
    if not exporters:
        yield None
        return

    parent = current_trace_context()
    if parent is not None:
        ctx = parent.child(metadata={**parent.metadata, **(metadata or {})})
    else:
        ctx = TraceContext(session_id=session_id, metadata=metadata or {})

    def _fan(hook: str, **kw: Any) -> None:
        for exp in exporters:
            try:
                getattr(exp, hook)(ctx, **kw)
            except Exception:  # noqa: BLE001 — an exporter must never break the run
                logger.debug("trace exporter %s.%s failed", exp.name, hook, exc_info=True)

    span = RunSpan(ctx, _fan)
    token = push_trace_context(ctx)
    _fan("on_run_start", name=name, input=input)
    try:
        yield span
    except BaseException as exc:  # noqa: BLE001 — record then re-raise
        span.status = "error"
        _fan("on_error", message=repr(exc))
        raise
    finally:
        try:
            pop_trace_context(token)
        except Exception:  # noqa: BLE001 — token from a different context, ignore
            pass
        _fan("on_run_finish", status=span.status, output=span.output)
        if flush:
            for exp in exporters:
                try:
                    exp.flush()
                except Exception:  # noqa: BLE001
                    logger.debug(
                        "trace exporter %s.flush failed", exp.name, exc_info=True
                    )
