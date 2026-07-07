"""Translate a live agent event stream into exporter lifecycle calls.

:class:`TraceStreamObserver` is the bridge between the agent's typed event stream
(``neurosurfer/agents/conversation/events.py``) and the backend-neutral
:class:`~.base.TraceExporter` contract. It is a **side-channel observer**, created
per run and fed every event alongside the ``AgentTrace`` spinner in
:meth:`neurosurfer.agents.base.BaseAgent._tap` — it observes, never consumes.

State it holds for one run:
    * the run's :class:`~neurosurfer.observability.context.TraceContext`,
    * the answer text streamed so far (→ run output),
    * the answer text of the *current* turn (→ that generation's output),
      reset at each ``TurnCompleted`` boundary,
    * the terminal ``status`` reported by ``RunFinished`` / ``AgentError``.

Every fan-out to exporters is guarded: a misbehaving exporter can never break the
agent run (same guarantee ``AgentTrace.handle`` gives).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from neurosurfer.agents.conversation import events
from neurosurfer.observability.context import pop_trace_context, push_trace_context

if TYPE_CHECKING:
    from neurosurfer.observability.context import TraceContext

    from .base import TraceExporter

logger = logging.getLogger("neurosurfer.observability")


class TraceStreamObserver:
    def __init__(
        self,
        ctx: TraceContext,
        exporters: list[TraceExporter],
        *,
        model: str | None,
        name: str,
    ) -> None:
        self._ctx = ctx
        self._exporters = exporters
        self._model = model
        self._name = name
        self._answer: list[str] = []       # whole-run answer text
        self._turn_answer: list[str] = []   # current turn's answer text
        self._status: str | None = None
        self._started = False
        self._cv_token = None               # contextvar token for nesting

    # ── fan-out helper ──────────────────────────────────────────────────────
    def _fan(self, hook: str, **kw) -> None:
        for exp in self._exporters:
            try:
                getattr(exp, hook)(self._ctx, **kw)
            except Exception:  # noqa: BLE001 — an exporter must never break the run
                logger.debug("trace exporter %s.%s failed", exp.name, hook, exc_info=True)

    # ── lifecycle ───────────────────────────────────────────────────────────
    def start(self, *, input: str | None = None) -> None:
        self._started = True
        # Publish this run as the current context so sub-agents / node agents that
        # start inside it nest under this run's span.
        self._cv_token = push_trace_context(self._ctx)
        self._fan("on_run_start", name=self._name, input=input)

    def handle(self, ev: events.Event) -> None:
        if isinstance(ev, events.TextDelta):
            self._answer.append(ev.text)
            self._turn_answer.append(ev.text)
        elif isinstance(ev, events.ToolStarted):
            self._fan("on_tool_start", call_id=ev.id, name=ev.name, args=ev.args)
        elif isinstance(ev, events.ToolFinished):
            self._fan(
                "on_tool_finish",
                call_id=ev.id,
                name=ev.name,
                result=ev.result,
                is_error=bool(ev.result.is_error),
            )
        elif isinstance(ev, events.TurnCompleted):
            self._fan(
                "on_turn",
                usage=ev.usage,
                model=self._model,
                stop_reason=ev.stop_reason,
                output="".join(self._turn_answer) or None,
            )
            self._turn_answer.clear()
        elif isinstance(ev, events.ModeChanged):
            self._fan("on_event", kind="mode_changed", mode=ev.mode, reason=ev.reason)
        elif isinstance(ev, events.Compacted):
            self._fan(
                "on_event",
                kind="compacted",
                tokens_before=ev.tokens_before,
                tokens_after=ev.tokens_after,
            )
        elif isinstance(ev, events.RunFinished):
            self._status = ev.status
        elif isinstance(ev, events.AgentError):
            self._status = "error"
            self._fan("on_error", message=ev.message)

    def on_run_exception(self, exc: BaseException) -> None:
        """The run raised out of the stream (not an ``AgentError`` event)."""
        self._status = "error"
        self._fan("on_error", message=repr(exc))

    def close(self) -> None:
        """Close the trace and flush exporters. Safe to call once per run."""
        if not self._started:
            return
        if self._cv_token is not None:
            try:
                pop_trace_context(self._cv_token)
            except Exception:  # noqa: BLE001 — token from a different context, ignore
                pass
            self._cv_token = None
        self._fan(
            "on_run_finish",
            status=self._status or "completed",
            output="".join(self._answer) or None,
        )
        for exp in self._exporters:
            try:
                exp.flush()
            except Exception:  # noqa: BLE001
                logger.debug("trace exporter %s.flush failed", exp.name, exc_info=True)
