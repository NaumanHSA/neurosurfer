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


def _summarize_message(m) -> dict:
    """Compact, JSON-friendly view of a single canonical message. Image payloads
    are elided so traces stay small; all other blocks (thinking / text / tool_use
    / tool_result) are preserved as-is."""
    blocks: list[dict] = []
    for b in m.content:
        d = b.model_dump()
        if d.get("type") == "image":
            d["data"] = "<base64 image elided>" if d.get("data") else None
        blocks.append(d)
    return {"role": m.role, "content": blocks}


def _summarize_messages(messages: list | None) -> list[dict] | None:
    """The messages sent to the model this turn — the generation's *input*."""
    if not messages:
        return None
    return [_summarize_message(m) for m in messages]


def _run_output(answer: str, last_message) -> str | None:
    """Run-level output: the streamed answer text, or — when the final turn
    produced no user-facing text (its answer lives in the thinking channel, as
    some reasoning models do) — that turn's thinking, so the trace isn't blank."""
    if answer:
        return answer
    if last_message is not None:
        thinking = "".join(getattr(b, "thinking", "") or "" for b in last_message.content)
        if thinking:
            return thinking
    return None


class TraceStreamObserver:
    def __init__(
        self,
        ctx: TraceContext,
        exporters: list[TraceExporter],
        *,
        model: str | None,
        name: str,
        system: str | None = None,
    ) -> None:
        self._ctx = ctx
        self._exporters = exporters
        self._model = model
        self._name = name
        # The agent's system prompt — prepended to each generation's traced input so
        # the trace shows the FULL prompt the model saw (a node's interpolated
        # purpose/goal lives here, not in the message history).
        self._system = system
        self._answer: list[str] = []       # whole-run answer text (→ run output)
        self._last_output = None            # last turn's assistant message (fallback output)
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

    def _turn_input(self, messages: list | None) -> list[dict] | None:
        """The generation's input: the turn's messages, with the system prompt
        prepended so the trace shows the full prompt the model actually saw."""
        summarized = _summarize_messages(messages) or []
        if self._system:
            return [{"role": "system", "content": self._system}, *summarized]
        return summarized or None

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
            self._last_output = ev.output
            self._fan(
                "on_turn",
                usage=ev.usage,
                model=self._model,
                stop_reason=ev.stop_reason,
                input=self._turn_input(ev.input),
                # The model's own output this turn — thinking + text + tool_use —
                # so each generation shows what *it* produced, not just the answer text.
                output=_summarize_message(ev.output) if ev.output is not None else None,
            )
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
            output=_run_output("".join(self._answer), self._last_output),
        )
        for exp in self._exporters:
            try:
                exp.flush()
            except Exception:  # noqa: BLE001
                logger.debug("trace exporter %s.flush failed", exp.name, exc_info=True)
