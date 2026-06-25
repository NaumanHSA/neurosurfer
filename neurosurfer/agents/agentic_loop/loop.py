"""The AgenticLoop implementation — multi-step native-tool-use agent loop.

``run(user_input)`` is an async generator of :mod:`events`. Each iteration:
build the request → stream the model (emitting text/thinking deltas) → if the
turn produced native ``tool_use`` blocks, gate + execute them and append results →
repeat; else the turn's text is the final answer. Plan approval flips the
permission mode out of plan; ``finish`` ends the run. Provider, tools, and
guardrails are injected via :class:`~neurosurfer.agents.base.BaseAgent`.

This is the production agent for providers with a native tool-calling API. For
providers without one, use :class:`~neurosurfer.agents.react.ReactAgent`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from neurosurfer.agents.base import BaseAgent
from neurosurfer.agents.conversation import events
from neurosurfer.agents.runtime.loop import execute_tool_uses
from neurosurfer.llm import types as lt


class AgenticLoop(BaseAgent):
    async def _run(self, user_input: str) -> AsyncIterator[events.Event]:
        self.history.add_user_text(user_input)
        async for ev in self._loop():
            yield ev

    # ── the loop ────────────────────────────────────────────────────────────────
    async def _loop(self) -> AsyncIterator[events.Event]:
        while self.turns < self.guardrails.max_turns:
            self.turns += 1

            if self.context_manager is not None:
                async for compact_ev in self.context_manager.maybe_compact(
                    self.provider, self.history, self._effective_system(), self.tools.schemas()
                ):
                    yield compact_ev

            response: lt.CanonicalResponse | None = None
            async for stream_ev in self._stream_model():
                if isinstance(stream_ev, lt.TextDelta):
                    yield events.TextDelta(stream_ev.text)
                elif isinstance(stream_ev, lt.ThinkingDelta):
                    yield events.ThinkingDelta(stream_ev.text)
                elif isinstance(stream_ev, lt.Done):
                    response = stream_ev.response
            if response is None:  # pragma: no cover - stream always ends with Done
                yield events.AgentError("Model stream ended without a response.")
                return

            self.usage = self.usage.add(response.usage)
            self.history.add_assistant_response(response)
            yield events.TurnCompleted(response.usage, response.stop_reason)

            tool_uses = response.tool_uses()
            if not tool_uses:
                # Text was already emitted turn-by-turn via TextDelta events;
                # don't pass it as report or render.py would print it again.
                yield events.RunFinished("completed")
                return

            for tu in tool_uses:
                yield events.ToolStarted(
                    tu.id, tu.name, tu.input, title=self.tools.progress_message(tu.name, tu.input)
                )

            outcomes = await execute_tool_uses(
                tool_uses,
                tools=self.tools,
                ctx=self._ctx,
                permissions=self.permissions,
                mode=self.mode,
            )

            finish_signal: dict | None = None
            for oc in outcomes:
                yield events.ToolFinished(oc.id, oc.name, oc.result)
                ctrl = oc.result.control
                if ctrl.get("plan_approved") and self.mode == "plan":
                    self.mode = "default"
                    yield events.ModeChanged("default", "plan approved")
                if ctrl.get("finished"):
                    finish_signal = ctrl

            tool_images = [img for oc in outcomes for img in oc.result.images]
            self.history.add_tool_results(
                [(oc.id, oc.result.content, oc.result.is_error) for oc in outcomes],
                images=tool_images,
            )

            if finish_signal is not None:
                yield events.RunFinished(
                    finish_signal.get("status", "success"),
                    finish_signal.get("report", ""),
                )
                return

        yield events.RunFinished(
            "max_turns", f"Reached the turn limit ({self.guardrails.max_turns})."
        )
