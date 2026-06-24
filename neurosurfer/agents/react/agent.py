"""ReactAgent — a production-grade text-parsing ReAct loop.

For providers **without** a native tool-calling API (small / local models). The
model is prompted to emit ``Thought / Action / Action Input`` and we parse that text,
run the tool through the *same* permission-gated execution path the native loop uses
(:func:`~neurosurfer.agents.runtime.loop.execute_tool_uses` via a synthesized
``ToolUseBlock``), feed back an ``Observation``, and repeat until ``Final Answer:``.

Hardened against the failure modes of the retired vendored ReAct:
- **No sentinel leakage** — only the text *after* ``Final Answer:`` is emitted.
- **Tolerant parsing** — code fences stripped, first ``{...}`` recovered; unparseable
  input feeds a corrective Observation back to the model instead of crashing.
- **Deterministic termination** — ends on ``Final Answer`` or ``max_turns``; on the
  turn limit it returns the best partial text, never an empty string. ``Observation:``
  is a stop sequence, so the model can't hallucinate tool output.

For providers *with* native tool-use prefer :class:`~neurosurfer.agents.agentic_loop.AgenticLoop`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import AsyncIterator

from neurosurfer.agents.base import BaseAgent
from neurosurfer.agents.conversation import events
from neurosurfer.agents.runtime.loop import execute_tool_uses
from neurosurfer.llm import types as lt

from .parser import (
    _FINAL_PREFIX_RE,
    _MARKER_HOLDBACK,
    _looks_like_refusal,
    _parse_action_input,
    _parse_react_output,
)
from .prompt import _build_react_system


class ReactAgent(BaseAgent):
    """A bounded ReAct loop over text-parsed actions."""

    async def _run(self, user_input: str) -> AsyncIterator[events.Event]:
        self.history.add_user_text(user_input)
        # Build the ReAct system prompt once (base prompt + tool catalog + format).
        react_system = _build_react_system(self.system_prompt, self.tools)
        # print(f"React system prompt:\n{react_system}\n{'─' * 60}")
        # Stop before the model fabricates an Observation.
        cfg = dataclasses.replace(
            self.gen_config,
            stop_sequences=[*self.gen_config.stop_sequences, "Observation:"],
        )

        last_text = ""
        corrected = False  # whether we have already pushed a corrective observation
        while self.turns < self.guardrails.max_turns:
            self.turns += 1

            if self.context_manager is not None:
                async for ev in self.context_manager.maybe_compact(
                    self.provider, self.history, react_system, []
                ):
                    yield ev

            # Stream the turn live, classifying tokens as they arrive: the model's
            # reasoning and Thought/Action scaffolding stream as ThinkingDelta (the UI
            # collapses these to a "Thinking…" indicator); once "Final Answer:" appears,
            # everything after it streams token-by-token as TextDelta — the real answer.
            # Pass react_system so the model actually receives the tool catalog + format.
            parts: list[str] = []
            response: lt.CanonicalResponse | None = None
            final_start: int | None = None  # index in the buffer where the answer begins
            emitted_think = 0               # raw chars already emitted as ThinkingDelta
            emitted_final = 0               # raw chars already consumed from the answer
            final_has_content = False       # whether a non-whitespace answer char was shown

            async for ev in self._stream_model(tool_schemas=[], system=react_system):
                if isinstance(ev, lt.Done):
                    response = ev.response
                    continue
                if isinstance(ev, lt.ThinkingDelta):
                    yield events.ThinkingDelta(ev.text)  # native reasoning channel
                    continue
                if not isinstance(ev, lt.TextDelta):
                    continue

                parts.append(ev.text)
                buf = "".join(parts)

                # Detect the answer boundary once; reasoning before it is thinking.
                if final_start is None:
                    m = _FINAL_PREFIX_RE.search(buf)
                    if m:
                        final_start = m.end()
                    else:
                        # Still reasoning — emit thinking, holding back a tail in case a
                        # "Final Answer:" marker is forming across this chunk boundary.
                        safe = len(buf) - _MARKER_HOLDBACK
                        if safe > emitted_think:
                            yield events.ThinkingDelta(buf[emitted_think:safe])
                            emitted_think = safe

                # Stream the answer tail token-by-token, dropping leading whitespace
                # until the first real character so the answer starts clean.
                if final_start is not None:
                    raw = buf[final_start + emitted_final:]
                    if raw:
                        emitted_final += len(raw)
                        shown = raw if final_has_content else raw.lstrip()
                        if shown:
                            final_has_content = True
                            yield events.TextDelta(shown)

            turn_text = "".join(parts)
            last_text = turn_text.strip()
            final_streamed = final_start is not None

            # Flush any reasoning we held back (only matters for verbatim thinking UIs).
            if final_start is None and len(turn_text) > emitted_think:
                yield events.ThinkingDelta(turn_text[emitted_think:])

            if response is not None:
                self.usage = self.usage.add(response.usage)
                self.history.add_assistant_response(response)
            else:
                yield events.AgentError("Model stream ended without a response.")
                return

            final, action_name, action_input_raw = _parse_react_output(turn_text)

            if final is not None:
                # Already streamed inline above; only emit here if detection missed it.
                if not final_streamed:
                    yield events.TextDelta(final)
                yield events.RunFinished("completed", final)
                return

            if action_name is None:
                if not corrected and _looks_like_refusal(last_text):
                    # The model ignored the tool instructions and refused conversationally
                    # ("I don't have access…"). Push a corrective observation once and retry —
                    # it usually follows the ReAct format on the second attempt. Genuine
                    # direct answers (no refusal) fall through and are accepted as final.
                    corrected = True
                    self.history.add_user_text(
                        "Observation: You DO have working tools available (listed above) and "
                        "must use them instead of refusing. Respond strictly in this format:\n"
                        "Thought: <your reasoning>\n"
                        "Action: <one of the listed tool names>\n"
                        "Action Input: <JSON object of arguments>"
                    )
                    continue
                # No action and not a refusal — the prose IS the answer (e.g. a question
                # that needs no tools, or the model's post-correction reply). The text was
                # streamed as thinking; emit it as the final answer too.
                yield events.TextDelta(last_text)
                yield events.RunFinished("completed", last_text)
                return

            # Parse the action input; on failure feed a corrective Observation.
            parsed, parse_err = _parse_action_input(action_input_raw)
            if parse_err is not None:
                obs = (
                    f"Action Input was not valid JSON ({parse_err}). "
                    "Respond with `Action Input:` followed by a single JSON object."
                )
                self.history.add_user_text(f"Observation: {obs}")
                continue

            tu = lt.ToolUseBlock(id=f"react-{self.turns}", name=action_name, input=parsed)
            yield events.ToolStarted(
                tu.id, tu.name, tu.input, title=self.tools.progress_message(tu.name, tu.input)
            )
            outcomes = await execute_tool_uses(
                [tu],
                tools=self.tools,
                ctx=self._ctx,
                permissions=self.permissions,
                mode=self.mode,
            )
            outcome = outcomes[0]
            yield events.ToolFinished(outcome.id, outcome.name, outcome.result)

            # A finishing tool (e.g. `finish`) ends the run immediately.
            ctrl = outcome.result.control
            if ctrl.get("finished"):
                report = ctrl.get("report", "") or outcome.result.content
                yield events.RunFinished(ctrl.get("status", "success"), report)
                return

            self.history.add_user_text(f"Observation: {outcome.result.content}")

        # Turn limit: return the best partial text rather than nothing.
        yield events.RunFinished(
            "max_turns", last_text or f"Reached the turn limit ({self.guardrails.max_turns})."
        )
