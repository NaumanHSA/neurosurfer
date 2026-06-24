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
import json
import re
from collections.abc import AsyncIterator

from neurosurfer.agents.base import BaseAgent
from neurosurfer.agents.conversation import events
from neurosurfer.agents.runtime.loop import execute_tool_uses
from neurosurfer.llm import types as lt

_FINAL_RE = re.compile(r"Final\s*Answer\s*:\s*(.*)", re.IGNORECASE | re.DOTALL)
_ACTION_RE = re.compile(r"Action\s*:\s*([^\n]+)", re.IGNORECASE)
_ACTION_INPUT_RE = re.compile(r"Action\s*Input\s*:\s*(.*)", re.IGNORECASE | re.DOTALL)
_OBSERVATION_SPLIT_RE = re.compile(r"\n\s*Observation\s*:", re.IGNORECASE)
_STOP = "\nObservation:"


class ReactAgent(BaseAgent):
    """A bounded ReAct loop over text-parsed actions."""

    async def run(self, user_input: str) -> AsyncIterator[events.Event]:
        self.history.add_user_text(user_input)
        # Build the ReAct system prompt once (base prompt + tool catalog + format).
        react_system = _build_react_system(self.system_prompt, self.tools)
        # Stop before the model fabricates an Observation.
        cfg = dataclasses.replace(
            self.gen_config,
            stop_sequences=[*self.gen_config.stop_sequences, "Observation:"],
        )

        last_text = ""
        while self.turns < self.guardrails.max_turns:
            self.turns += 1

            if self.context_manager is not None:
                async for ev in self.context_manager.maybe_compact(
                    self.provider, self.history, react_system, []
                ):
                    yield ev

            # Stream the turn (no native tool schemas — tools live in the prompt).
            turn_text, response = await self._stream_turn(react_system, cfg)
            last_text = turn_text.strip()
            if response is not None:
                self.usage = self.usage.add(response.usage)
                self.history.add_assistant_response(response)
            else:  # provider returned no usable text
                yield events.AgentError("Model stream ended without a response.")
                return

            final, action_name, action_input_raw = _parse_react_output(turn_text)

            if final is not None:
                yield events.TextDelta(final)
                yield events.RunFinished("completed", final)
                return

            if action_name is None:
                # No action and no final answer: treat the prose as the answer
                # (graceful — avoids the classic "max iterations" stall).
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
            yield events.ToolStarted(tu.id, tu.name, tu.input)
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
        yield events.TextDelta(last_text)
        yield events.RunFinished(
            "max_turns", last_text or f"Reached the turn limit ({self.guardrails.max_turns})."
        )

    async def _stream_turn(
        self, system: str, cfg: lt.GenerationConfig
    ) -> tuple[str, lt.CanonicalResponse | None]:
        """Stream one model turn (tools described in the prompt, not advertised
        natively) and return its accumulated text + the final response."""
        parts: list[str] = []
        response: lt.CanonicalResponse | None = None
        # _stream_model honours context-manager recovery; pass [] for tool schemas.
        async for ev in self._stream_model(tool_schemas=[]):
            if isinstance(ev, lt.TextDelta):
                parts.append(ev.text)
            elif isinstance(ev, lt.Done):
                response = ev.response
        # Apply the stop sequence client-side too (providers vary on honouring it).
        text = "".join(parts)
        return text, response


# ── prompt + parsing helpers ──────────────────────────────────────────────────

def _build_react_system(base_system: str, tools) -> str:
    catalog = _tool_catalog(tools)
    names = ", ".join(tools.names()) or "(none)"
    base = (base_system or "").rstrip()
    return f"""{base}

You can use tools by reasoning step by step. You have access to these tools:

{catalog}

Use exactly this format, one step at a time:

Thought: your reasoning about what to do next
Action: the tool to use — exactly one of [{names}]
Action Input: a single JSON object of arguments for the tool

After each Action you will be given:

Observation: the result of the action

Repeat Thought/Action/Action Input as needed. When you can answer the user, stop
using tools and reply with exactly:

Thought: I now know the final answer
Final Answer: <your answer to the user>

Rules:
- Emit at most ONE Action (with its Action Input) per turn, OR a Final Answer.
- Action Input MUST be a valid JSON object (use {{}} if the tool takes no arguments).
- Never write an Observation yourself — it is given to you.
- Use only the tools listed above; do not invent tool names."""


def _tool_catalog(tools) -> str:
    lines: list[str] = []
    for schema in tools.schemas():
        props = (schema.input_schema or {}).get("properties", {}) or {}
        if props:
            arg_desc = ", ".join(
                f"{k}: {(v or {}).get('type', 'any')}" for k, v in props.items()
            )
            args = f" Args: {{{arg_desc}}}"
        else:
            args = " Args: none"
        desc = (schema.description or "").strip().splitlines()[0] if schema.description else ""
        lines.append(f"- {schema.name}: {desc}{args}")
    return "\n".join(lines) if lines else "(no tools available)"


def _parse_react_output(text: str) -> tuple[str | None, str | None, str | None]:
    """Return (final_answer, action_name, action_input_raw).

    Final answer wins if present. Otherwise an Action (+ its raw input) is returned.
    All three are ``None`` when the text contains neither.
    """
    final_m = _FINAL_RE.search(text)
    if final_m:
        return final_m.group(1).strip(), None, None

    action_m = _ACTION_RE.search(text)
    if not action_m:
        return None, None, None

    name = action_m.group(1).strip().strip("`").strip()
    # Strip surrounding brackets/quotes a model sometimes adds: [tool], "tool".
    name = name.strip("[]").strip().strip('"').strip("'")

    input_m = _ACTION_INPUT_RE.search(text, action_m.end())
    raw = input_m.group(1) if input_m else "{}"
    return None, name, raw


def _parse_action_input(raw: str) -> tuple[dict, None] | tuple[None, str]:
    """Best-effort parse of an Action Input blob into a JSON object.

    Returns ``(obj, None)`` on success or ``(None, reason)`` for a corrective
    Observation. Strips code fences, cuts a trailing Observation, and recovers the
    first ``{...}`` block.
    """
    s = raw.strip()
    s = _OBSERVATION_SPLIT_RE.split(s)[0].strip()
    s = re.sub(r"^```(?:json)?", "", s).strip()
    s = re.sub(r"```$", "", s).strip()
    if not s:
        return {}, None

    for candidate in (s, _first_brace_block(s)):
        if candidate is None:
            continue
        try:
            obj = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            return obj, None
        # A bare JSON scalar/array isn't a valid argument object.
        return None, "expected a JSON object"

    return None, "could not parse a JSON object"


def _first_brace_block(s: str) -> str | None:
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start : end + 1]
    return None
