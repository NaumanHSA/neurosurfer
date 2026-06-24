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
_FINAL_PREFIX_RE = re.compile(r"Final\s*Answer\s*:\s*", re.IGNORECASE)
_ACTION_RE = re.compile(r"Action\s*:\s*([^\n]+)", re.IGNORECASE)
_ACTION_INPUT_RE = re.compile(r"Action\s*Input\s*:\s*(.*)", re.IGNORECASE | re.DOTALL)
_OBSERVATION_SPLIT_RE = re.compile(r"\n\s*Observation\s*:", re.IGNORECASE)
_STOP = "\nObservation:"
# Chars held back while streaming reasoning, so a "Final Answer:" marker forming at
# the chunk boundary is never mis-emitted as thinking before we can detect it.
_MARKER_HOLDBACK = 32

# Prose that looks like a capability refusal / deferral rather than a real answer.
# When an instruction-tuned model ignores the tool instructions it tends to say
# "I don't have access…" or "as an AI… you can run the following". We nudge it back
# into the ReAct format once; genuine direct answers (no tools needed) are accepted.
_REFUSAL_RE = re.compile(
    r"i (?:do not|don't|cannot|can't|am unable to|am not able to)\b[^.]*\b"
    r"(?:access|see|view|list|read|open|browse|retrieve|execute|run)\b"
    r"|i (?:do not|don't) have (?:access|the ability|direct access)"
    r"|as an ai\b"
    r"|i (?:am|'m) (?:just |only )?(?:a|an) (?:[a-z]+[- ]){0,3}(?:model|assistant|ai|bot|llm)\b"
    r"|you can (?:use|run|try|find) (?:the following|this|these)",
    re.IGNORECASE | re.DOTALL,
)


def _looks_like_refusal(text: str) -> bool:
    return bool(_REFUSAL_RE.search(text or ""))

# ── native tool-call fallback patterns ───────────────────────────────────────
# Local models that were fine-tuned for tool use sometimes ignore the ReAct text
# format and emit tool calls in their own delimited format. We detect and convert
# those here so ReactAgent works regardless of which convention the model uses.

# <|tool_call>call:NAME{ARGS}<tool_call|>  (and minor casing/bracket variants)
_NATIVE_CALL_RE = re.compile(
    r"<\|tool_call>call:([\w.:-]+)\s*(\{.*?\})\s*<tool_call\|>",
    re.DOTALL | re.IGNORECASE,
)
# <tool_call>{"name":"NAME","arguments":{...}}</tool_call>  (Hermes / NousHermes)
_HERMES_CALL_RE = re.compile(
    r"<\|?tool_call\|?>\s*(\{.*?\})\s*</?\|?tool_call\|?>",
    re.DOTALL | re.IGNORECASE,
)
# [TOOL_CALLS] [{"name":"NAME","arguments":{...}}]  (Mistral)
_MISTRAL_CALL_RE = re.compile(
    r"\[TOOL_CALLS\]\s*\[\s*(\{.*?\})\s*\]",
    re.DOTALL | re.IGNORECASE,
)


class ReactAgent(BaseAgent):
    """A bounded ReAct loop over text-parsed actions."""

    async def run(self, user_input: str) -> AsyncIterator[events.Event]:
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


# ── prompt + parsing helpers ──────────────────────────────────────────────────

def _build_react_system(base_system: str, tools) -> str:
    catalog = _tool_catalog(tools)
    names = ", ".join(tools.names()) or "(none)"
    base = (base_system or "").rstrip()
    return f"""{base}

You are an autonomous agent running inside a live environment. The tools listed
below are REAL and connected: calling one actually executes and returns a real
result from this machine. You are NOT a disconnected chatbot. Therefore:
- Never say you "cannot access" files, directories, systems, or data. If a tool
  can get it, call the tool and get it.
- Never tell the user to run a command themselves (e.g. `ls`, `dir`, a Python
  snippet). You run it, by emitting an Action.
- When a task needs information you don't already have, your FIRST move is a tool
  call — not an apology and not a guess.

## Available tools
{catalog}

## Response format
Think step by step. On every turn you output EITHER exactly one action OR a final
answer — never both — using these exact labels:

Thought: your reasoning about what to do next
Action: one tool name, exactly one of [{names}]
Action Input: a single-line JSON object of arguments for that tool

After each action the environment will reply with:

Observation: the result of the tool call

Keep repeating the Thought → Action → Action Input → Observation cycle until you
have everything you need. Only then, finish with:

Thought: I now have enough information to answer.
Final Answer: your complete answer to the user

## Worked example
Question: How many Python files are in the src directory?
Thought: I need to list the Python files under src, then count them. I'll use list_dir.
Action: list_dir
Action Input: {{"path": "src", "pattern": "**/*.py"}}
Observation: src/app.py
src/utils.py
src/models/user.py
Thought: There are three Python files. I now have enough information to answer.
Final Answer: There are 3 Python files in src: app.py, utils.py, and models/user.py.

## Hard rules
- ALWAYS use a tool when the task requires information or actions you cannot
  fulfil from memory alone.
- Emit at most ONE Action (with its Action Input) per turn, OR a Final Answer.
- Action must be exactly one of: [{names}]. Never invent or rename a tool.
- Action Input MUST be a valid JSON object on a single line (use {{}} when the
  tool takes no arguments).
- NEVER write an "Observation:" line yourself — the environment provides it.
- NEVER refuse a task for "lack of access"; you have the tools above — use them.
- Give a Final Answer only after the observations give you what you need."""


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


def _repair_kwargs(raw: str) -> str:
    """Convert unquoted ``key=value`` kwargs to ``"key": value`` so json.loads can parse."""
    return re.sub(r'(?<!["\'\w])(\w+)\s*=\s*', r'"\1": ', raw)


def _try_parse_native_tool(text: str) -> tuple[str | None, str | None]:
    """Try to extract ``(tool_name, action_input_raw)`` from native tool-call delimiters.

    Handles three common local-model formats so ReactAgent degrades gracefully when a
    model ignores the ReAct text instructions and emits its own tool-call syntax.
    Returns ``(None, None)`` when none of the patterns match.
    """
    # Format 1: <|tool_call>call:NAME{ARGS}<tool_call|>
    m = _NATIVE_CALL_RE.search(text)
    if m:
        name = m.group(1).strip()
        raw = m.group(2).strip()
        # Models sometimes use key=value instead of "key": value.
        # Check for unquoted key= patterns rather than just presence of "=".
        if re.search(r'(?<!["\'\w])\w+\s*=', raw):
            raw = _repair_kwargs(raw)
        return name, raw

    # Format 2: <tool_call>{"name": "NAME", "arguments": {...}}</tool_call>  (Hermes)
    m = _HERMES_CALL_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                name = obj.get("name") or obj.get("function")
                args = obj.get("arguments") or obj.get("parameters") or {}
                if name:
                    return str(name), json.dumps(args)
        except (json.JSONDecodeError, ValueError):
            pass

    # Format 3: [TOOL_CALLS] [{"name": "NAME", "arguments": {...}}]  (Mistral)
    m = _MISTRAL_CALL_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                name = obj.get("name") or obj.get("function")
                args = obj.get("arguments") or obj.get("parameters") or {}
                if name:
                    return str(name), json.dumps(args)
        except (json.JSONDecodeError, ValueError):
            pass

    return None, None


def _parse_react_output(text: str) -> tuple[str | None, str | None, str | None]:
    """Return (final_answer, action_name, action_input_raw).

    Final answer wins if present. Otherwise an Action (+ its raw input) is returned.
    Falls back to native tool-call format detection when the model doesn't follow the
    ReAct text convention. All three are ``None`` when the text contains neither.
    """
    final_m = _FINAL_RE.search(text)
    if final_m:
        return final_m.group(1).strip(), None, None

    action_m = _ACTION_RE.search(text)
    if not action_m:
        # Model didn't follow ReAct text format — try native tool-call patterns.
        native_name, native_raw = _try_parse_native_tool(text)
        if native_name:
            return None, native_name, native_raw or "{}"
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
