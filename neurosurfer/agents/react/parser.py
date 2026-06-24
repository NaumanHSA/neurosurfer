"""Tolerant parsing of a model's ReAct text output.

Pulls the ``Final Answer`` / ``Action`` / ``Action Input`` out of free-form model text,
recovering from the usual mess (code fences, trailing prose, a fabricated Observation),
and — when a model ignores the ReAct format entirely — falling back to the native
tool-call delimiters several local models emit. Pure functions, no agent state.
"""

from __future__ import annotations

import json
import re

_FINAL_RE = re.compile(r"Final\s*Answer\s*:\s*(.*)", re.IGNORECASE | re.DOTALL)
_FINAL_PREFIX_RE = re.compile(r"Final\s*Answer\s*:\s*", re.IGNORECASE)
_ACTION_RE = re.compile(r"Action\s*:\s*([^\n]+)", re.IGNORECASE)
_ACTION_INPUT_RE = re.compile(r"Action\s*Input\s*:\s*(.*)", re.IGNORECASE | re.DOTALL)
_OBSERVATION_SPLIT_RE = re.compile(r"\n\s*Observation\s*:", re.IGNORECASE)
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
