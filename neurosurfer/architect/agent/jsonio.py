"""Tolerant JSON extraction from model text.

Small models do not honour "STRICT JSON only". They open with "Thinking…", quote a
fragment of the schema back at you, wrap the answer in a code fence, and then emit
the object you asked for. Everything in the Architect that asks a model for
structured output has to survive that, so the parsing lives in one place rather
than being reinvented per call site.

The rule that makes it work: scan **every** balanced ``{...}`` in the text and take
the **last** one that parses (optionally requiring a key). The real answer comes
after the reasoning, and any example objects quoted along the way come before it.
"""

from __future__ import annotations

import json
from typing import Any

__all__ = ["balanced_objects", "parse_json"]


def balanced_objects(s: str) -> list[str]:
    """Every top-level brace-balanced ``{...}`` substring (string-literal aware)."""
    objs: list[str] = []
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                objs.append(s[start:i + 1])
                start = -1
    return objs


def parse_json(text: str, *, want: str | None = None) -> Any:
    """The last balanced object in *text* that parses, and has *want* if given."""
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    best: Any = None
    for chunk in balanced_objects(text):
        try:
            parsed = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        if want is None or want in parsed:
            best = parsed  # keep the last match
    return best
