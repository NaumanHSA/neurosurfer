import json
import re
from typing import Optional, Dict, Any, List

from .types import ToolCall
from .exceptions import ToolCallParseError
from ..common.utils import extract_and_repair_json


TRAILING_COMMA = re.compile(r",\s*([}\]])")


def _strip_code_fences(text: str) -> str:
    """
    Remove Markdown fences like ```json ... ``` or ``` ... ```.

    We do this *before* trying to parse JSON so that the parser is not
    confused by code-block markers.
    """
    if not text:
        return text

    # Remove ```lang\n at the start of blocks
    text = re.sub(r"```[a-zA-Z0-9_-]*\s*", "", text)

    # Remove remaining bare ``` markers
    text = text.replace("```", "")

    return text


def _tidy_json(s: str) -> str:
    """Clean up likely LLM mistakes in JSON."""
    s = _strip_code_fences(s).strip()

    # Remove trailing commas before } or ]
    s = TRAILING_COMMA.sub(r"\1", s)

    # Normalize booleans for final_answer
    s = re.sub(
        r'("final_answer"\s*:\s*)"true"',
        r"\1 true",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(
        r'("final_answer"\s*:\s*)"false"',
        r"\1 false",
        s,
        flags=re.IGNORECASE,
    )

    # Auto-close braces if model cut output early
    if s.count("{") > s.count("}"):
        s += "}" * (s.count("{") - s.count("}"))
    if s.count("[") > s.count("]"):
        s += "]" * (s.count("[") - s.count("]"))

    return s


def _force_object(s: str) -> Dict[str, Any]:
    """
    Try to coerce a malformed JSON string into a Python dict.

    - First attempt a normal json.loads.
    - If that fails, try trimming to the last '}' and parse again.
    """
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        last = s.rfind("}")
        if last != -1:
            try:
                return json.loads(s[: last + 1])
            except Exception:
                pass
        raise


def _normalize_memory_keys(raw: Any) -> Optional[List[str]]:
    """
    Normalize `memory_keys` to a list of strings or None.
    Accepts:
      - None
      - single string
      - list of strings / ints
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [str(k) for k in raw if isinstance(k, (str, int))]
    return None


def _normalize_final_answer(raw: Any) -> bool:
    """
    Normalize the `final_answer` field to a boolean.
    Accepts:
      - true/false (bool)
      - "true"/"false" (string, any case)
      - 1/0 or "1"/"0"
    """
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)

    if isinstance(raw, str):
        v = raw.strip().lower()
        if v in {"true", "yes", "y", "1"}:
            return True
        if v in {"false", "no", "n", "0"}:
            return False

    return False


def _find_last_action_index(text: str) -> int:
    """
    Find the index of the last occurrence of 'Action:' (case-insensitive).
    Returns -1 if not found.
    """
    lower = text.lower()
    return lower.rfind("action:")


def _extract_braced_json_from(text: str, start_idx: int) -> Optional[str]:
    """
    Given a string and an index where a '{' character appears,
    extract a balanced JSON object string starting at that '{'.

    - Tracks brace depth.
    - Respects quotes and escapes.
    - Returns the substring from the starting '{' through the matching '}'.
    - If no matching '}' is found, returns substring from '{' to end.
    """
    depth = 0
    in_string = False
    escape = False
    s = text

    for i in range(start_idx, len(s)):
        ch = s[i]

        if escape:
            escape = False
            continue

        if ch == "\\":
            escape = True
            continue

        if ch == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                # include this closing brace
                return s[start_idx : i + 1]

    # If we reach here, braces never balanced; return from start_idx to end
    if depth > 0:
        return s[start_idx:]
    return None


class ToolCallParser:
    """
    Extracts and normalizes a tool call from an LLM message.

    Behavior:
    - If an `Action:` block is present:
        - If it is explicitly `Action: None` or `Action: null` → ToolCall(tool=None, ...).
        - Else, expect JSON after `Action:` and:
            - On parse failure → raise ToolCallParseError (so agent can repair).
            - On success → return ToolCall (tool may still be None if JSON says so).
    - If no `Action:` is present:
        - Try a bare JSON fallback (for ```json { ... } ``` style).
        - If that fails → return None (no tool call).
    """

    def extract(self, text: str) -> Optional[ToolCall]:
        if not text:
            return None

        cleaned = _strip_code_fences(text)

        # 1) Find the last 'Action:' marker
        act_idx = _find_last_action_index(cleaned)
        if act_idx == -1:
            # No explicit Action:; try bare JSON as a fallback
            try:
                obj = extract_and_repair_json(cleaned, return_dict=True)
            except Exception:
                return None

            if isinstance(obj, dict) and ("tool" in obj or "inputs" in obj):
                return self._to_tool_call(obj)
            return None

        # 2) From there, skip 'Action:' and whitespace
        i = act_idx + len("Action:")
        while i < len(cleaned) and cleaned[i].isspace():
            i += 1
        if i >= len(cleaned):
            # Action: at end of string → treat as malformed
            raise ToolCallParseError("Action block found but no JSON or value after it.")

        # 3) If the next non-whitespace token is 'None' / 'null' → explicit no-tool
        tail = cleaned[i:].lstrip()
        if tail.lower().startswith("none") or tail.lower().startswith("null"):
            return ToolCall(tool=None, inputs={}, final_answer=False, memory_keys=None)

        # 4) Otherwise we expect a JSON object starting with '{'
        brace_idx = cleaned.find("{", i)
        if brace_idx == -1:
            # The model tried to emit an Action but didn't produce JSON
            raise ToolCallParseError("Action block found but no JSON object after it.")

        raw_block = _extract_braced_json_from(cleaned, brace_idx)
        if not raw_block:
            raise ToolCallParseError("Could not extract a JSON object from Action block.")

        raw_block = _tidy_json(raw_block)

        # Try strict parse, then repair; on total failure, raise parse error
        try:
            obj = _force_object(raw_block)
        except Exception:
            try:
                obj = extract_and_repair_json(raw_block, return_dict=True)
            except Exception as e:
                raise ToolCallParseError(f"Invalid JSON in Action block: {e}") from e

        if not isinstance(obj, dict):
            raise ToolCallParseError("Action JSON must be an object at the top level.")

        return self._to_tool_call(obj)

    # ---------- Internals ----------

    def _to_tool_call(self, obj: Dict[str, Any]) -> ToolCall:
        tool = obj.get("tool")
        inputs = obj.get("inputs", {}) or {}
        memory_keys_raw = obj.get("memory_keys", None)
        final_answer_raw = obj.get("final_answer", False)

        memory_keys = _normalize_memory_keys(memory_keys_raw)
        final_answer = _normalize_final_answer(final_answer_raw)

        if not isinstance(inputs, dict):
            # This is malformed; let the outer code trigger repair
            raise ToolCallParseError("`inputs` must be a JSON object.")

        if tool is None:
            # Valid JSON, but explicit "no tool"
            return ToolCall(
                tool=None,
                inputs={},
                final_answer=False,
                memory_keys=memory_keys,
            )

        return ToolCall(
            tool=str(tool),
            inputs=inputs,
            memory_keys=memory_keys,
            final_answer=final_answer,
        )
