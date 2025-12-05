import json
import re
from typing import Optional, Dict, Any, List

from .types import ToolCall
from .exceptions import ToolCallParseError
from ..common.utils import extract_and_repair_json

# Capture an Action block with braces (non-greedy, dotall)
JSON_BLOCK = re.compile(
    r"Action:\s*({.*?})(?:$|\n|```)",
    re.DOTALL | re.IGNORECASE
)

TRAILING_COMMA = re.compile(r",\s*([}\]])")


def _strip_code_fences(text: str) -> str:
    """
    Remove Markdown fences like ```json ... ``` or ``` ... ```.

    We do this *before* trying to parse JSON so that the parser is not
    confused by code-block markers.
    """
    if not text:
        return text

    # Remove ```lang\n at the start of a block
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
      - list of strings
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        # Filter to strings only, ignore garbage
        return [str(k) for k in raw if isinstance(k, (str, int))]
    # Anything else → ignore
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


class ToolCallParser:
    """
    Extracts and normalizes a tool call from an LLM message.

    Behavior:
    - Prefers a proper `Action: { ... }` block (your normal ReAct pattern).
    - Tolerant to fenced blocks (``` / ```json).
    - Tolerant to trailing commas and slightly malformed JSON.
    - If no `Action:` is found but there is a bare JSON object with a "tool"
      field, it will treat that as the Action.
    - Returns a ToolCall (possibly with tool=None if JSON is missing "tool").
    """

    def extract(self, text: str) -> Optional[ToolCall]:
        if not text:
            return None

        # First, strip obvious code fences from the whole message.
        cleaned = _strip_code_fences(text)

        # 1) Try the canonical pattern: `Action: { ... }`
        m = JSON_BLOCK.search(cleaned)
        if m:
            raw = _tidy_json(m.group(1))
            obj = self._parse_json_with_repair(raw)
            return self._to_tool_call(obj)

        # 2) Fallback: try to recover a JSON object from the whole text
        #    (handles the case where the model only emits ```json { ... } ```).
        try:
            obj = extract_and_repair_json(cleaned, return_dict=True)
        except Exception:
            # No valid JSON → no Action
            return None

        # If we got something that looks like a tool call, use it.
        if isinstance(obj, dict) and ("tool" in obj or "inputs" in obj):
            return self._to_tool_call(obj)

        # Otherwise, treat as "no tool call" and let the agent decide.
        return None

    # ---------- Internals ----------

    def _parse_json_with_repair(self, raw: str) -> Dict[str, Any]:
        """
        Try to parse JSON, fall back to extract_and_repair_json if needed.
        """
        try:
            return _force_object(raw)
        except json.JSONDecodeError:
            try:
                return extract_and_repair_json(raw, return_dict=True)
            except Exception as e:
                raise ToolCallParseError(f"Invalid JSON in Action block: {e}") from e

    def _to_tool_call(self, obj: Dict[str, Any]) -> ToolCall:
        tool = obj.get("tool")
        inputs = obj.get("inputs", {}) or {}
        memory_keys_raw = obj.get("memory_keys", None)
        final_answer_raw = obj.get("final_answer", False)

        # Normalize fields
        memory_keys = _normalize_memory_keys(memory_keys_raw)
        final_answer = _normalize_final_answer(final_answer_raw)

        if tool is None:
            # We found JSON but it doesn't actually specify a tool.
            # Return an "empty" ToolCall so the agent can decide what to do.
            return ToolCall(
                tool=None,
                inputs={},
                final_answer=False,
                memory_keys=memory_keys,
            )

        if not isinstance(inputs, dict):
            raise ToolCallParseError("`inputs` must be a JSON object.")

        return ToolCall(
            tool=str(tool),
            inputs=inputs,
            memory_keys=memory_keys,
            final_answer=final_answer,
        )
