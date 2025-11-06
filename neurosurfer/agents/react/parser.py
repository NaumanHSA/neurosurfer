import json, re
from typing import Optional, Dict, Any
from .types import ToolCall
from .exceptions import ToolCallParseError

# Capture an Action block with braces (non-greedy, dotall)
JSON_BLOCK = re.compile(
    r"Action:\s*({.*?})(?:$|\n|```)",
    re.DOTALL | re.IGNORECASE
)

TRAILING_COMMA = re.compile(r",\s*([}\]])")

def _strip_code_fences(text: str) -> str:
    # Remove Markdown fences like ```json ... ```
    text = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text)
    return text

def _tidy_json(s: str) -> str:
    """Clean up likely LLM mistakes in JSON."""
    s = _strip_code_fences(s).strip()
    # Remove trailing commas before } or ]
    s = TRAILING_COMMA.sub(r"\1", s)
    # Normalize booleans
    s = re.sub(r'("final_answer"\s*:\s*)"true"', r'\1 true', s, flags=re.IGNORECASE)
    s = re.sub(r'("final_answer"\s*:\s*)"false"', r'\1 false', s, flags=re.IGNORECASE)

    # Auto-close braces if model cut output early
    if s.count("{") > s.count("}"):
        s += "}" * (s.count("{") - s.count("}"))
    if s.count("[") > s.count("]"):
        s += "]" * (s.count("[") - s.count("]"))
    return s

def _force_object(s: str) -> Dict[str, Any]:
    """Try to coerce a malformed JSON string into a Python dict."""
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        # Attempt recovery by trimming to last complete brace
        last = s.rfind("}")
        if last != -1:
            try:
                return json.loads(s[:last + 1])
            except Exception:
                pass
        raise e

class ToolCallParser:
    """
    Extracts and normalizes a tool call from an LLM message.
    - tolerant to fenced blocks
    - tolerant to trailing commas and missing braces
    - returns ToolCall
    """
    def extract(self, text: str) -> Optional[ToolCall]:
        m = JSON_BLOCK.search(text)
        if not m:
            return None

        raw = _tidy_json(m.group(1))
        try:
            obj = _force_object(raw)
        except json.JSONDecodeError as e:
            raise ToolCallParseError(f"Invalid JSON in Action block: {e}") from e
        
        tool = obj.get("tool")
        inputs = obj.get("inputs", {}) or {}
        final_answer = bool(obj.get("final_answer", False))

        if tool is None:
            return ToolCall(tool=None, inputs={}, final_answer=False)
        if not isinstance(inputs, dict):
            raise ToolCallParseError("`inputs` must be a JSON object.")

        return ToolCall(tool=str(tool), inputs=inputs, final_answer=final_answer)
