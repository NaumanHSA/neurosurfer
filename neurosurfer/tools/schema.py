"""pydantic model → JSON Schema for tool inputs.

Both provider adapters consume the returned schema (Anthropic ``input_schema`` /
OpenAI ``function.parameters``). We strip pydantic's ``title`` noise and inline
trivial ``$defs`` so weaker local models see a flat, predictable schema.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


def _strip_titles(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _strip_titles(v) for k, v in node.items() if k != "title"}
    if isinstance(node, list):
        return [_strip_titles(v) for v in node]
    return node


def model_to_schema(model: type[BaseModel]) -> dict[str, Any]:
    schema = model.model_json_schema()
    schema = _strip_titles(schema)
    schema.setdefault("type", "object")
    # additionalProperties False helps strict-mode local servers reject junk.
    schema.setdefault("additionalProperties", False)
    return schema
