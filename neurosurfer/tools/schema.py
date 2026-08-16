"""pydantic model → JSON Schema for tool inputs.

Both provider adapters consume the returned schema (Anthropic ``input_schema`` /
OpenAI ``function.parameters``). We strip pydantic's ``title`` noise and inline
trivial ``$defs`` so weaker local models see a flat, predictable schema.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

#: Schema keywords whose keys are **names the author chose**, not keywords. The
#: values are schemas; the keys must survive untouched.
_NAME_KEYED = ("properties", "$defs", "definitions", "patternProperties",
               "dependentSchemas")


def _strip_titles(node: Any) -> Any:
    """Drop pydantic's ``title`` annotations — and **only** the annotations.

    The one-line version filtered ``k != "title"`` at every level, which also
    deleted a *field named* ``title`` from ``properties``: its key is a property
    name, not a keyword. So a model with a ``title`` field was described to the
    LLM without it, while ``required`` still demanded it — the model could not
    supply what it was never shown, and structured output failed deterministically
    with "title Field required", three retries deep, on every attempt.

    Found by tutorial 01's `MovieReview`, whose first field is `title`. It affected
    any tool argument or output schema with that name.
    """
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for k, v in node.items():
            if k == "title":
                continue
            if k in _NAME_KEYED and isinstance(v, dict):
                # Recurse into the *values*; the keys here are field names.
                out[k] = {name: _strip_titles(sub) for name, sub in v.items()}
            else:
                out[k] = _strip_titles(v)
        return out
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
