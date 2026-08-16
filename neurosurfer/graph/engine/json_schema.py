"""JSON Schema → pydantic model, for a node that declares its own output shape.

## Why the schema is written here rather than imported

``output_schema`` began as an import path — ``my_module:ResultModel`` — which
means the only way to get structured output from a node was to put a Python file
on the server and restart it. That is the same defect ``callable`` had on the
``function`` node: *the field that decides what the node produces cannot be
filled in from the surface you build the node on.* An author working in the studio could pick "structured" and then had
nowhere to say what the structure was.

JSON Schema is the answer rather than a bespoke field list because it is a format
authors already know and can paste from somewhere else — an OpenAPI document, a
tool definition, another model's response format — and because it stays
expressive as the shapes get harder: nested objects, arrays of objects, enums and
optionals all have an obvious spelling in it, and none of them need new UI here.

## Why it is not `exec`

Building the model is :func:`pydantic.create_model` over a dict, so nothing in
the workflow is executed to find out what a node returns. That matters more than
it looks: the alternative — letting an author write a ``class Result(BaseModel)``
and running it — is a sandbox question, and the plan deliberately keeps
"describes a shape" on the near side of the line that "runs arbitrary code" sits
on. A schema is data all the way down.

The import path still works, and is still the right answer for a model that
already exists in code and is shared between workflows.
"""

from __future__ import annotations

from typing import Any, Literal, Union

from pydantic import BaseModel, Field, create_model

__all__ = ["JsonSchemaError", "model_from_json_schema"]


class JsonSchemaError(ValueError):
    """The schema is not something a model can be built from.

    Carries the JSON-Pointer-ish path to the offending sub-schema, because "type
    'strng' is not known" is only actionable when you know *which* property said
    it.
    """


#: The scalar spellings. `integer` and `number` are distinct in JSON Schema and
#: mapping both to `float` would silently accept 3.5 where an int was declared.
_SCALARS: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
}


def _fail(where: str, message: str) -> JsonSchemaError:
    return JsonSchemaError(f"{where or 'schema'}: {message}")


def _python_type(schema: dict[str, Any], where: str, name_hint: str) -> Any:
    """The annotation for one sub-schema."""
    if not isinstance(schema, dict):
        raise _fail(where, "expected an object describing a type")

    # `enum` wins over `type`: it is the stricter statement, and a schema that
    # gives both means the enum.
    enum = schema.get("enum")
    if enum is not None:
        if not isinstance(enum, list) or not enum:
            raise _fail(where, "`enum` must be a non-empty list")
        return Literal[tuple(enum)]  # type: ignore[valid-type]

    declared = schema.get("type")

    # A union — `["string", "null"]` is how optionality is usually spelled.
    if isinstance(declared, list):
        parts = [
            _python_type({**schema, "type": t}, where, name_hint)
            for t in declared
            if t != "null"
        ]
        if not parts:
            raise _fail(where, "a type list of only `null` describes nothing")
        annotation = parts[0] if len(parts) == 1 else Union[tuple(parts)]  # noqa: UP007
        return annotation | None if "null" in declared else annotation

    if declared is None:
        # No `type` at all is legal JSON Schema and means "anything". Honour it
        # rather than guessing: a node returning a free-form object is a real
        # thing to want, and refusing it would be inventing a rule.
        return Any

    if declared in _SCALARS:
        return _SCALARS[declared]

    if declared == "array":
        items = schema.get("items")
        if items is None:
            return list[Any]
        return list[_python_type(items, f"{where}[]", f"{name_hint}Item")]

    if declared == "object":
        # A nested object becomes a nested model, so validation reaches all the
        # way down rather than stopping at `dict`.
        if not schema.get("properties"):
            return dict[str, Any]
        return model_from_json_schema(schema, name=name_hint, _where=where)

    raise _fail(where, f"unknown type {declared!r}")


def model_from_json_schema(
    schema: dict[str, Any],
    name: str = "Output",
    *,
    _where: str = "",
) -> type[BaseModel]:
    """Build a pydantic model from a JSON Schema object.

    Supports the subset an author actually writes for an LLM's response shape:
    objects, nested objects, arrays, the four scalars, ``enum``, ``required``,
    ``default`` and ``description``. Anything outside that raises
    :class:`JsonSchemaError` with the path to the part that could not be read —
    which is what validation surfaces, before the run rather than inside it.
    """
    if not isinstance(schema, dict):
        raise _fail(_where, "expected a JSON object")

    declared = schema.get("type", "object")
    if declared != "object":
        raise _fail(_where, f"the top level must be an object, not {declared!r}")

    properties = schema.get("properties")
    if not isinstance(properties, dict) or not properties:
        raise _fail(_where, "needs a non-empty `properties` object")

    required = schema.get("required", [])
    if not isinstance(required, list):
        raise _fail(_where, "`required` must be a list of property names")
    unknown = [r for r in required if r not in properties]
    if unknown:
        raise _fail(_where, f"`required` names properties that do not exist: {unknown}")

    fields: dict[str, Any] = {}
    for prop, sub in properties.items():
        where = f"{_where}.{prop}" if _where else prop
        annotation = _python_type(sub, where, f"{name}_{prop}".title().replace("_", ""))
        description = sub.get("description") if isinstance(sub, dict) else None

        if prop in required:
            default: Any = ...
        elif isinstance(sub, dict) and "default" in sub:
            default = sub["default"]
        else:
            # Optional with no stated default. The annotation admits None so the
            # model can be built when the field is simply absent.
            default = None
            annotation = annotation | None

        fields[prop] = (annotation, Field(default, description=description))

    # A model name has to be an identifier; the caller's hint may be a node id
    # with hyphens in it.
    safe = "".join(ch for ch in name if ch.isalnum() or ch == "_") or "Output"
    return create_model(safe, **fields)  # type: ignore[call-overload, no-any-return]
