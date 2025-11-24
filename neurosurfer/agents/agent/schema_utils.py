from __future__ import annotations
from typing import Any, Dict, List, Set, Optional, Tuple, Type, Union, get_origin, get_args
from dataclasses import dataclass

from pydantic import BaseModel as PydanticModel
from .templates import STRUCTURED_CONTRACT_PROMPT


# Options for structured prompts
@dataclass
class StructuredPromptOptions:
    title: bool = True
    comment_for_arrays: bool = True
    rules_minimal: bool = True
    max_inline_chars: int = 100           # wrap to multiline if an inline object/array exceeds this length
    indent: int = 4                       # spaces per indent level
    array_comment: str = "add as many as needed"
    object_comment: str = "object with fields"
    trail_commas: bool = False            # purely stylistic; JSON-like without commas if False

# Pydantic -> ultra-minimal shape
def _type_to_pyhint(t: Any) -> str:
    """
    Map a python/pydantic type annotation to a compact hint string:
      - str | int | float | bool | object
      - [T] for homogeneous arrays
      - { a: str, b: int } for nested models (inline)
    """
    origin = get_origin(t)
    args = get_args(t)

    # Optional[T] -> T
    if origin is Union and len(args) == 2 and type(None) in args:
        t = args[0] if args[1] is type(None) else args[1]
        origin = get_origin(t)
        args = get_args(t)

    # Containers
    if origin in (list, List, tuple, Tuple, set, Set):
        elem = args[0] if args else str
        return f"[{_type_to_pyhint(elem)}]"

    if origin in (dict, Dict):
        return "object"

    # Primitives
    if t in (str,):
        return "str"
    if t in (int,):
        return "int"
    if t in (float,):
        return "float"
    if t in (bool,):
        return "bool"

    # Nested Pydantic model -> inline object
    try:
        fields = getattr(t, "model_fields", None)
        if isinstance(fields, dict):
            return _py_model_to_structure_inline(t)
    except Exception:
        pass

    return "object"


def _py_model_to_structure_inline(schema_cls: Type[PydanticModel]) -> str:
    """Inline single-line object: { a: str, b: int }."""
    fields = schema_cls.model_fields
    parts = []
    for name, f in fields.items():
        parts.append(f"{name}: {_type_to_pyhint(f.annotation)}")
    return "{ " + ", ".join(parts) + " }"

def model_to_structure_block(
    schema_cls: Type[PydanticModel],
    *,
    title: bool = True,
    comment_for_arrays: bool = True,
    max_inline_chars: int = 100,
    indent: int = 2,
    array_comment: str = "add as many as needed",
    object_comment: str = "object with fields",
    trail_commas: bool = False,
) -> str:
    """
    Pretty, width-aware schema shape:

    Car: {
      make: str
      model: str
      features: [
        {
          name: str
          description: str
          location: { name: str, description: str }  // wrapped if short; multiline if long
        }
      ]  // add as many as needed
    }
    """
    return _render_top_object(schema_cls, title, comment_for_arrays, max_inline_chars, indent, array_comment, object_comment, trail_commas)

    
def _render_top_object(
    schema_cls: Type[PydanticModel],
    title: bool,
    comment_for_arrays: bool,
    max_inline_chars: int,
    indent: int,
    array_comment: str,
    object_comment: str,
    trail_commas: bool,
) -> str:
    head = f"{schema_cls.__name__}: " if title else ""
    body = _render_object_schema(
        schema_cls,
        level=0,
        max_inline_chars=max_inline_chars,
        indent=indent,
        comment_for_arrays=comment_for_arrays,
        array_comment=array_comment,
        object_comment=object_comment,
        trail_commas=trail_commas,
    )
    return f"{head}{body}"

def _render_object_schema(
    schema_cls: Type[PydanticModel],
    level: int,
    max_inline_chars: int,
    indent: int,
    comment_for_arrays: bool,
    array_comment: str,
    object_comment: str,
    trail_commas: bool,
) -> str:
    pad = " " * (indent * level)
    inner_pad = " " * (indent * (level + 1))

    # First, attempt a compact single-line form like: { a: str, b: [ { ... } ] }
    compact = _inline_object(schema_cls)
    if len(compact) <= max_inline_chars and "\n" not in compact:
        return compact

    # Otherwise, multiline
    lines: list[str] = ["{"]

    fields = schema_cls.model_fields
    items = []
    for i, (name, f) in enumerate(fields.items()):
        rendered = _render_field(
            name=name,
            annotation=f.annotation,
            level=level + 1,
            max_inline_chars=max_inline_chars,
            indent=indent,
            comment_for_arrays=comment_for_arrays,
            array_comment=array_comment,
            object_comment=object_comment,
            trail_commas=trail_commas,
        )
        if trail_commas and i < len(fields) - 1:
            rendered = rendered + ","
        items.append(inner_pad + rendered)

    lines.extend(items)
    lines.append(pad + "}")
    return "\n".join(lines)

def _inline_object(schema_cls: Type[PydanticModel]) -> str:
    """Inline single-line object if possible."""
    fields = schema_cls.model_fields
    parts = []
    for name, f in fields.items():
        parts.append(f"{name}: {_hint_or_render_inline(f.annotation)}")
    return "{ " + ", ".join(parts) + " }"

def _hint_or_render_inline(t: Any) -> str:
    """
    Return inline hint text for a type:
      - str/int/float/bool -> "str"/"int"/...
      - list[T] -> "[<hint>]" (inline if T is simple or inline object)
      - dict -> "object"
      - nested model -> "{ a: str, b: int }"
    """
    origin = get_origin(t)
    args = get_args(t)

    # Optional[T] -> T
    if origin is Union and len(args) == 2 and type(None) in args:
        t = args[0] if args[1] is type(None) else args[1]
        origin = get_origin(t)
        args = get_args(t)

    if origin in (list, List, tuple, Tuple, set, Set):
        elem = args[0] if args else str
        inner = _hint_or_render_inline(elem)
        return f"[{inner}]"

    if origin in (dict, Dict):
        return "object"

    if t in (str,):
        return "str"
    if t in (int,):
        return "int"
    if t in (float,):
        return "float"
    if t in (bool,):
        return "bool"

    # pydantic model -> inline object
    try:
        fields = getattr(t, "model_fields", None)
        if isinstance(fields, dict):
            parts = []
            for name, f in fields.items():
                parts.append(f"{name}: {_hint_or_render_inline(f.annotation)}")
            return "{ " + ", ".join(parts) + " }"
    except Exception:
        pass

    return "object"

def _render_field(
    name: str,
    annotation: Any,
    level: int,
    max_inline_chars: int,
    indent: int,
    comment_for_arrays: bool,
    array_comment: str,
    object_comment: str,
    trail_commas: bool,
) -> str:
    """
    Render a field either as:
      - inline "name: { ... }" / "name: [ ... ]" if short,
      - or multiline:
            name: [
              { ... }
            ]  // add as many as needed
    """
    origin = get_origin(annotation)
    args = get_args(annotation)

    # Unwrap Optional[T]
    if origin is Union and len(args) == 2 and type(None) in args:
        annotation = args[0] if args[1] is type(None) else args[1]
        origin = get_origin(annotation)
        args = get_args(annotation)

    # Arrays
    if origin in (list, List, tuple, Tuple, set, Set):
        elem = args[0] if args else str
        inline_elem = _hint_or_render_inline(elem)
        inline = f"{name}: [{inline_elem}]"
        if len(inline) <= max_inline_chars and "{" not in inline_elem or ("{" in inline_elem and len(inline_elem) <= max_inline_chars):
            # Keep inline if short
            if comment_for_arrays:
                inline += f"  // {array_comment}"
            return inline

        # Multiline array
        pad = " " * (indent * level)
        inner_pad = " " * (indent * (level + 1))
        lines = [f"{name}: ["]
        # If elem is a model, render multiline object example
        elem_origin = get_origin(elem)
        try:
            fields = getattr(elem, "model_fields", None)
            if isinstance(fields, dict):
                # render object multiline
                lines.append(inner_pad + "{")
                for i, (n2, f2) in enumerate(fields.items()):
                    sub = _render_field(
                        name=n2,
                        annotation=f2.annotation,
                        level=level + 2,
                        max_inline_chars=max_inline_chars,
                        indent=indent,
                        comment_for_arrays=comment_for_arrays,
                        array_comment=array_comment,
                        object_comment=object_comment,
                        trail_commas=trail_commas,
                    )
                    if trail_commas and i < len(fields) - 1:
                        sub = sub + ","
                    lines.append(" " * (indent * (level + 2)) + sub)
                lines.append(inner_pad + "}")
            else:
                # primitive/list inline as a single item
                lines.append(inner_pad + inline_elem)
        except Exception:
            lines.append(inner_pad + inline_elem)
        lines.append(pad + "]" + (f"  // {array_comment}" if comment_for_arrays else ""))
        return "\n".join(lines)

    # Dict -> object
    if origin in (dict, Dict):
        return f"{name}: object  // {object_comment}"

    # Primitive
    if annotation in (str, int, float, bool):
        return f"{name}: {_hint_or_render_inline(annotation)}"

    # Nested object (pydantic model)
    try:
        fields = getattr(annotation, "model_fields", None)
        if isinstance(fields, dict):
            inline = f"{name}: {_hint_or_render_inline(annotation)}"
            if len(inline) <= max_inline_chars and "\n" not in inline:
                return inline
            # multiline nested object
            pad = " " * (indent * level)
            inner_pad = " " * (indent * (level + 1))
            lines = [f"{name}: {{"]

            items = []
            for i, (n2, f2) in enumerate(fields.items()):
                sub = _render_field(
                    name=n2,
                    annotation=f2.annotation,
                    level=level + 1,
                    max_inline_chars=max_inline_chars,
                    indent=indent,
                    comment_for_arrays=comment_for_arrays,
                    array_comment=array_comment,
                    object_comment=object_comment,
                    trail_commas=trail_commas,
                )
                if trail_commas and i < len(fields) - 1:
                    sub = sub + ","
                items.append(inner_pad + sub)

            lines.extend(items)
            lines.append(pad + "}")
            return "\n".join(lines)
    except Exception:
        pass

    # Fallback
    return f"{name}: object  // {object_comment}"

def build_structured_system_prompt(
    base_system_prompt: str,
    schema_cls: Type[PydanticModel],
    *,
    options: Optional[StructuredPromptOptions] = None,
    use_model_json_schema: bool = True,
) -> str:
    opts = options or StructuredPromptOptions()
    base_system_prompt = (
        "You are a stateless tool router"
        "Your task is to select exactly ONE tool from the catalog below and output STRICT JSON describing how to call it."
    ) or base_system_prompt.strip()
    rules = [
        "Return a single JSON object that matches the structure below.",
        "Valid JSON only (RFC 8259). No code fences, no markdown, no explanations.",
        "Do not include extra keys beyond the structure.",
    ]
    if not opts.rules_minimal:
        rules.append('If a field is unknown, return an empty but valid value (e.g., "", 0, [], false).')

    model_structure = schema_cls.model_json_schema() if use_model_json_schema else model_to_structure_block(
        schema_cls,
        title=opts.title,
        comment_for_arrays=opts.comment_for_arrays,
        max_inline_chars=opts.max_inline_chars,
        indent=opts.indent,
        array_comment=opts.array_comment,
        object_comment=opts.object_comment,
        trail_commas=opts.trail_commas,
    )
    sys_prompt = STRUCTURED_CONTRACT_PROMPT.format(base_system_prompt=base_system_prompt, schema=model_structure)
    return sys_prompt, model_structure

def maybe_unwrap_named_root(json_obj: dict, schema_cls: Type[PydanticModel]) -> dict:
    """
    If model returned {"Car": {...}} for schema `Car`, unwrap to {...}.
    Otherwise return the original string.
    """
    if isinstance(json_obj, dict) and len(json_obj) == 1:
        [(k, v)] = json_obj.items()
        if isinstance(k, str) and k.strip() == schema_cls.__name__ and isinstance(v, dict):
            return v
    return json_obj
