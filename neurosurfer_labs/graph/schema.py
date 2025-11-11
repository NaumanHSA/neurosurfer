# neurosurfer_labs/schema/dynamic.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple, Union, Optional
from pydantic import BaseModel, create_model
import typing

# ---------- Simple type spec -> Pydantic ----------
# Allowed spec forms:
#   "str" | "int" | "float" | "bool"
#   {"$array": <spec>}         # list[...] 
#   {"$object": { key: <spec>, ... }}  # nested object

_SIMPLE = {"str": str, "int": int, "float": float, "bool": bool}

def spec_to_type(spec: Any) -> Any:
    if isinstance(spec, str):
        if spec not in _SIMPLE:
            raise ValueError(f"Unsupported primitive: {spec}")
        return _SIMPLE[spec]

    if isinstance(spec, dict):
        if "$array" in spec:
            inner = spec_to_type(spec["$array"])
            return List[inner]  # typing.List
        if "$object" in spec:
            fields = spec["$object"]
            if not isinstance(fields, dict):
                raise ValueError("`$object` must map to a dict")
            sub_fields: Dict[str, Tuple[Any, ...]] = {}
            for k, v in fields.items():
                sub_fields[k] = (spec_to_type(v), ...)
            # name an anonymous nested object (won’t leak in prompt; only type identity)
            return create_model("Obj", **sub_fields)  # type: ignore
    raise ValueError(f"Unsupported spec: {spec!r}")

def pydantic_model_from_outputs(outputs: Dict[str, Any], model_name: str="NodeOutput") -> type[BaseModel]:
    """
    Build a Pydantic model from outputs mapping:
      outputs:
        num1: float
        meta: { $object: { unit: str, method: str } }
        tags: { $array: str }
    """
    fields: Dict[str, Tuple[Any, ...]] = {}
    for name, spec in outputs.items():
        t = spec_to_type(spec)
        fields[name] = (t, ...)
    return create_model(model_name, **fields)

# ---------- Compact structure guidance ----------
def structure_lines_from_spec(outputs: Dict[str, Any], indent: int=2, max_inline: int=80) -> str:
    def render(spec: Any, lvl: int=0) -> str:
        pad = " " * (indent * lvl)
        if isinstance(spec, str):
            return spec  # "str"/"int"/...

        if isinstance(spec, dict) and "$array" in spec:
            inner = render(spec["$array"], lvl)
            # inline if short
            candidate = f"[{inner}]"
            if len(candidate) <= max_inline and "\n" not in inner:
                return candidate
            return "[\n" + pad + " " * indent + inner + "\n" + pad + "]"

        if isinstance(spec, dict) and "$object" in spec:
            obj = spec["$object"]
            parts = [f"{k}: {render(v, lvl+1)}" for k, v in obj.items()]
            inline = "{ " + ", ".join(parts) + " }"
            if len(inline) <= max_inline and all("\n" not in p for p in parts):
                return inline
            # multiline
            body = "\n".join((" " * (indent * (lvl+1))) + p for p in parts)
            return "{\n" + body + "\n" + pad + "}"
        raise ValueError(f"Bad spec: {spec!r}")

    lines = []
    for name, spec in outputs.items():
        lines.append(f"{name}: {render(spec, 1)}")
    return "\n".join(lines)

def structure_block_for_outputs(outputs: Dict[str, Any], title: Optional[str]=None) -> str:
    """
    Returns a compact, LLM-friendly “Structure” section (no JSON Schema verbosity).
    """
    head = (title + " = {\n") if title else "{\n"
    body = structure_lines_from_spec(outputs)
    tail = "\n}"
    return head + "  " + body.replace("\n", "\n  ") + tail
