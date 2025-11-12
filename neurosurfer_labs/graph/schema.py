# neurosurfer/agents/graph/schema.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple, Type
from pydantic import BaseModel, create_model

_SIMPLE = {"str": str, "string": str, "int": int, "integer": int, "float": float, "number": float, "bool": bool, "boolean": bool}

def _parse_field(line: str) -> Tuple[str, Any]:
    # "name: float" | "tags: str[]" | "choice: enum{a,b,c}" | "meta: object{unit:str, m:int}"
    name, typ = [x.strip() for x in line.split(":", 1)]
    tl = typ.lower()

    # enum
    if tl.startswith("enum{") and tl.endswith("}"):
        items = [x.strip() for x in typ[5:-1].split(",")]
        from typing import Literal
        return name, Literal[tuple(items)]  # type: ignore

    # arrays like str[]
    for base in _SIMPLE:
        if tl == f"{base}[]":
            from typing import List as TList
            return name, TList[_SIMPLE[base]]  # type: ignore

    # object{...}
    if tl.startswith("object{") and tl.endswith("}"):
        inner = typ[len("object{"):-1].strip()
        fields: Dict[str, Tuple[Any, Any]] = {}
        if inner:
            for pair in [p.strip() for p in inner.split(",") if p.strip()]:
                k, t = [s.strip() for s in pair.split(":", 1)]
                sub_ann = _SIMPLE.get(t.strip().lower(), str)
                fields[k] = (sub_ann, ...)
        Sub = create_model("SubObject", **fields)  # type: ignore
        return name, Sub

    # primitive
    return name, _SIMPLE.get(tl, str)

def pydantic_model_from_outputs(outputs_spec: List[str], model_name: str = "StructuredOutput") -> Type[BaseModel]:
    fields: Dict[str, Tuple[Any, Any]] = {}
    for line in outputs_spec:
        if ":" not in line:
            # tolerate bare names like "text" => str
            fields[line.strip()] = (str, ...)
        else:
            k, ann = _parse_field(line)
            fields[k] = (ann, ...)
    return create_model(model_name, **fields)  # type: ignore

def structure_block_for_outputs(outputs_spec: List[str], title: str = "Output") -> str:
    # Minimal, single readable block. Example:
    # Output = {
    #   num1: float
    #   num2: float
    #   operation: enum{add,subtract,multiply,divide}
    # }
    lines = [f"{title} = {{"] + [f"  {line.strip()}" for line in outputs_spec] + ["}"]
    return "\n".join(lines)
