from __future__ import annotations
import json, re
from typing import Any, Dict, List, Optional, Tuple, Type
from pydantic import BaseModel as PydModel, create_model

# ---------- compact-structure helpers ----------

_SIMPLE_TOKENS = {
    "str": str, "string": str,
    "int": int, "integer": int,
    "float": float, "number": float,
    "bool": bool, "boolean": bool,
}

def _parse_outputs_line(line: str) -> Tuple[str, Any]:
    """
    Convert "name: type" lines to Python type hints suitable for create_model.
    Supports:
      - primitives: str|int|float|bool
      - arrays: str[] / int[] / float[] / bool[]
      - enum: enum{a,b,c}
      - object: object{field1:str, field2:int}
    """
    name, typ = [x.strip() for x in line.split(":", 1)]
    tl = typ.lower()

    # enum
    if tl.startswith("enum{") and tl.endswith("}"):
        items = [x.strip() for x in typ[5:-1].split(",") if x.strip()]
        from typing import Literal
        return name, Literal[tuple(items)]  # type: ignore

    # arrays
    for base in _SIMPLE_TOKENS:
        if tl == f"{base}[]":
            from typing import List as TList
            return name, TList[_SIMPLE_TOKENS[base]]  # type: ignore

    # object
    if tl.startswith("object{") and tl.endswith("}"):
        inner = typ[len("object{"):-1].strip()
        fields: Dict[str, Tuple[Any, Any]] = {}
        if inner:
            parts = [p.strip() for p in inner.split(",") if p.strip()]
            for part in parts:
                if ":" not in part:
                    raise ValueError(f"Bad object field spec: {part}")
                k, t = [s.strip() for s in part.split(":", 1)]
                fields[k] = (_SIMPLE_TOKENS.get(t.lower(), str), ...)
        sub = create_model("SubObject", **fields)  # type: ignore
        return name, sub

    # primitive default
    return name, _SIMPLE_TOKENS.get(tl, str)

def pydantic_model_from_outputs(outputs_spec: List[str], model_name: str = "StructuredOutput") -> Type[PydModel]:
    fields: Dict[str, Tuple[Any, Any]] = {}
    for line in outputs_spec:
        if ":" not in line:
            fields[line.strip()] = (str, ...)
        else:
            k, ann = _parse_outputs_line(line)
            fields[k] = (ann, ...)
    return create_model(model_name, **fields)  # type: ignore

def structure_block(outputs_spec: List[str], title: str = "Output") -> str:
    lines = [f"{title} = {{"] + [f"  {ln.strip()}" for ln in outputs_spec] + ["}"]
    return "\n".join(lines)

# ---------- parsing/normalization helpers ----------

def json_first_object(text: str) -> Optional[Dict[str, Any]]:
    s = text.strip().strip("`")
    try:
        o = json.loads(s)
        return o if isinstance(o, dict) else None
    except Exception:
        pass
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if not m:
        return None
    try:
        o = json.loads(m.group(0))
        return o if isinstance(o, dict) else None
    except Exception:
        return None

def apply_synonyms(inputs: Dict[str, Any], synonyms: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    if not synonyms:
        return inputs
    out = dict(inputs)
    for field, mapping in synonyms.items():
        if field in out and isinstance(out[field], str):
            out[field] = mapping.get(out[field], out[field])
    return out
