# neurosurfer_labs/graph/utils.py
from __future__ import annotations
import json, re
from typing import Any, Dict, List, Tuple, Type
from pydantic import BaseModel, create_model

def json_first_object(text: str) -> Dict[str, Any] | None:
    s = text.strip().strip("`")
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if not m: return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None

def jsonpath_lite_get(obj: Any, path: str):
    if not path or not path.startswith("$."):
        return None
    cur = obj
    for part in path[2:].split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur

# --- schema parsing: "num1: float", "thing: enum{a,b,c}", "count: int"
_PRIMS = {
    "str": (str, ...), 
    "string": (str, ...), 
    "int": (int, ...), 
    "integer": (int, ...),
    "float": (float, ...), 
    "number": (float, ...), 
    "bool": (bool, ...), 
    "boolean": (bool, ...)
}

def parse_field_spec(s: str) -> Tuple[str, Any, Dict[str, Any]]:
    # returns (name, annotation, field_kwargs)
    name, typ = [x.strip() for x in s.split(":", 1)]
    typ_l = typ.lower()
    if typ_l.startswith("enum{") and typ_l.endswith("}"):
        items = [x.strip() for x in typ[5:-1].split(",")]
        # represent enum as string literal union
        from typing import Literal
        ann = Literal[tuple(items)]  # type: ignore
        return name, ann, {}
    if typ_l.endswith("[]") or typ_l.startswith("list["):
        # very light: treat as list[str] or list[float], default to list[str]
        inner = "str"
        for k in _PRIMS.keys():
            if k in typ_l:
                inner = k
                break
        from typing import List as TList
        ann = TList[ _PRIMS.get(inner,(str,...))[0] ]  # type: ignore
        return name, ann, {}
    base = _PRIMS.get(typ_l, (str, ...))
    return name, base[0], {}

def pydantic_from_output_specs(specs: List[str]) -> Type[BaseModel]:
    fields = {}
    for line in specs:
        k, ann, kw = parse_field_spec(line)
        fields[k] = (ann, ...)
    return create_model("StructuredNodeOutput", **fields)  # type: ignore

def make_tools_catalog(toolkit, tool_names: List[str]) -> str:
    blocks = []
    for name in tool_names:
        t = toolkit.registry.get(name)
        if not t: continue
        blocks.append(t.get_tool_description().strip())
    return "\n\n".join(blocks)
