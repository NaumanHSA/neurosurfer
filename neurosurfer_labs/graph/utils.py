from __future__ import annotations

import importlib
import re
from typing import Any, Mapping, Sequence, List, Dict
import logging

from .schema import GraphNode, GraphSpec, GraphInputSpec
from .errors import GraphConfigurationError

logger = logging.getLogger("neurosurfer.graph.utils")
_TMPL_RE = re.compile(r"\{\{\s*([^}]+?)\s*\}\}")


def get_from_ctx(ctx: Mapping[str, Any], path: str) -> Any:
    """
    Resolve a dotted path like "nodes.research.summary" or "inputs.topic".
    Supports nested dicts and simple list indices.
    """
    parts = path.split(".")
    cur: Any = ctx
    for part in parts:
        part = part.strip()
        if isinstance(cur, Mapping):
            cur = cur[part]
        elif isinstance(cur, (list, tuple)):
            try:
                idx = int(part)
            except ValueError:
                raise KeyError(f"Cannot index list by non-int key: {part!r}")
            cur = cur[idx]
        else:
            raise KeyError(f"Cannot descend into non-container: {cur!r}")
    return cur

def render_template(text: str, ctx: Mapping[str, Any]) -> str:
    """
    Very small templating helper.
    Replaces `{{ path.to.value }}` with a value looked up via `get_from_ctx`.
    If lookup fails, leaves the placeholder untouched.
    """
    if not text: return text
    def repl(m: re.Match) -> str:
        path = m.group(1).strip()
        try:
            v = get_from_ctx(ctx, path)
            return str(v)
        except Exception:
            return m.group(0)
    return _TMPL_RE.sub(repl, text)

def topo_sort(nodes: Sequence[GraphNode]) -> List[str]:
    """
    Topologically sort nodes based on their `depends_on` list.
    Raises GraphConfigurationError on cycles or unknown dependencies.
    """
    node_ids = {n.id for n in nodes}
    deps: Dict[str, set[str]] = {n.id: set(n.depends_on) for n in nodes}

    # Validate dependencies reference valid node IDs
    unknown = {d for n in nodes for d in n.depends_on if d not in node_ids}
    if unknown:
        raise GraphConfigurationError(f"Unknown dependency node IDs: {sorted(unknown)}")

    # Kahn's algorithm
    result: List[str] = []
    no_deps = [nid for nid, ds in deps.items() if not ds]
    while no_deps:
        nid = no_deps.pop()
        result.append(nid)
        for other, ds in deps.items():
            if nid in ds:
                ds.remove(nid)
                if not ds:
                    no_deps.append(other)

    if len(result) != len(nodes):
        raise GraphConfigurationError("Graph contains a cycle; DAG is required.")
    return result

def import_string(path: str) -> Any:
    """
    Import an object from a "module:attr" or "module.attr" path.
    Example:
        "myproj.schemas.Answer" -> myproj.schemas.Answer
    """
    if ":" in path:
        mod_name, attr = path.split(":", 1)
    else:
        parts = path.split(".")
        if len(parts) < 2:
            raise ImportError(f"Cannot import from path: {path}")
        mod_name, attr = ".".join(parts[:-1]), parts[-1]
    module = importlib.import_module(mod_name)
    try:
        return getattr(module, attr)
    except AttributeError as e:
        raise ImportError(f"Module {mod_name!r} has no attribute {attr!r}") from e


# Normalize and Validate Graph Inputs
def normalize_and_validate_graph_inputs(graph: GraphSpec, inputs: Any) -> Dict[str, Any]:
    """
    Enforce graph-level input spec if declared.

    - If `graph.inputs` is empty:
        - dict -> used as-is
        - anything else -> wrapped as `{"query": inputs}`
    - If `graph.inputs` is non-empty:
        - inputs must be a dict
        - missing required keys -> GraphConfigurationError
        - extra keys -> warned and ignored
        - values are cast according to GraphInputSpec.type
    """
    specs = graph.inputs

    # No spec: be permissive
    if not specs:
        if isinstance(inputs, dict):
            return dict(inputs)
        # Common ergonomic case: user passes a single query string
        return {"query": inputs}

    # Spec exists: require mapping
    if not isinstance(inputs, dict):
        expected = [s.name for s in specs]
        raise GraphConfigurationError(
            f"Graph '{graph.name}' expects a mapping of inputs with keys {expected}, "
            f"but got {type(inputs).__name__}."
        )

    provided = inputs
    allowed_names = {s.name for s in specs}
    extra = set(provided.keys()) - allowed_names
    if extra:
        logger.warning(
            "Ignoring extra inputs not declared in graph spec: %s",
            ", ".join(sorted(extra)),
        )

    normalized: Dict[str, Any] = {}
    for spec in specs:
        name = spec.name
        if name not in provided:
            if spec.required:
                raise GraphConfigurationError(
                    f"Missing required graph input '{name}' (type {spec.type})."
                )
            # optional -> simply skip; node prompts can handle absence
            continue
        raw_value = provided[name]
        try:
            normalized[name] = _cast_input_value(spec, raw_value)
        except Exception as e:
            raise GraphConfigurationError(
                f"Invalid value for graph input '{name}' "
                f"(expected {spec.type}, got {raw_value!r}): {e}"
            ) from e
    return normalized

def _cast_input_value(spec: GraphInputSpec, value: Any) -> Any:
    """
    Cast a raw input value to the type declared in GraphInputSpec.

    Types:
        - string
        - integer
        - float
        - boolean
        - object (dict)
        - array (list)
    Unknown types are passed through unchanged.
    """
    t = (spec.type or "string").lower()

    # string
    if t == "string":
        return str(value)

    # integer
    if t == "integer":
        if isinstance(value, bool):
            raise ValueError("boolean is not accepted as integer")
        if isinstance(value, int):
            return value
        return int(value)

    # float / number
    if t == "float":
        if isinstance(value, bool):
            raise ValueError("boolean is not accepted as float")
        if isinstance(value, (int, float)):
            return float(value)
        return float(value)

    # boolean
    if t == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            s = value.strip().lower()
            if s in {"true", "1", "yes", "y"}:
                return True
            if s in {"false", "0", "no", "n"}:
                return False
        raise ValueError("expected a boolean-like value (true/false/yes/no/0/1).")

    # object
    if t == "object":
        if isinstance(value, dict):
            return value
        raise ValueError("expected an object (mapping)")

    # array
    if t == "array":
        if isinstance(value, (list, tuple)):
            return list(value)
        raise ValueError("expected an array (list/tuple)")

    # unknown / custom type => pass through
    return value