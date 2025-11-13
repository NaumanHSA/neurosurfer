from __future__ import annotations

import importlib
import re
from typing import Any, Mapping, Sequence, List, Dict

from .schema import GraphNode
from .errors import GraphConfigurationError


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

    if not text:
        return text

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
