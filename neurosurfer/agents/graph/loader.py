from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Union

import yaml
from pydantic import BaseModel, ValidationError

from .schema import Graph, GraphNode
from .errors import GraphConfigurationError
from .utils import topo_sort  # uses same error type for cycles / unknown deps

logger = logging.getLogger("neurosurfer.agents.graph.loader")


def _get_model_fields(model: type[BaseModel]) -> set[str]:
    """
    Return the set of field names for a Pydantic model, compatible with
    both Pydantic v1 (`__fields__`) and v2 (`model_fields`).
    """
    if hasattr(model, "model_fields"):  # Pydantic v2
        return set(model.model_fields.keys())  # type: ignore[attr-defined]
    # Pydantic v1
    return set(model.__fields__.keys())  # type: ignore[attr-defined]


def _sanitize_graph_dict(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove unknown keys from the raw dict for Graph / GraphNode and
    emit warnings describing what was ignored.

    This makes the loader more forgiving: extra keys in YAML won't crash
    validation, but the user is clearly informed.

    Top-level extras -> warning and ignore.
    Per-node extras -> warning and ignore.
    """
    cleaned: Dict[str, Any] = dict(raw)

    # --- Top-level Graph fields ---
    graph_fields = _get_model_fields(Graph)
    extra_graph_keys = set(cleaned.keys()) - graph_fields
    if extra_graph_keys:
        logger.warning(
            "Ignoring unknown top-level keys in graph spec: %s",
            ", ".join(sorted(extra_graph_keys)),
        )
        for key in extra_graph_keys:
            cleaned.pop(key, None)

    # --- Per-node fields ---
    node_fields = _get_model_fields(GraphNode)

    nodes = cleaned.get("nodes", [])
    if not isinstance(nodes, list):
        # Let Pydantic handle the actual type error, but we still keep it as-is.
        return cleaned

    new_nodes = []
    for idx, node in enumerate(nodes):
        if not isinstance(node, dict):
            # Again, keep as-is; Pydantic will raise a nice type error later.
            new_nodes.append(node)
            continue

        node_clean = dict(node)
        extra_node_keys = set(node_clean.keys()) - node_fields
        if extra_node_keys:
            path = f"nodes[{idx}]"
            logger.warning(
                "Ignoring unknown keys in %s: %s",
                path,
                ", ".join(sorted(extra_node_keys)),
            )
            for key in extra_node_keys:
                node_clean.pop(key, None)

        new_nodes.append(node_clean)

    cleaned["nodes"] = new_nodes
    return cleaned


def _format_validation_error(e: ValidationError) -> str:
    """
    Turn a Pydantic ValidationError into a human-readable multi-line message
    that points to the exact location(s) in the graph spec.
    """
    parts = ["Invalid graph specification:"]
    for err in e.errors():
        loc = err.get("loc", ())
        msg = err.get("msg", "")
        typ = err.get("type", "")

        # loc is a tuple like ('nodes', 0, 'id') -> "nodes[0].id"
        loc_str_parts = []
        for p in loc:
            if isinstance(p, int):
                loc_str_parts[-1] = f"{loc_str_parts[-1]}[{p}]"
            else:
                loc_str_parts.append(str(p))
        loc_str = ".".join(loc_str_parts) if loc_str_parts else "<root>"

        parts.append(f"  - at {loc_str}: {msg} (type={typ})")
    return "\n".join(parts)


def _pydantic_from_dict(model: type[BaseModel], data: Dict[str, Any]) -> BaseModel:
    """
    Wrapper around Pydantic model construction with version compatibility.
    """
    if hasattr(model, "model_validate"):  # Pydantic v2
        return model.model_validate(data)  # type: ignore[attr-defined]
    # Pydantic v1
    return model.parse_obj(data)  # type: ignore[call-arg]


def _validate_graph_spec(spec: Graph) -> None:
    """
    Perform semantic validation of a Graph:

    - Ensure at least one node exists.
    - Ensure all `outputs` refer to existing node IDs.
    - Ensure all `depends_on` entries refer to existing node IDs.
    - Ensure no node depends on itself.
    - Ensure the graph is a DAG (no cycles) via topological sort.

    Raises:
        GraphConfigurationError on any invalid condition.
    """
    nodes = spec.nodes
    if not nodes:
        raise GraphConfigurationError("Graph must define at least one node.")

    node_ids = {n.id for n in nodes}

    # ---- outputs -> valid node IDs ----
    unknown_outputs = [out for out in spec.outputs if out not in node_ids]
    if unknown_outputs:
        raise GraphConfigurationError(
            "Graph outputs refer to unknown node IDs: "
            f"{', '.join(sorted(unknown_outputs))}. "
            f"Defined node IDs: {', '.join(sorted(node_ids))}."
        )

    # Optional: warn on empty outputs (but don't fail)
    if not spec.outputs:
        logger.warning(
            "Graph '%s' does not define any outputs. "
            "Executor will default to the last node in topological order.",
            spec.name,
        )

    # ---- depends_on -> valid node IDs & no self-deps ----
    dep_errors: list[str] = []
    for idx, n in enumerate(nodes):
        # self-dependency check
        if n.id in (n.depends_on or []):
            dep_errors.append(
                f"nodes[{idx}] (id={n.id!r}) cannot depend on itself in 'depends_on'."
            )

        for dep in n.depends_on or []:
            if dep not in node_ids:
                dep_errors.append(
                    f"nodes[{idx}] (id={n.id!r}) depends_on unknown node id {dep!r}."
                )

    if dep_errors:
        raise GraphConfigurationError(
            "Invalid dependencies in graph specification:\n  - " +
            "\n  - ".join(dep_errors)
        )

    # ---- DAG / cycle check via topo_sort ----
    # topo_sort already checks for unknown deps internally, but we've done
    # that above for better error messages, so here it's mainly cycle detection.
    try:
        topo_sort(nodes)
    except GraphConfigurationError as e:
        # Re-raise with additional context, but keep the original message
        raise GraphConfigurationError(
            f"Graph '{spec.name}' has invalid dependency structure (likely a cycle): {e}"
        ) from e


def load_graph_from_dict(data: Dict[str, Any]) -> Graph:
    """
    Load a Graph from a raw dict, with:
      - Unknown keys warned and ignored
      - Detailed error messages on schema violations
      - Semantic validation of outputs / depends_on / DAG structure
    """
    if not isinstance(data, dict):
        raise GraphConfigurationError(
            f"Graph spec must be a mapping at the top level, got {type(data).__name__!r}."
        )
    cleaned = _sanitize_graph_dict(data)
    try:
        spec = _pydantic_from_dict(Graph, cleaned)  # type: ignore[return-value]
    except ValidationError as e:
        msg = _format_validation_error(e)
        raise GraphConfigurationError(msg) from e

    # Additional semantic validation (outputs, depends_on, cycles, etc.)
    _validate_graph_spec(spec)
    return spec


def load_graph(path: Union[str, Path]) -> Graph:
    """
    Load a Graph from a YAML or JSON file.

    Features:
      - YAML/JSON parse errors are reported with filename and cause.
      - Unknown keys are warned and ignored (top-level + per-node).
      - Schema errors show precise locations (e.g., nodes[1].id).
      - Semantic errors (unknown outputs, bad depends_on, cycles) are reported
        via GraphConfigurationError with clear messages.

    Example:
        spec = load_graph("flows/blog_workflow.yaml")
    """
    p = Path(path)

    try:
        text = p.read_text(encoding="utf-8")
    except OSError as e:
        raise GraphConfigurationError(
            f"Failed to read graph file {p!s}: {e}"
        ) from e

    # Parse YAML/JSON
    if p.suffix.lower() in {".yaml", ".yml"}:
        try:
            data = yaml.safe_load(text) or {}
        except yaml.YAMLError as e:
            raise GraphConfigurationError(
                f"Failed to parse YAML graph file {p!s}: {e}"
            ) from e
    else:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise GraphConfigurationError(
                f"Failed to parse JSON graph file {p!s}: {e}"
            ) from e

    return load_graph_from_dict(data)
