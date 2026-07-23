"""Programmatic graph builder (Phase 1j).

A fluent Python API that constructs the same :class:`Graph` IR as YAML, so the
Architect agent (and users) can author workflows in code with full control-flow
support. :meth:`GraphBuilder.build` runs the *same* semantic validation the YAML
loader does (via :func:`load_graph_from_dict`), guaranteeing builder output and
YAML output are identical and equally validated.

Example
-------
::

    g = (
        GraphBuilder("triage", description="Route a ticket")
        .input("text", type="string", required=True)
        .base("classify", purpose="Classify urgency of: {text}", writes="label")
        .router(
            "route",
            cases=[{"when": "nodes.classify == 'urgent'", "to": "page"}],
            default="queue",
            depends_on=["classify"],
        )
        .base("page", purpose="Page on-call", depends_on=["route"])
        .base("queue", purpose="Add to queue", depends_on=["route"])
        .outputs("page", "queue")
        .build()
    )

Round-trip: ``load_graph_from_dict(g.model_dump(mode="json"))`` reproduces ``g``.
"""

from __future__ import annotations

from typing import Any

from .loader import load_graph_from_dict
from .schema import Graph, GraphNode

__all__ = ["GraphBuilder"]


class GraphBuilder:
    """Fluent builder for a :class:`Graph`. Every method returns ``self``."""

    def __init__(self, name: str, *, description: str | None = None,
                 fail_fast: bool = False, strict_inputs: bool = False) -> None:
        self.name = name
        self.description = description
        self.fail_fast = fail_fast
        self.strict_inputs = strict_inputs
        self._inputs: list[dict[str, Any]] = []
        self._nodes: list[dict[str, Any]] = []
        self._outputs: list[str] = []

    # ── inputs / outputs ────────────────────────────────────────────────────
    def input(self, name: str, *, type: str = "string", required: bool = True,
              description: str | None = None) -> GraphBuilder:
        self._inputs.append(
            {"name": name, "type": type, "required": required, "description": description}
        )
        return self

    def outputs(self, *node_ids: str) -> GraphBuilder:
        self._outputs.extend(node_ids)
        return self

    # ── generic node escape hatch ────────────────────────────────────────────
    def node(self, node: GraphNode | dict[str, Any]) -> GraphBuilder:
        self._nodes.append(node.model_dump(mode="json") if isinstance(node, GraphNode) else dict(node))
        return self

    def _add(self, spec: dict[str, Any]) -> GraphBuilder:
        # Drop None values so YAML/JSON stays clean and loader defaults apply.
        self._nodes.append({k: v for k, v in spec.items() if v is not None})
        return self

    # ── LLM / classic nodes ──────────────────────────────────────────────────
    def base(self, id: str, *, purpose: str | None = None, goal: str | None = None,
             expected_result: str | None = None, depends_on: list[str] | None = None,
             mode: str | None = None, output_schema: str | None = None,
             when: str | None = None, writes: str | None = None,
             on_error: str | None = None, **extra: Any) -> GraphBuilder:
        return self._add({
            "id": id, "kind": "base", "purpose": purpose, "goal": goal,
            "expected_result": expected_result, "depends_on": depends_on or [],
            "mode": mode, "output_schema": output_schema, "when": when,
            "writes": writes, "on_error": on_error, **extra,
        })

    def react(self, id: str, *, tools: list[str], purpose: str | None = None,
              goal: str | None = None, depends_on: list[str] | None = None,
              when: str | None = None, writes: str | None = None,
              on_error: str | None = None, **extra: Any) -> GraphBuilder:
        return self._add({
            "id": id, "kind": "react", "tools": tools, "purpose": purpose,
            "goal": goal, "depends_on": depends_on or [], "when": when,
            "writes": writes, "on_error": on_error, **extra,
        })

    def function(self, id: str, *, callable: str, depends_on: list[str] | None = None,
                 when: str | None = None, writes: str | None = None,
                 on_error: str | None = None, **extra: Any) -> GraphBuilder:
        return self._add({
            "id": id, "kind": "function", "callable": callable,
            "depends_on": depends_on or [], "when": when, "writes": writes,
            "on_error": on_error, **extra,
        })

    def tool(self, id: str, *, tools: list[str], tool_args: dict[str, Any] | None = None,
             depends_on: list[str] | None = None, when: str | None = None,
             writes: str | None = None, on_error: str | None = None, **extra: Any) -> GraphBuilder:
        return self._add({
            "id": id, "kind": "tool", "tools": tools, "tool_args": tool_args,
            "depends_on": depends_on or [], "when": when, "writes": writes,
            "on_error": on_error, **extra,
        })

    # ── control-flow nodes ───────────────────────────────────────────────────
    def router(self, id: str, *, routes: dict[str, str] | None = None,
               cases: list[dict[str, Any]] | None = None, default: str | None = None,
               repair: bool = True, depends_on: list[str] | None = None,
               purpose: str | None = None, goal: str | None = None,
               **extra: Any) -> GraphBuilder:
        """`routes` = the simple classification router (label → target, one LLM
        call instructed by purpose/goal); `cases` = deterministic predicates."""
        return self._add({
            "id": id, "kind": "router", "routes": routes, "cases": cases,
            "default": default, "repair": repair, "depends_on": depends_on or [],
            "purpose": purpose, "goal": goal, **extra,
        })

    def loop(self, id: str, *, body: list[Any], max_iterations: int,
             until: str | None = None, break_when: str | None = None,
             accumulate: str | None = None, depends_on: list[str] | None = None,
             body_outputs: list[str] | None = None, **extra: Any) -> GraphBuilder:
        """`until` = plain-English stop condition (judged each iteration, CONTINUE
        reasons become the next iteration's {feedback}); `break_when` = expression."""
        return self._add({
            "id": id, "kind": "loop", "body": _dump_body(body),
            "max_iterations": max_iterations, "until": until,
            "break_when": break_when, "accumulate": accumulate,
            "depends_on": depends_on or [], "body_outputs": body_outputs or [],
            **extra,
        })

    def map(self, id: str, *, over: str, body: list[Any], as_: str = "item",
            concurrency: int = 1, depends_on: list[str] | None = None,
            body_outputs: list[str] | None = None, **extra: Any) -> GraphBuilder:
        spec = {
            "id": id, "kind": "map", "over": over, "body": _dump_body(body),
            "as": as_, "concurrency": concurrency, "depends_on": depends_on or [],
            "body_outputs": body_outputs or [], **extra,
        }
        return self._add(spec)

    def subgraph(self, id: str, *, body: list[Any], depends_on: list[str] | None = None,
                 body_outputs: list[str] | None = None, **extra: Any) -> GraphBuilder:
        return self._add({
            "id": id, "kind": "subgraph", "body": _dump_body(body),
            "depends_on": depends_on or [], "body_outputs": body_outputs or [], **extra,
        })

    def input_node(self, id: str, *, purpose: str, options: list[str] | None = None,
                   writes: str | None = None, depends_on: list[str] | None = None,
                   **extra: Any) -> GraphBuilder:
        return self._add({
            "id": id, "kind": "input", "purpose": purpose, "options": options or [],
            "writes": writes, "depends_on": depends_on or [], **extra,
        })

    # ── materialize ──────────────────────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "fail_fast": self.fail_fast,
            "strict_inputs": self.strict_inputs,
            "inputs": self._inputs,
            "nodes": self._nodes,
            "outputs": self._outputs,
        }

    def build(self) -> Graph:
        """Validate and return the :class:`Graph` (same path as the YAML loader)."""
        return load_graph_from_dict(self.to_dict())


def _dump_body(body: list[Any]) -> list[dict[str, Any]]:
    """Normalize a body list (GraphNode | dict | GraphBuilder) into node dicts."""
    out: list[dict[str, Any]] = []
    for item in body:
        if isinstance(item, GraphNode):
            out.append(item.model_dump(mode="json"))
        elif isinstance(item, GraphBuilder):
            out.extend(item._nodes)
        else:
            out.append(dict(item))
    return out
