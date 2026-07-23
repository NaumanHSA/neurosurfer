"""Typed shared workflow state (Phase 1a).

The legacy executor passed node outputs around implicitly: each node received
``{dep_id: raw_output}`` for its declared ``depends_on``. That is enough for a
run-once DAG, but conditional edges, routers, and loops need to *read* arbitrary
prior results and explicit variables to make control-flow decisions.

:class:`WorkflowState` is the single object threaded through execution. It holds:

- ``inputs`` — the validated graph inputs.
- ``nodes``  — every completed node's ``raw_output``, keyed by node id.
- ``vars``   — explicit named variables set by constructs (loop accumulators,
  router decisions, map results, or a node writing to a declared ``writes`` key).

It exposes :meth:`namespace` for the expression evaluator (``inputs`` / ``nodes`` /
``vars`` / ``state``) and :meth:`snapshot` for JSON-serializable tracing / the UI.

Backward compatibility: nothing forces a node to declare what it reads or writes.
``nodes`` is populated automatically from results, so existing graphs that relied on
implicit ``depends_on`` output-passing keep working — the executor still builds the
per-node ``dependency_results`` from this state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

__all__ = ["WorkflowState"]


def _jsonable(value: Any) -> Any:
    """Best-effort convert *value* into something JSON-serializable for snapshots."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    # Pydantic models
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return _jsonable(dump())
        except Exception:  # noqa: BLE001
            pass
    # Fall back to a truncated repr so a snapshot never explodes or fails.
    text = repr(value)
    return text if len(text) <= 2000 else text[:2000] + "…"


@dataclass
class WorkflowState:
    """Live, mutable state for one workflow execution.

    Parameters
    ----------
    inputs:
        Validated graph inputs (produced by ``normalize_and_validate_graph_inputs``).
    nodes:
        node_id → raw_output for every completed node. Auto-populated by the executor.
    vars:
        Explicit named variables set by control-flow constructs or nodes' ``writes``.
    scope:
        Iteration-local overlay (loop index / map item / accumulator). Reads fall
        through to the parent state; writes stay local unless promoted. Used by
        loop and map constructs so per-iteration values don't leak globally.
    """

    inputs: dict[str, Any] = field(default_factory=dict)
    nodes: dict[str, Any] = field(default_factory=dict)
    vars: dict[str, Any] = field(default_factory=dict)
    scope: dict[str, Any] = field(default_factory=dict)

    # ── writes ────────────────────────────────────────────────────────────────
    def set_node_output(self, node_id: str, output: Any) -> None:
        """Record a completed node's raw output under ``nodes[node_id]``."""
        self.nodes[node_id] = output

    def set_var(self, name: str, value: Any) -> None:
        """Set an explicit variable readable as ``vars.<name>`` in expressions."""
        self.vars[name] = value

    def update_vars(self, values: dict[str, Any]) -> None:
        for k, v in values.items():
            self.vars[k] = v

    # ── reads ─────────────────────────────────────────────────────────────────
    def get_node_output(self, node_id: str, default: Any = None) -> Any:
        return self.nodes.get(node_id, default)

    def namespace(self) -> dict[str, Any]:
        """Return the root namespace passed to the expression evaluator.

        Names available in predicates: ``inputs``, ``nodes``, ``vars``, plus any
        iteration ``scope`` keys hoisted to the top level (``index``, ``item`` …),
        and ``state`` for the whole thing.
        """
        ns: dict[str, Any] = {
            "inputs": self.inputs,
            "nodes": self.nodes,
            "vars": self.vars,
        }
        # Hoist iteration-scope keys so predicates can say `index < 3` or
        # `item.status == "ok"` directly inside a loop/map body.
        for k, v in self.scope.items():
            ns.setdefault(k, v)
        ns["state"] = ns
        return ns

    # ── scoping (loops / maps) ────────────────────────────────────────────────
    def child_scope(self, overlay: dict[str, Any]) -> WorkflowState:
        """Return a shallow child state sharing inputs/nodes/vars with a new scope.

        The child shares the *same* ``inputs``/``nodes``/``vars`` dicts (so a loop
        body's node outputs and variable writes are visible to the parent and to
        later iterations), but gets its own ``scope`` overlay for iteration-local
        values like the loop index or current map item.
        """
        merged = {**self.scope, **overlay}
        return WorkflowState(
            inputs=self.inputs,
            nodes=self.nodes,
            vars=self.vars,
            scope=merged,
        )

    # ── serialization ─────────────────────────────────────────────────────────
    def snapshot(self) -> dict[str, Any]:
        """A JSON-serializable snapshot of the state for tracing / the UI."""
        return {
            "inputs": _jsonable(self.inputs),
            "nodes": _jsonable(self.nodes),
            "vars": _jsonable(self.vars),
            "scope": _jsonable(self.scope),
        }

    def to_json(self) -> str:
        return json.dumps(self.snapshot(), ensure_ascii=False)
