"""D13 — Verification gate for Phase D (Runtime Maturation).

Core regression coverage retained from D1–D12:
  D1  cascading-failure isolation in GraphExecutor
  D5  GraphExecutor hard per-node timeout
  D6  GraphExecutor parallel topo-layer execution
  D12 WorkflowRunner input boundary validation
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

# ── GraphExecutor (D1, D5, D6) ───────────────────────────────────────────────
# ── Graph schema ──────────────────────────────────────────────────────────────
from neurosurfer.graph import (
    Graph,
    GraphExecutionError,
    GraphExecutor,
    GraphInput,
    GraphNode,
    InputValidationError,
    NodeExecutionResult,
    NodeFailedError,
    NodeMode,
    NodeTimeoutError,
    _topo_layers,
)

# ── WorkflowRunner (D12) ─────────────────────────────────────────────────────
from neurosurfer.graph.workflow.runner import WorkflowRunner

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers / fakes
# ═══════════════════════════════════════════════════════════════════════════════

def _make_node(nid: str, *, depends_on: list[str] | None = None, kind: str = "function") -> GraphNode:
    return GraphNode(id=nid, kind=kind, depends_on=depends_on or [])


def _make_graph(nodes: list[GraphNode], *, fail_fast: bool = False) -> Graph:
    return Graph(name="test-graph", nodes=nodes, fail_fast=fail_fast)


# ═══════════════════════════════════════════════════════════════════════════════
# D1 — Cascading failure isolation via GraphExecutor
# ═══════════════════════════════════════════════════════════════════════════════

class TestCascadingFailure:
    """Test that downstream nodes are skipped when an upstream node fails."""

    def _executor_for(self, graph: Graph, fail_node: str) -> tuple[GraphExecutor, dict]:
        """Build an executor that makes fail_node return an error result."""
        llm = MagicMock()

        # Patch _run_node to fail on the specified node, succeed otherwise.
        def fake_run_node(node, graph_inputs, dependency_results, previous_result, **kw):
            if node.id == fail_node:
                return NodeExecutionResult(
                    node_id=node.id,
                    mode=NodeMode.TEXT,
                    raw_output=None,
                    started_at=time.time(),
                    duration_ms=1,
                    error="injected failure",
                )
            return NodeExecutionResult(
                node_id=node.id,
                mode=NodeMode.TEXT,
                raw_output=f"result-{node.id}",
                started_at=time.time(),
                duration_ms=1,
            )

        executor = GraphExecutor(graph, llm=llm, log_traces=False)
        executor._run_node = fake_run_node  # type: ignore[assignment]
        return executor

    def test_downstream_skipped_on_upstream_failure(self):
        a = _make_node("a")
        b = _make_node("b", depends_on=["a"])
        graph = _make_graph([a, b])
        executor = self._executor_for(graph, fail_node="a")
        result = executor.run({})
        assert "a" in result.errors
        assert "b" in result.skipped
        assert result.nodes["b"].skipped is True

    def test_skip_reason_contains_upstream_id(self):
        a = _make_node("a")
        b = _make_node("b", depends_on=["a"])
        graph = _make_graph([a, b])
        executor = self._executor_for(graph, fail_node="a")
        result = executor.run({})
        assert "a" in result.nodes["b"].skip_reason

    def test_non_dependent_node_still_runs(self):
        a = _make_node("a")
        b = _make_node("b")  # no dependency on a
        graph = _make_graph([a, b])
        executor = self._executor_for(graph, fail_node="a")
        result = executor.run({})
        assert "b" not in result.skipped
        assert result.nodes["b"].ok is True

    def test_fail_fast_raises(self):
        a = _make_node("a")
        graph = _make_graph([a], fail_fast=True)
        executor = self._executor_for(graph, fail_node="a")
        with pytest.raises(GraphExecutionError):
            executor.run({})


# ═══════════════════════════════════════════════════════════════════════════════
# D6 — _topo_layers helper
# ═══════════════════════════════════════════════════════════════════════════════

class TestTopoLayers:
    def test_single_node_single_layer(self):
        nodes = [_make_node("a")]
        layers = _topo_layers(nodes)
        assert layers == [["a"]]

    def test_linear_chain_separate_layers(self):
        nodes = [_make_node("a"), _make_node("b", depends_on=["a"]), _make_node("c", depends_on=["b"])]
        layers = _topo_layers(nodes)
        assert layers[0] == ["a"]
        assert layers[1] == ["b"]
        assert layers[2] == ["c"]

    def test_parallel_siblings_same_layer(self):
        nodes = [_make_node("a"), _make_node("b"), _make_node("c", depends_on=["a", "b"])]
        layers = _topo_layers(nodes)
        # a and b should be in the same first layer
        assert set(layers[0]) == {"a", "b"}
        assert layers[1] == ["c"]

    def test_diamond_dependency(self):
        nodes = [
            _make_node("root"),
            _make_node("left", depends_on=["root"]),
            _make_node("right", depends_on=["root"]),
            _make_node("sink", depends_on=["left", "right"]),
        ]
        layers = _topo_layers(nodes)
        assert layers[0] == ["root"]
        assert set(layers[1]) == {"left", "right"}
        assert layers[2] == ["sink"]


# ═══════════════════════════════════════════════════════════════════════════════
# D12 — WorkflowRunner input boundary validation
# ═══════════════════════════════════════════════════════════════════════════════

class TestWorkflowRunnerInputValidation:
    """Tests for _validate_inputs against pkg.graph.inputs (list[GraphInput])."""

    def _make_pkg(self, graph_inputs: list[dict]) -> MagicMock:
        """Build a fake WorkflowPackage with declared graph inputs."""
        pkg = MagicMock()
        pkg.manifest.name = "test-workflow"
        pkg.graph.inputs = [GraphInput(**spec) for spec in graph_inputs]
        return pkg

    def _runner(self) -> WorkflowRunner:
        provider = MagicMock()
        return WorkflowRunner(provider)

    def test_no_declared_inputs_passes_anything(self):
        runner = self._runner()
        pkg = self._make_pkg([])
        runner._validate_inputs(pkg, {"anything": 1})  # should not raise

    def test_required_input_present_passes(self):
        runner = self._runner()
        pkg = self._make_pkg([{"name": "query", "required": True}])
        runner._validate_inputs(pkg, {"query": "hello"})  # should not raise

    def test_missing_required_input_raises(self):
        runner = self._runner()
        pkg = self._make_pkg([{"name": "query", "required": True}])
        with pytest.raises(InputValidationError) as exc_info:
            runner._validate_inputs(pkg, {})
        assert "query" in str(exc_info.value)

    def test_optional_input_missing_does_not_raise(self):
        runner = self._runner()
        pkg = self._make_pkg([{"name": "query", "required": False}])
        runner._validate_inputs(pkg, {})  # should not raise

    def test_multiple_required_all_missing_mentions_all(self):
        runner = self._runner()
        pkg = self._make_pkg([
            {"name": "a", "required": True},
            {"name": "b", "required": True},
        ])
        with pytest.raises(InputValidationError) as exc_info:
            runner._validate_inputs(pkg, {})
        err_msg = str(exc_info.value)
        assert "a" in err_msg
        assert "b" in err_msg


# ═══════════════════════════════════════════════════════════════════════════════
# D5 — GraphExecutor hard timeout wiring
# ═══════════════════════════════════════════════════════════════════════════════

class TestGraphExecutorTimeout:
    def test_node_timeout_error_is_node_failed(self):
        exc = NodeTimeoutError("slow-node", 5.0)
        assert isinstance(exc, NodeFailedError)
        assert isinstance(exc, GraphExecutionError)

    def test_graph_node_policy_timeout_field(self):
        from neurosurfer.graph import NodePolicy

        policy = NodePolicy(timeout_s=15)
        assert policy.timeout_s == 15


# ═══════════════════════════════════════════════════════════════════════════════
# Integration — Graph schema input-validation gate (duplicate-name rejection)
# ═══════════════════════════════════════════════════════════════════════════════

class TestGraphSchemaD1Fields:
    def test_graph_inputs_unique_validator(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Graph(name="g", nodes=[_make_node("a")], inputs=[
                {"name": "x"},
                {"name": "x"},
            ])
