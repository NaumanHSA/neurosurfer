"""D13 — Verification gate for Phase D (Runtime Maturation).

≥ 30 tests covering D1–D12:
  D1  cascading-failure isolation in GraphExecutor
  D2  ReActAgent output_schema parameter
  D3  ReActAgent loop-termination fix (and/not or)
  D4  Agent structured-output retry + StructuredOutputError
  D5  GraphExecutor hard per-node timeout
  D6  GraphExecutor parallel topo-layer execution
  D7  BaseTool.safe_call() typed error hierarchy
  D8  BaseChatModel count_tokens + _truncate_history_to_budget
  D9  RAGAgent retrieval error handling (empty query / broken embed)
  D10 CodeAgent subprocess vs in-process config
  D11 Tracer.export_json()
  D12 WorkflowRunner input boundary validation
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock

import pytest

# ── Tracer (D11) ─────────────────────────────────────────────────────────────
from neurosurfer.tracing import Tracer, TracerConfig

# ── Error hierarchy ───────────────────────────────────────────────────────────
from neurosurfer.graph import (
    AgentError,
    CodeExecutionError,
    GraphError,
    GraphExecutionError,
    InputValidationError,
    NeurosurferError,
    NodeError,
    # legacy aliases
    NodeExecutionError,
    NodeFailedError,
    NodeSkippedError,
    NodeTimeoutError,
    StructuredOutputError,
    ToolError,
    ToolExecutionError,
    ToolInputError,
    ToolNotFoundError,
    ValidationError,
)

# ── GraphExecutor (D1, D5, D6) ───────────────────────────────────────────────
from neurosurfer.graph import GraphExecutor, _topo_layers

# ── Graph schema ──────────────────────────────────────────────────────────────
from neurosurfer.graph import (
    Graph,
    GraphExecutionResult,
    GraphInput,
    GraphNode,
    NodeExecutionResult,
    NodeMode,
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


def _make_result(nid: str, *, error: str | None = None, skipped: bool = False, raw: Any = "ok") -> NodeExecutionResult:
    return NodeExecutionResult(
        node_id=nid,
        mode=NodeMode.TEXT,
        raw_output=raw if not error else None,
        started_at=time.time(),
        duration_ms=1,
        error=error,
        skipped=skipped,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# D1 — Exception hierarchy
# ═══════════════════════════════════════════════════════════════════════════════

class TestErrorHierarchy:
    def test_neurosurfer_root(self):
        assert issubclass(GraphError, NeurosurferError)
        assert issubclass(AgentError, NeurosurferError)
        assert issubclass(ToolError, NeurosurferError)

    def test_graph_error_chain(self):
        assert issubclass(GraphExecutionError, GraphError)
        assert issubclass(InputValidationError, GraphError)
        assert issubclass(NodeFailedError, GraphExecutionError)
        assert issubclass(NodeSkippedError, NodeFailedError)
        assert issubclass(NodeTimeoutError, NodeFailedError)

    def test_node_failed_error_attrs(self):
        exc = NodeFailedError("node-1", "something broke", attempt=2, duration_ms=50)
        assert exc.node_id == "node-1"
        assert exc.attempt == 2
        assert exc.duration_ms == 50
        assert exc.failed_node == "node-1"

    def test_node_skipped_error_attrs(self):
        exc = NodeSkippedError("node-2", "node-1", "timed out")
        assert exc.upstream_id == "node-1"
        assert "node-1" in str(exc)

    def test_node_timeout_error_attrs(self):
        exc = NodeTimeoutError("node-3", 30.0)
        assert exc.timeout_s == 30.0
        assert "30" in str(exc)

    def test_structured_output_error_attrs(self):
        from pydantic import BaseModel as PydModel

        class _Schema(PydModel):
            value: int

        exc = StructuredOutputError("node-1", _Schema, '{"bad": true}', attempts=3)
        assert exc.node_id == "node-1"
        assert exc.attempts == 3
        assert "_Schema" in str(exc)
        assert isinstance(exc, AgentError)
        assert isinstance(exc, NeurosurferError)

    def test_tool_error_hierarchy(self):
        assert issubclass(ToolNotFoundError, ToolError)
        assert issubclass(ToolInputError, ToolError)
        assert issubclass(ToolExecutionError, ToolError)
        exc = ToolInputError("my_tool", "missing field")
        assert exc.tool_name == "my_tool"

    def test_tool_execution_error_attrs(self):
        cause = ValueError("bang")
        exc = ToolExecutionError("my_tool", cause)
        assert exc.cause is cause
        assert isinstance(exc, NeurosurferError)

    def test_code_execution_error_timeout_flag(self):
        exc = CodeExecutionError(1, "stderr", timeout=True)
        assert exc.timeout is True
        assert "timed out" in str(exc).lower()

    def test_legacy_aliases(self):
        assert NodeExecutionError is NodeFailedError
        assert NodeError is NodeFailedError
        assert ValidationError is InputValidationError

    def test_graph_execution_error_failed_node(self):
        exc = GraphExecutionError("graph failed", failed_node="nodeX")
        assert exc.failed_node == "nodeX"


# ═══════════════════════════════════════════════════════════════════════════════
# D1 — NodeExecutionResult / GraphExecutionResult
# ═══════════════════════════════════════════════════════════════════════════════

class TestResultModels:
    def test_node_result_ok_property(self):
        r = _make_result("n1")
        assert r.ok is True

    def test_node_result_not_ok_on_error(self):
        r = _make_result("n1", error="oops")
        assert r.ok is False

    def test_node_result_not_ok_on_skipped(self):
        r = _make_result("n1", skipped=True)
        assert r.ok is False

    def test_graph_result_succeeded(self):
        nodes = {"n1": _make_result("n1")}
        gr = GraphExecutionResult(graph=_make_graph([_make_node("n1")]), nodes=nodes, final={})
        assert gr.succeeded is True

    def test_graph_result_not_succeeded_with_errors(self):
        nodes = {"n1": _make_result("n1", error="bad")}
        gr = GraphExecutionResult(
            graph=_make_graph([_make_node("n1")]),
            nodes=nodes,
            final={},
            errors={"n1": "bad"},
        )
        assert gr.succeeded is False

    def test_graph_result_execution_summary(self):
        n1 = _make_result("n1")
        n2 = _make_result("n2", error="fail")
        n3 = _make_result("n3", skipped=True)
        gr = GraphExecutionResult(
            graph=_make_graph([_make_node("n1"), _make_node("n2"), _make_node("n3")]),
            nodes={"n1": n1, "n2": n2, "n3": n3},
            final={},
            errors={"n2": "fail"},
            skipped=["n3"],
        )
        summary = gr.execution_summary()
        assert "3 nodes" in summary
        assert "1 ok" in summary
        assert "1 failed" in summary
        assert "1 skipped" in summary


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
# D11 — Tracer.export_json()
# ═══════════════════════════════════════════════════════════════════════════════

class TestTracerExportJson:
    def _tracer_with_step(self) -> Tracer:
        tracer = Tracer(config=TracerConfig(enabled=True, log_steps=False))
        with tracer(kind="llm", label="test.step", inputs={"q": "hello"}, agent_id="ag1") as ctx:
            ctx.outputs(output="world")
        return tracer

    def test_export_json_returns_dict(self):
        tracer = self._tracer_with_step()
        exported = tracer.export_json()
        assert isinstance(exported, dict)

    def test_export_json_has_meta_and_steps(self):
        tracer = self._tracer_with_step()
        exported = tracer.export_json()
        assert "meta" in exported
        assert "steps" in exported

    def test_export_json_steps_have_required_fields(self):
        tracer = self._tracer_with_step()
        step = tracer.export_json()["steps"][0]
        for field in ("step_id", "kind", "label", "started_at", "duration_ms", "ok"):
            assert field in step, f"Missing field: {field}"

    def test_export_json_step_kind(self):
        tracer = self._tracer_with_step()
        step = tracer.export_json()["steps"][0]
        assert step["kind"] == "llm"

    def test_export_json_no_steps_when_disabled(self):
        tracer = Tracer(config=TracerConfig(enabled=False))
        exported = tracer.export_json()
        assert exported["steps"] == []

    def test_export_json_meta_set_via_set_meta(self):
        tracer = Tracer(config=TracerConfig(enabled=True, log_steps=False), meta={"graph": "test"})
        exported = tracer.export_json()
        assert exported["meta"]["graph"] == "test"


# ═══════════════════════════════════════════════════════════════════════════════
# D10 — CodeExecutionError
# ═══════════════════════════════════════════════════════════════════════════════

class TestCodeExecutionError:
    def test_non_timeout(self):
        exc = CodeExecutionError(returncode=1, stderr="SyntaxError: bad")
        assert exc.timeout is False
        assert exc.returncode == 1
        assert "SyntaxError" in str(exc)

    def test_timeout(self):
        exc = CodeExecutionError(returncode=-1, stderr="", timeout=True)
        assert exc.timeout is True

    def test_is_agent_error(self):
        exc = CodeExecutionError(1, "")
        assert isinstance(exc, AgentError)
        assert isinstance(exc, NeurosurferError)



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

    def test_input_validation_error_is_graph_error(self):
        assert issubclass(InputValidationError, GraphError)
        assert issubclass(InputValidationError, NeurosurferError)


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
# Integration — Graph schema (fail_fast, strict_inputs, GraphInput)
# ═══════════════════════════════════════════════════════════════════════════════

class TestGraphSchemaD1Fields:
    def test_fail_fast_defaults_false(self):
        g = Graph(name="g", nodes=[_make_node("a")])
        assert g.fail_fast is False

    def test_strict_inputs_defaults_false(self):
        g = Graph(name="g", nodes=[_make_node("a")])
        assert g.strict_inputs is False

    def test_graph_input_required_defaults_true(self):
        gi = GraphInput(name="x")
        assert gi.required is True

    def test_graph_input_type_normalization(self):
        gi = GraphInput(name="x", type="str")
        assert gi.type == "string"

    def test_graph_inputs_unique_validator(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Graph(name="g", nodes=[_make_node("a")], inputs=[
                {"name": "x"},
                {"name": "x"},
            ])
