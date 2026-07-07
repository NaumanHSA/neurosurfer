"""Phase 5: a graph workflow renders as one nested trace.

Hierarchy exercised here::

    workflow:<name>          (root span, opened by WorkflowRunner/executor)
    └── node:<id>            (one span per graph node)
        └── <Agent>.run      (the node's agent, if it is an LLM node)
            └── tool:<name>  (the agent's tool calls)

The root + node spans come from ``traced_run``; each node's agent flows through
``base._tap`` and nests under its node span — across threads too (parallel /
timeout nodes hop to worker threads that inherit a copied contextvar).
"""

from __future__ import annotations

from pathlib import Path

import yaml

from neurosurfer.graph.engine import GraphExecutor, load_graph_from_dict
from neurosurfer.graph.workflow.package import load_package
from neurosurfer.graph.workflow.runner import WorkflowRunner
from neurosurfer.observability.exporters import (
    MemoryExporter,
    register_exporter,
    reset_exporters,
)
from neurosurfer.observability.run import traced_run
from neurosurfer.tools.base import ToolPool

from .fakes import ScriptedProvider


def _base_node_pkg(pkg_dir: Path) -> None:
    """A one-node workflow whose single node is a base (LLM) node."""
    pkg_dir.mkdir(parents=True, exist_ok=True)
    (pkg_dir / "workflow.yaml").write_text(
        yaml.dump({"name": pkg_dir.name, "version": "0.1.0", "entrypoint": "graph.yaml"}),
        encoding="utf-8",
    )
    graph = {
        "name": pkg_dir.name,
        "nodes": [{"id": "n", "kind": "base", "purpose": "answer"}],
        "outputs": ["n"],
    }
    (pkg_dir / "graph.yaml").write_text(yaml.dump(graph), encoding="utf-8")


def _by_name(starts, prefix):
    return [s for s in starts if s["name"].startswith(prefix)]


def _one(starts, prefix):
    matches = _by_name(starts, prefix)
    assert len(matches) == 1, f"expected one {prefix!r} span, got {len(matches)}"
    return matches[0]


def test_workflow_node_agent_hierarchy(tmp_path):
    """workflow → node → agent: three nested spans, one trace."""
    reset_exporters()
    mem = MemoryExporter()
    register_exporter(mem)
    try:
        pkg_dir = tmp_path / "wf"
        _base_node_pkg(pkg_dir)
        pkg = load_package(pkg_dir)
        WorkflowRunner(ScriptedProvider([("the answer", [])])).run(pkg, {"query": "hi"})
    finally:
        reset_exporters()

    starts = mem.of("run_start")
    wf = _one(starts, "workflow:")
    node = _one(starts, "node:")
    agent = _one(starts, "Agent")  # OneShotAgent → "Agent.run"

    # workflow is the root; node nests under workflow; agent nests under node.
    assert wf["parent_span_id"] is None
    assert node["parent_span_id"] == wf["span_id"]
    assert agent["parent_span_id"] == node["span_id"]
    # all share one trace
    assert node["trace_id"] == wf["trace_id"] == agent["trace_id"]
    # every span opened is closed
    assert len(mem.of("run_finish")) == len(starts) == 3


def test_function_node_is_visible(tmp_path):
    """A non-agent function node leaves no agent run, but still gets its own node
    span — so it is visible in the trace (the whole point of the node layer)."""
    reset_exporters()
    mem = MemoryExporter()
    register_exporter(mem)
    try:
        graph = load_graph_from_dict(
            {
                "name": "fn",
                "nodes": [{"id": "f", "kind": "function", "callable": "os:getcwd"}],
                "outputs": ["f"],
            }
        )
        executor = GraphExecutor(graph, provider=ScriptedProvider([]), native_tools=ToolPool([]))
        with traced_run("workflow:fn", metadata={"kind": "workflow"}):
            executor.run({"query": "hi"})
    finally:
        reset_exporters()

    starts = mem.of("run_start")
    wf = _one(starts, "workflow:")
    node = _one(starts, "node:")
    assert node["parent_span_id"] == wf["span_id"]
    # No agent ran — just workflow + node.
    assert len(starts) == 2


def test_failing_node_span_marked_error(tmp_path):
    """A node that returns an error (not raises) marks its span errored."""
    reset_exporters()
    mem = MemoryExporter()
    register_exporter(mem)
    try:
        graph = load_graph_from_dict(
            {
                "name": "boom",
                # function node whose callable blows up → NodeExecutionResult.error set
                "nodes": [{"id": "x", "kind": "function", "callable": "os:nonexistent_fn"}],
                "outputs": ["x"],
            }
        )
        executor = GraphExecutor(graph, provider=ScriptedProvider([]), native_tools=ToolPool([]))
        with traced_run("workflow:boom", metadata={"kind": "workflow"}):
            executor.run({"query": "hi"})
    finally:
        reset_exporters()

    # The node span recorded an error and finished with status "error".
    assert mem.of("error"), "expected on_error to fire for the failing node"
    node_finish = next(f for f in mem.of("run_finish") if f["status"] == "error")
    assert node_finish is not None


def test_parallel_nodes_nest_across_threads():
    """Parallel nodes run on ThreadPoolExecutor workers; a copied contextvar carries
    the ambient TraceContext across the thread boundary so each node span (and its
    agent) still nests under the workflow."""
    reset_exporters()
    mem = MemoryExporter()
    register_exporter(mem)
    try:
        graph = load_graph_from_dict(
            {
                "name": "par",
                "nodes": [
                    {"id": "a", "kind": "base", "purpose": "a"},
                    {"id": "b", "kind": "base", "purpose": "b"},
                ],
                "outputs": ["a", "b"],
            }
        )
        executor = GraphExecutor(
            graph,
            provider=ScriptedProvider([("ra", []), ("rb", [])]),
            native_tools=ToolPool([]),
            parallelism=2,
        )
        with traced_run("workflow:par", metadata={"kind": "workflow"}):
            executor.run({"query": "hi"})
    finally:
        reset_exporters()

    starts = mem.of("run_start")
    wf = _one(starts, "workflow:")
    node_spans = _by_name(starts, "node:")
    agent_spans = _by_name(starts, "Agent")
    assert len(node_spans) == 2 and len(agent_spans) == 2
    # both node spans nest under the workflow...
    for n in node_spans:
        assert n["parent_span_id"] == wf["span_id"]
        assert n["trace_id"] == wf["trace_id"]
    # ...and each agent nests under one of the node spans (not the workflow directly)
    node_ids = {n["span_id"] for n in node_spans}
    for a in agent_spans:
        assert a["parent_span_id"] in node_ids


def test_timeout_node_nests_across_thread():
    """A node with a timeout runs on a worker thread; its agent must still nest
    under the node span."""
    reset_exporters()
    mem = MemoryExporter()
    register_exporter(mem)
    try:
        graph = load_graph_from_dict(
            {
                "name": "to",
                "nodes": [
                    {"id": "n", "kind": "base", "purpose": "n", "policy": {"timeout_s": 30}}
                ],
                "outputs": ["n"],
            }
        )
        executor = GraphExecutor(
            graph, provider=ScriptedProvider([("ok", [])]), native_tools=ToolPool([])
        )
        with traced_run("workflow:to", metadata={"kind": "workflow"}):
            executor.run({"query": "hi"})
    finally:
        reset_exporters()

    starts = mem.of("run_start")
    wf = _one(starts, "workflow:")
    node = _one(starts, "node:")
    agent = _one(starts, "Agent")
    assert node["parent_span_id"] == wf["span_id"]
    assert agent["parent_span_id"] == node["span_id"]


def test_no_workflow_span_when_no_exporter(tmp_path):
    """Zero overhead: with no exporter active, ``traced_run`` is a pure no-op."""
    reset_exporters()
    mem = MemoryExporter()  # built but never registered → not active
    try:
        pkg_dir = tmp_path / "wf2"
        _base_node_pkg(pkg_dir)
        pkg = load_package(pkg_dir)
        WorkflowRunner(ScriptedProvider([("x", [])])).run(pkg, {"query": "hi"})
    finally:
        reset_exporters()
    assert mem.calls == []
