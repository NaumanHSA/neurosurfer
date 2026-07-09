"""Phase B: WorkflowPackage — multi-file package loader, registry, runner.

Covers:
1. load_package: reads workflow.yaml + graph.yaml, merges agents/ overrides.
2. WorkflowRegistry: list / get / save / delete.
3. GraphExecutor with function-kind nodes (no LLM).
4. GraphExecutor with tool-kind nodes via native ToolPool.
5. End-to-end 3-node package: agent + function + tool (using echo provider).
6. R3+R4: native provider + ToolPool drive GraphExecutor directly (no adapter layer).
7. WorkflowRunner writes a structured JSON trace when trace_path is set.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from neurosurfer.graph import (
    Graph,
    GraphExecutor,
    GraphNode,
    NodeMode,
)
from neurosurfer.graph.workflow.package import (
    PackageLoadError,
    _PackagePathContext,
    load_package,
    save_package,
)
from neurosurfer.graph.workflow.registry import WorkflowNotFoundError, WorkflowRegistry
from neurosurfer.graph.workflow.runner import WorkflowRunner
from neurosurfer.graph.workflow.schema import WorkflowManifest
from neurosurfer.llm.base import Provider
from neurosurfer.llm.capabilities import ProviderCapabilities
from neurosurfer.llm.types import CanonicalResponse, Done, TextBlock, Usage
from neurosurfer.tools.base import ToolContext, ToolPool
from neurosurfer.tools.registry import all_tools

# ── fixtures ──────────────────────────────────────────────────────────────────

class _EchoProvider(Provider):
    model = "echo-model"
    capabilities = ProviderCapabilities(
        context_window=8192,
        max_output_tokens=2048,
        supports_thinking=False,
        supports_prompt_cache=False,
        supports_token_count=False,
        tool_call_style="openai",
    )

    async def stream(self, messages, system, tools, config):
        last = messages[-1].text() if messages else ""
        yield Done(
            response=CanonicalResponse(
                content=[TextBlock(text=f"REPLY[{last[:60]}]")],
                stop_reason="stop",
                usage=Usage(),
            )
        )

    async def count_tokens(self, messages, system, tools):
        return 0


class _StubIO:
    async def ask(self, question, options=None):
        return ""

    async def request_plan_approval(self, plan):
        return (True, "")

    async def request_shell_approval(self, command, reason):
        return True

    async def request_write_approval(self, path, summary):
        return "deny"

    def notify(self, message):
        return None


class _DummyProvider:
    """No LLM nodes are exercised here, so the provider is never called."""

    model = "dummy"
    capabilities = type("Caps", (), {"context_window": 8192})()


def _function_only_pkg(pkg_dir: Path) -> None:
    pkg_dir.mkdir(parents=True, exist_ok=True)
    (pkg_dir / "workflow.yaml").write_text(
        yaml.dump({"name": pkg_dir.name, "version": "0.1.0", "entrypoint": "graph.yaml"}),
        encoding="utf-8",
    )
    graph = {
        "name": pkg_dir.name,
        "nodes": [{"id": "n", "kind": "function", "callable": "os:getcwd"}],
        "outputs": ["n"],
    }
    (pkg_dir / "graph.yaml").write_text(yaml.dump(graph), encoding="utf-8")


def _make_pkg_dir(tmp_path: Path, *, extra_nodes: list[dict] | None = None) -> Path:
    """Write a minimal valid package directory for testing."""
    pkg_dir = tmp_path / "test_workflow"
    pkg_dir.mkdir(parents=True)

    manifest = {
        "name": "test_workflow",
        "version": "1.0.0",
        "description": "A test workflow",
        "entrypoint": "graph.yaml",
    }
    (pkg_dir / "workflow.yaml").write_text(yaml.dump(manifest))

    nodes = [
        {
            "id": "research",
            "kind": "base",
            "purpose": "Research {topic}",
        },
    ]
    if extra_nodes:
        nodes.extend(extra_nodes)

    graph: dict = {
        "name": "test_workflow",
        "inputs": [{"name": "topic", "type": "string"}],
        "nodes": nodes,
        "outputs": [nodes[-1]["id"]],
    }
    (pkg_dir / "graph.yaml").write_text(yaml.dump(graph))
    return pkg_dir


# ── WorkflowManifest ──────────────────────────────────────────────────────────

def test_manifest_round_trip():
    d = {
        "name": "my_workflow",
        "version": "2.0",
        "description": "does things",
        "tags": ["nlp"],
    }
    m = WorkflowManifest.from_dict(d)
    assert m.name == "my_workflow"
    assert m.version == "2.0"
    assert m.tags == ["nlp"]
    back = m.to_dict()
    assert back["name"] == "my_workflow"
    assert back["tags"] == ["nlp"]


# ── load_package ──────────────────────────────────────────────────────────────

def test_load_package_basic(tmp_path):
    pkg_dir = _make_pkg_dir(tmp_path)
    pkg = load_package(pkg_dir)

    assert pkg.name == "test_workflow"
    assert pkg.version == "1.0.0"
    assert len(pkg.graph.nodes) == 1
    assert pkg.graph.nodes[0].id == "research"
    assert pkg.path == pkg_dir


def test_load_package_missing_manifest(tmp_path):
    pkg_dir = tmp_path / "bad"
    pkg_dir.mkdir()
    (pkg_dir / "graph.yaml").write_text("{name: x, nodes: []}")
    with pytest.raises(PackageLoadError, match="workflow.yaml"):
        load_package(pkg_dir)


def test_load_package_missing_dir(tmp_path):
    with pytest.raises(PackageLoadError, match="not found"):
        load_package(tmp_path / "nonexistent")


def test_load_package_merges_agent_overrides(tmp_path):
    pkg_dir = _make_pkg_dir(tmp_path)

    agents_dir = pkg_dir / "agents"
    agents_dir.mkdir()
    override = {"id": "research", "purpose": "OVERRIDDEN PURPOSE"}
    (agents_dir / "research.yaml").write_text(yaml.dump(override))

    pkg = load_package(pkg_dir)
    assert pkg.graph.nodes[0].purpose == "OVERRIDDEN PURPOSE"


def test_load_package_function_node(tmp_path):
    extra = [{"id": "compute", "kind": "function", "callable": "operator.add", "depends_on": ["research"]}]
    pkg_dir = _make_pkg_dir(tmp_path, extra_nodes=extra)
    pkg = load_package(pkg_dir)
    fn_node = next(n for n in pkg.graph.nodes if n.kind == "function")
    assert fn_node.callable == "operator.add"


def test_load_package_tool_node(tmp_path):
    extra = [{"id": "scan", "kind": "tool", "tools": ["list_dir"], "depends_on": ["research"]}]
    pkg_dir = _make_pkg_dir(tmp_path, extra_nodes=extra)
    pkg = load_package(pkg_dir)
    tool_node = next(n for n in pkg.graph.nodes if n.kind == "tool")
    assert tool_node.tools == ["list_dir"]


# ── save_package / _PackagePathContext ────────────────────────────────────────

def test_save_package_copies_directory(tmp_path):
    src_dir = _make_pkg_dir(tmp_path)
    pkg = load_package(src_dir)

    dst = tmp_path / "copy"
    result = save_package(pkg, dst)

    assert result == dst
    assert (dst / "workflow.yaml").exists()
    assert (dst / "graph.yaml").exists()


def test_save_package_noop_when_same_path(tmp_path):
    pkg_dir = _make_pkg_dir(tmp_path)
    pkg = load_package(pkg_dir)
    result = save_package(pkg, pkg_dir)
    assert result == pkg_dir


def test_package_path_context(tmp_path):
    import sys

    pkg_dir = _make_pkg_dir(tmp_path)
    pkg = load_package(pkg_dir)
    path_str = str(pkg_dir)

    assert path_str not in sys.path
    with _PackagePathContext(pkg):
        assert path_str in sys.path
    assert path_str not in sys.path


# ── WorkflowRegistry ──────────────────────────────────────────────────────────

def test_registry_empty(tmp_path):
    reg = WorkflowRegistry(workflows_dir=tmp_path)
    assert reg.list() == []


def test_registry_save_and_list(tmp_path):
    src = _make_pkg_dir(tmp_path / "src")
    pkg = load_package(src)

    reg_dir = tmp_path / "registry"
    reg = WorkflowRegistry(workflows_dir=reg_dir)
    reg.save(pkg)

    assert reg.list() == ["test_workflow"]
    assert reg.exists("test_workflow")


def test_registry_get(tmp_path):
    src = _make_pkg_dir(tmp_path / "src")
    pkg = load_package(src)

    reg = WorkflowRegistry(workflows_dir=tmp_path / "reg")
    reg.save(pkg)

    loaded = reg.get("test_workflow")
    assert loaded.name == "test_workflow"


def test_registry_get_not_found(tmp_path):
    reg = WorkflowRegistry(workflows_dir=tmp_path)
    with pytest.raises(WorkflowNotFoundError):
        reg.get("does_not_exist")


def test_registry_delete(tmp_path):
    src = _make_pkg_dir(tmp_path / "src")
    pkg = load_package(src)

    reg = WorkflowRegistry(workflows_dir=tmp_path / "reg")
    reg.save(pkg)
    reg.delete("test_workflow")

    assert reg.list() == []


# ── GraphNode schema — new kinds ──────────────────────────────────────────────

def test_graphnode_accepts_function_kind():
    node = GraphNode(id="n1", kind="function", callable="operator.add")
    assert node.kind == "function"
    assert node.callable == "operator.add"


def test_graphnode_accepts_python_kind():
    node = GraphNode(id="n1", kind="python", callable="os.getcwd")
    assert node.kind == "python"


def test_graphnode_accepts_tool_kind():
    node = GraphNode(id="n1", kind="tool", tools=["list_dir"], tool_args={"path": "."})
    assert node.kind == "tool"
    assert node.tool_args == {"path": "."}


def test_graphnode_rejects_unknown_kind():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="kind must be one of"):
        GraphNode(id="n1", kind="unknown_kind")


# ── GraphExecutor: function node ──────────────────────────────────────────────

def _concat_strings(topic: str, **_: Any) -> str:
    return f"processed:{topic}"


def test_executor_function_node():
    graph = Graph(
        name="fn_test",
        inputs=[{"name": "topic", "type": "string"}],
        nodes=[
            GraphNode(
                id="process",
                kind="function",
                callable="tests.test_workflow_package._concat_strings",
            )
        ],
        outputs=["process"],
    )
    executor = GraphExecutor(graph, provider=_EchoProvider(), log_traces=False)
    result = executor.run({"topic": "hello"})

    assert result.nodes["process"].error is None
    assert result.nodes["process"].raw_output == "processed:hello"


def test_executor_function_node_missing_callable():
    graph = Graph(
        name="fn_test",
        nodes=[GraphNode(id="bad", kind="function")],
        outputs=["bad"],
    )
    executor = GraphExecutor(graph, provider=_EchoProvider(), log_traces=False)
    result = executor.run({})
    assert result.nodes["bad"].error is not None
    assert "callable" in result.nodes["bad"].error


def test_executor_function_node_bad_import():
    graph = Graph(
        name="fn_test",
        nodes=[GraphNode(id="bad", kind="function", callable="no_such_module.fn")],
        outputs=["bad"],
    )
    executor = GraphExecutor(graph, provider=_EchoProvider(), log_traces=False)
    result = executor.run({})
    assert result.nodes["bad"].error is not None


# ── GraphExecutor: tool node ──────────────────────────────────────────────────

def test_executor_tool_node(tmp_path):
    list_dir = {t.name: t for t in all_tools()}["list_dir"]
    ctx = ToolContext(cwd=tmp_path, io=_StubIO())
    pool = ToolPool([list_dir])

    graph = Graph(
        name="tool_test",
        nodes=[
            GraphNode(
                id="scan",
                kind="tool",
                tools=["list_dir"],
                tool_args={"path": "."},
            )
        ],
        outputs=["scan"],
    )
    executor = GraphExecutor(
        graph, provider=_EchoProvider(), native_tools=pool, tool_ctx=ctx, log_traces=False
    )
    result = executor.run({})

    assert result.nodes["scan"].error is None
    assert result.nodes["scan"].raw_output is not None


def test_executor_tool_node_no_toolkit():
    graph = Graph(
        name="tool_test",
        nodes=[GraphNode(id="scan", kind="tool", tools=["list_dir"])],
        outputs=["scan"],
    )
    executor = GraphExecutor(graph, provider=_EchoProvider(), log_traces=False)
    result = executor.run({})
    assert result.nodes["scan"].error is not None


# ── end-to-end: 3-node package (agent + function + tool) ─────────────────────

def test_e2e_three_node_package(tmp_path):
    """3-node workflow: base-agent → function → tool, loaded from disk, run via WorkflowRunner."""
    pkg_dir = tmp_path / "e2e_workflow"
    pkg_dir.mkdir()

    manifest = {"name": "e2e_workflow", "version": "1.0.0", "entrypoint": "graph.yaml"}
    (pkg_dir / "workflow.yaml").write_text(yaml.dump(manifest))

    graph_spec = {
        "name": "e2e_workflow",
        "inputs": [{"name": "topic", "type": "string"}],
        "nodes": [
            {
                "id": "research",
                "kind": "base",
                "purpose": "Research {topic}",
            },
            {
                "id": "transform",
                "kind": "function",
                "callable": "tests.test_workflow_package._concat_strings",
                "depends_on": ["research"],
            },
            {
                "id": "list_files",
                "kind": "tool",
                "tools": ["list_dir"],
                "tool_args": {"path": "."},
                "depends_on": ["transform"],
            },
        ],
        "outputs": ["list_files"],
    }
    (pkg_dir / "graph.yaml").write_text(yaml.dump(graph_spec))

    pkg = load_package(pkg_dir)
    assert len(pkg.graph.nodes) == 3

    provider = _EchoProvider()
    tool_ctx = ToolContext(cwd=tmp_path, io=_StubIO())

    runner = WorkflowRunner(
        provider,
        cwd=tmp_path,
        tool_context=tool_ctx,
    )
    result = runner.run(pkg, {"topic": "graph workflows"})

    assert result.nodes["research"].error is None
    assert result.nodes["transform"].error is None
    assert result.nodes["list_files"].error is None

    # agent node echo output flows through
    assert "REPLY[" in str(result.nodes["research"].raw_output)
    # function node result
    assert result.nodes["transform"].raw_output == "processed:graph workflows"
    # tool node result is non-empty
    assert result.nodes["list_files"].raw_output


# ── R3+R4: native provider + ToolPool drive GraphExecutor (no adapter layer) ──
# Replaces the old Phase-A tests that verified ProviderChatModel / MasterAgentToolAdapter.
# Those adapter classes still exist for backward compat but are no longer the primary path.

def test_native_provider_drives_graph_executor():
    """Native provider (no ProviderChatModel wrapper) drives a 2-node base graph."""
    provider = _EchoProvider()
    graph = Graph(
        name="native_smoke",
        inputs=[{"name": "topic", "type": "string"}],
        nodes=[
            GraphNode(id="research", purpose="Research {topic}", mode=NodeMode.TEXT),
            GraphNode(
                id="summarize",
                depends_on=["research"],
                purpose="Summarize the research",
                mode=NodeMode.TEXT,
            ),
        ],
        outputs=["summarize"],
    )
    executor = GraphExecutor(graph, provider=provider, log_traces=False)
    result = executor.run({"topic": "graph workflows"})

    assert result.final, "final outputs should be populated"
    assert "summarize" in result.final
    assert all(node.error is None for node in result.nodes.values())
    # Dependency output must reach the downstream node (echo nests the prior reply).
    assert "REPLY[" in str(result.nodes["summarize"].raw_output)


def test_native_tool_pool_drives_tool_node():
    """Native ToolPool (no MasterAgentToolAdapter) executes a tool-kind node."""
    list_dir_tool = {t.name: t for t in all_tools()}["list_dir"]
    repo_root = Path(__file__).resolve().parent.parent
    ctx = ToolContext(cwd=repo_root, io=_StubIO())
    pool = ToolPool([list_dir_tool])

    graph = Graph(
        name="tool_node_smoke",
        nodes=[
            GraphNode(
                id="list_files",
                kind="tool",
                tools=["list_dir"],
                tool_args={"path": "."},
            ),
        ],
        outputs=["list_files"],
    )
    executor = GraphExecutor(
        graph, provider=_EchoProvider(), native_tools=pool, tool_ctx=ctx, log_traces=False
    )
    result = executor.run({})

    assert result.nodes["list_files"].error is None
    assert result.nodes["list_files"].raw_output  # non-empty listing


# ── WorkflowRunner: trace export ──────────────────────────────────────────────

def test_trace_written_and_valid(tmp_path):
    pkg_dir = tmp_path / "wf"
    _function_only_pkg(pkg_dir)
    pkg = load_package(pkg_dir)

    trace_path = tmp_path / "traces" / "run.json"
    runner = WorkflowRunner(_DummyProvider())
    runner.run(pkg, {}, trace_path=trace_path)

    assert trace_path.exists()
    data = json.loads(trace_path.read_text())
    assert "meta" in data and "steps" in data
    assert isinstance(data["steps"], list)
    assert data["meta"].get("workflow") == "wf"


def test_no_trace_when_path_omitted(tmp_path):
    pkg_dir = tmp_path / "wf2"
    _function_only_pkg(pkg_dir)
    pkg = load_package(pkg_dir)

    runner = WorkflowRunner(_DummyProvider())
    runner.run(pkg, {})  # no trace_path → nothing written
    assert not (tmp_path / "traces").exists()
