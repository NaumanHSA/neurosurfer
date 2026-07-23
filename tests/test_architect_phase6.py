"""Phase 6 — conditional/multi-node design + MCP wiring + branch coverage.

Covers:
- MCP runtime host: stdio FastMCP server → live registry → workflow-usable
  catalog → agent add_node (no warning) → WorkflowRunner executes an MCP tool
  inside a registered workflow (cross-loop marshalling included).
- Branch-coverage verification: extra cases run, never-executed nodes reported.
- Scripted agent builds a genuinely branching (router) workflow end to end.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

from neurosurfer.architect import ArchitectAgent
from neurosurfer.architect.agent import (
    AcceptancePlan,
    BuildSession,
    architect_tools,
    verify_workflow,
)
from neurosurfer.architect.knowledge import KnowledgeBase
from neurosurfer.graph.workflow.registry import WorkflowRegistry
from neurosurfer.tools.base import AutoApproveIOHandler, ToolContext

from .fakes import ScriptedProvider

FN = "tests.test_architect_phase6"


def _mark(**kwargs):
    return "ran"


def _is_big(n=None, **kwargs):
    return "big" if (n or 0) > 5 else "small"


@pytest.fixture(scope="module")
def kb():
    return KnowledgeBase()


def _ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, io=AutoApproveIOHandler())


def _tool(session, name):
    return next(t for t in architect_tools(session) if t.name == name)


# ── MCP: runtime host + workflow integration ───────────────────────────────────

mcp_mod = pytest.importorskip("mcp")

_SERVER_SCRIPT = textwrap.dedent('''
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("fixture6")

    @server.tool(annotations={"readOnlyHint": True})
    def city_motto(city: str) -> str:
        """Return the official motto for a city."""
        return f"motto of {city}: onwards and upwards"

    server.run()
''')


@pytest.fixture()
def mcp_store(tmp_path: Path):
    """A temp McpStore configured with a real stdio FastMCP server."""
    from neurosurfer.config.mcp import McpServerConfig, McpStore

    script = tmp_path / "fixture_server.py"
    script.write_text(_SERVER_SCRIPT, encoding="utf-8")
    store = McpStore(path=tmp_path / "mcp.json")
    store.add(McpServerConfig(
        name="fixture6", transport="stdio",
        command=sys.executable, args=[str(script)],
    ))
    return store


@pytest.fixture()
def mcp_connected(mcp_store):
    """Host the fixture server for the test, tearing the runtime down after."""
    from neurosurfer.mcp.runtime import ensure_mcp_tools, shutdown_mcp

    statuses = ensure_mcp_tools(mcp_store)
    yield statuses
    shutdown_mcp()


def test_mcp_runtime_publishes_workflow_usable_tools(mcp_connected):
    from neurosurfer.tools.registry import all_tools, workflow_node_tool_names

    assert any(s.connected for s in mcp_connected)
    names = {t.name for t in all_tools()}
    assert "city_motto" in names
    # Phase 6 policy: live MCP tools ARE workflow-usable (reconnect-on-demand).
    assert "city_motto" in workflow_node_tool_names()


def test_mcp_runtime_is_idempotent_and_shuts_down(mcp_store):
    from neurosurfer.mcp.runtime import ensure_mcp_tools, mcp_statuses, shutdown_mcp
    from neurosurfer.tools.registry import all_tools

    s1 = ensure_mcp_tools(mcp_store)
    s2 = ensure_mcp_tools(mcp_store)  # second call: cached, no reconnect
    assert [x.name for x in s1] == [x.name for x in s2]
    assert mcp_statuses()
    shutdown_mcp()
    assert "city_motto" not in {t.name for t in all_tools()}  # live tools cleared


async def test_agent_add_node_accepts_mcp_tool(tmp_path, kb, mcp_connected):
    session = BuildSession(
        intent="use mcp", staging_root=tmp_path / "staging",
        registry=WorkflowRegistry(workflows_dir=tmp_path / "registry"),
        knowledge=KnowledgeBase(),  # fresh: catalog includes the live tool
    )
    res = await _tool(session, "add_node").run(
        {"node": {"id": "motto", "kind": "react", "purpose": "fetch the motto",
                  "tools": ["city_motto"]}}, _ctx(tmp_path))
    assert not res.is_error
    assert "not registered" not in res.content  # no unknown-tool warning


def test_workflow_runner_executes_mcp_tool_node(tmp_path, mcp_connected):
    """A registered workflow with a tool-kind node backed by MCP actually runs —
    including the cross-loop hop from the executor thread to the MCP host loop."""
    import yaml

    from neurosurfer.graph.workflow.package import load_package
    from neurosurfer.graph.workflow.runner import WorkflowRunner

    pkg_dir = tmp_path / "motto_wf"
    pkg_dir.mkdir()
    (pkg_dir / "workflow.yaml").write_text(yaml.dump(
        {"name": "motto_wf", "version": "1.0.0", "entrypoint": "graph.yaml"}))
    (pkg_dir / "graph.yaml").write_text(yaml.dump({
        "name": "motto_wf",
        "inputs": [{"name": "city", "type": "string", "required": True}],
        "nodes": [{"id": "get_motto", "kind": "tool", "tools": ["city_motto"],
                   "tool_args": {}}],
        "outputs": ["get_motto"],
    }))

    class _Dummy:
        model = "dummy"
        capabilities = type("C", (), {"context_window": 8192,
                                      "max_output_tokens": 512})()

    result = WorkflowRunner(_Dummy()).run(load_package(pkg_dir), {"city": "Oslo"})
    assert result.nodes["get_motto"].error is None
    assert "motto of Oslo" in str(result.nodes["get_motto"].raw_output)


# ── branch-coverage verification ────────────────────────────────────────────────

def _router_session(tmp_path: Path, kb, provider) -> BuildSession:
    s = BuildSession(
        intent="handle big and small numbers differently",
        staging_root=tmp_path / "staging",
        registry=WorkflowRegistry(workflows_dir=tmp_path / "registry"),
        knowledge=kb, provider=provider,
    )
    s.name = "sizer"
    s.inputs = [{"name": "n", "type": "integer", "required": True}]
    s.nodes = [
        {"id": "classify", "kind": "function", "callable": f"{FN}._is_big"},
        {"id": "route", "kind": "router", "depends_on": ["classify"],
         "cases": [{"when": "contains(lower(nodes.classify), 'big')", "to": "big"}],
         "default": "small"},
        {"id": "big", "kind": "function", "callable": f"{FN}._mark",
         "depends_on": ["route"]},
        {"id": "small", "kind": "function", "callable": f"{FN}._mark",
         "depends_on": ["route"]},
    ]
    s.outputs = ["big", "small"]
    return s


def _judge_pass() -> str:
    return json.dumps({"verdicts": [{"id": "works", "passed": True, "reason": "ok"}]})


async def test_verify_reports_coverage_gap_with_single_case(tmp_path, kb):
    provider = ScriptedProvider(turns=[(_judge_pass(), [])])
    session = _router_session(tmp_path, kb, provider)
    session.stage()
    plan = AcceptancePlan(
        criteria=[{"id": "works", "description": "routes correctly"}],
        test_inputs={"n": 10},   # only the 'big' branch fires
    )
    report = await verify_workflow(
        provider, intent=session.intent,
        package_dir=session.staging_root / session.name, plan=plan)
    assert report.passed  # gaps warn, they don't fail
    assert report.coverage_gaps == ["small"]
    assert "COVERAGE WARNING" in report.render()


async def test_verify_full_coverage_with_extra_cases(tmp_path, kb):
    provider = ScriptedProvider(turns=[(_judge_pass(), [])])
    session = _router_session(tmp_path, kb, provider)
    session.stage()
    plan = AcceptancePlan(
        criteria=[{"id": "works", "description": "routes correctly"}],
        test_inputs={"n": 10},
        extra_cases=[{"label": "small_branch", "test_inputs": {"n": 1}}],
    )
    report = await verify_workflow(
        provider, intent=session.intent,
        package_dir=session.staging_root / session.name, plan=plan)
    assert report.passed
    assert report.coverage_gaps == []
    assert report.case_results == [
        {"label": "small_branch", "ok": True, "error": None}]


async def test_verify_failing_branch_case_fails_verification(tmp_path, kb):
    provider = ScriptedProvider(turns=[(_judge_pass(), [])])
    session = _router_session(tmp_path, kb, provider)
    # Break the small branch so the extra case errors.
    session.nodes[3] = {"id": "small", "kind": "function",
                        "callable": f"{FN}._boom", "depends_on": ["route"]}
    session.stage()
    plan = AcceptancePlan(
        criteria=[{"id": "works", "description": "routes correctly"}],
        test_inputs={"n": 10},
        extra_cases=[{"label": "small_branch", "test_inputs": {"n": 1}}],
    )
    report = await verify_workflow(
        provider, intent=session.intent,
        package_dir=session.staging_root / session.name, plan=plan)
    assert not report.passed
    assert report.case_results[0]["ok"] is False
    assert "branch" in report.diagnosis.lower()


def _boom(**kwargs):
    raise RuntimeError("branch boom")


# ── scripted agent builds a branching workflow end to end ──────────────────────

async def test_agent_builds_router_workflow_scripted(tmp_path, kb):
    turns = [
        ("", [("set_workflow", {
            "name": "sizer", "description": "route by size",
            "inputs": [{"name": "n", "type": "integer", "required": True}],
            "outputs": ["big", "small"],
        })]),
        ("", [("add_node", {"node": {"id": "classify", "kind": "function",
                                     "callable": f"{FN}._is_big"}})]),
        ("", [("add_node", {"node": {
            "id": "route", "kind": "router", "depends_on": ["classify"],
            "cases": [{"when": "contains(lower(nodes.classify), 'big')", "to": "big"}],
            "default": "small"}})]),
        ("", [("add_node", {"node": {"id": "big", "kind": "function",
                                     "callable": f"{FN}._mark", "depends_on": ["route"]}})]),
        ("", [("add_node", {"node": {"id": "small", "kind": "function",
                                     "callable": f"{FN}._mark", "depends_on": ["route"]}})]),
        ("", [("validate_workflow", {})]),
        ("", [("register_workflow", {})]),
        ("Registered.", []),
    ]
    agent = ArchitectAgent(
        ScriptedProvider(turns=turns),
        registry=WorkflowRegistry(workflows_dir=tmp_path / "registry"),
        staging_root=tmp_path / "staging",
        knowledge=kb,
    )
    path = await agent.build("route big and small numbers differently")

    from neurosurfer.graph.workflow.package import load_package
    pkg = load_package(Path(path))
    router = next(n for n in pkg.graph.nodes if n.kind == "router")
    assert router.cases and router.default == "small"
    # And the registered branching workflow actually executes both ways.
    from neurosurfer.graph.workflow.runner import WorkflowRunner

    class _Dummy:
        model = "dummy"
        capabilities = type("C", (), {"context_window": 8192,
                                      "max_output_tokens": 512})()

    big = WorkflowRunner(_Dummy()).run(pkg, {"n": 9})
    assert big.nodes["big"].skipped is False and big.nodes["small"].skipped is True
    small = WorkflowRunner(_Dummy()).run(pkg, {"n": 2})
    assert small.nodes["small"].skipped is False and small.nodes["big"].skipped is True
