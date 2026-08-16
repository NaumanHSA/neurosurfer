"""Phase 4 — ReAct Architect agent: session, toolbelt, agent loop, A/B harness.

The toolbelt is exercised directly (async tool calls against a BuildSession), and
the full agent loop is driven by the ScriptedProvider — deterministic native tool
calls, no LLM — proving the scaffold, gating, and terminal contract end-to-end.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from neurosurfer.architect import ArchitectAgent, WorkflowInfeasible
from neurosurfer.architect.agent import (
    BuildSession,
    architect_tools,
    render_report,
    run_harness,
)
from neurosurfer.architect.knowledge import KnowledgeBase
from neurosurfer.graph.workflow.registry import WorkflowRegistry
from neurosurfer.tools.base import AutoApproveIOHandler, ToolContext

from ..fakes import ScriptedProvider

FN = "tests.architect.test_architect_agent"


def _shout(**kwargs):
    return "hello"


@pytest.fixture(scope="module")
def kb():
    return KnowledgeBase()


@pytest.fixture()
def session(tmp_path: Path, kb) -> BuildSession:
    return BuildSession(
        intent="test intent",
        staging_root=tmp_path / "staging",
        registry=WorkflowRegistry(workflows_dir=tmp_path / "registry"),
        knowledge=kb,
        # Verification has its own suite; these cover the toolbelt.
        verification_mode="off",
    )


@pytest.fixture()
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, io=AutoApproveIOHandler())


def _tool(session, name):
    return next(t for t in architect_tools(session) if t.name == name)


# ── toolbelt units ──────────────────────────────────────────────────────────────

async def test_set_workflow_and_add_node(session, ctx):
    res = await _tool(session, "set_workflow").run(
        {"name": "My Flow!", "description": "d",
         "inputs": [{"name": "topic", "type": "string", "required": True}]},
        ctx,
    )
    assert not res.is_error
    assert session.name == "my_flow"  # sanitized to snake_case

    res = await _tool(session, "add_node").run(
        {"node": {"id": "write", "kind": "base", "purpose": "Write about {topic}"}}, ctx
    )
    assert not res.is_error
    assert session.node_ids() == ["write"]


async def test_add_node_rejects_bad_spec_and_duplicates(session, ctx):
    add = _tool(session, "add_node")
    bad = await add.run({"node": {"id": "x", "kind": "not_a_kind"}}, ctx)
    assert bad.is_error and "kind" in bad.content

    ok = await add.run({"node": {"id": "x", "kind": "base", "purpose": "p"}}, ctx)
    assert not ok.is_error
    dup = await add.run({"node": {"id": "x", "kind": "base", "purpose": "p2"}}, ctx)
    assert dup.is_error and "update_node" in dup.content


async def test_add_node_warns_on_unknown_tool(session, ctx):
    res = await _tool(session, "add_node").run(
        {"node": {"id": "r", "kind": "react", "purpose": "p",
                  "tools": ["no_such_tool_xyz"]}}, ctx
    )
    assert not res.is_error  # warned, not blocked (author_tool may follow)
    assert "not registered" in res.content


async def test_update_and_remove_node(session, ctx):
    await _tool(session, "add_node").run(
        {"node": {"id": "a", "kind": "base", "purpose": "old"}}, ctx)
    upd = await _tool(session, "update_node").run(
        {"id": "a", "patch": {"purpose": "new"}}, ctx)
    assert not upd.is_error
    assert session.get_node("a")["purpose"] == "new"

    missing = await _tool(session, "update_node").run(
        {"id": "ghost", "patch": {}}, ctx)
    assert missing.is_error

    rem = await _tool(session, "remove_node").run({"id": "a"}, ctx)
    assert not rem.is_error and session.nodes == []


async def test_validate_reports_issues_then_valid(session, ctx):
    validate = _tool(session, "validate_workflow")
    empty = await validate.run({}, ctx)
    assert empty.is_error and "no nodes" in empty.content

    await _tool(session, "set_workflow").run({"name": "vtest"}, ctx)
    # bad depends_on → structural error surfaced with the exact problem
    await _tool(session, "add_node").run(
        {"node": {"id": "b", "kind": "function", "callable": f"{FN}._shout",
                  "depends_on": ["ghost"]}}, ctx)
    bad = await validate.run({}, ctx)
    assert bad.is_error and "ghost" in bad.content

    await _tool(session, "update_node").run({"id": "b", "patch": {"depends_on": []}}, ctx)
    good = await validate.run({}, ctx)
    assert not good.is_error and "VALID" in good.content


async def test_register_refuses_invalid_then_registers(session, ctx):
    register = _tool(session, "register_workflow")
    refused = await register.run({}, ctx)
    assert refused.is_error and "Refusing" in refused.content
    assert session.registered_path is None

    await _tool(session, "set_workflow").run({"name": "regtest"}, ctx)
    await _tool(session, "add_node").run(
        {"node": {"id": "go", "kind": "function", "callable": f"{FN}._shout"}}, ctx)
    ok = await register.run({}, ctx)
    assert not ok.is_error
    assert session.registered_path is not None
    assert session.registry.exists("regtest")


async def test_knowledge_tools(session, ctx):
    desc = await _tool(session, "describe_capability").run({"name": "router"}, ctx)
    assert not desc.is_error and "branch" in desc.content.lower()
    tool_desc = await _tool(session, "describe_capability").run({"name": "read_file"}, ctx)
    assert not tool_desc.is_error and "input_schema" in tool_desc.content
    nope = await _tool(session, "describe_capability").run({"name": "zzz"}, ctx)
    assert nope.is_error

    docs = await _tool(session, "neurosurfer_docs").run(
        {"query": "workflow package"}, ctx)
    assert not docs.is_error


async def test_author_tool_requires_approval_channel(session, ctx):
    res = await _tool(session, "author_tool").run(
        {"name": "brand_new", "purpose": "does things"}, ctx)
    assert res.is_error and "approval" in res.content.lower()


async def test_declare_blocked_records_reason(session, ctx):
    res = await _tool(session, "declare_blocked").run(
        {"reason": "needs a production Postgres we don't have"}, ctx)
    assert not res.is_error
    assert "Postgres" in session.blocked_reason


# ── full agent loop (scripted provider, no LLM) ────────────────────────────────

def _scripted_agent(turns, tmp_path: Path, kb) -> ArchitectAgent:
    # plan=False: these drive the *builder* loop against a fixed script, so a
    # planning call at the front would consume the first scripted turn. The
    # planner has its own tests.
    return ArchitectAgent(
        ScriptedProvider(turns=turns),
        registry=WorkflowRegistry(workflows_dir=tmp_path / "registry"),
        staging_root=tmp_path / "staging",
        knowledge=kb,
        plan=False,
        # Same reason as plan=False: the script drives the builder loop, and
        # `required` (the default) would demand a test_workflow turn it has
        # not scripted. Verification has its own suite.
        verify="off",
    )


async def test_agent_builds_validates_and_registers(tmp_path, kb):
    turns = [
        ("", [("set_workflow", {
            "name": "greeter", "description": "greets people",
            "inputs": [{"name": "person", "type": "string", "required": True}],
        })]),
        ("", [("add_node", {"node": {
            "id": "greet", "kind": "function", "callable": f"{FN}._shout",
        }})]),
        ("", [("validate_workflow", {})]),
        ("", [("register_workflow", {})]),
        ("Registered the greeter workflow.", []),
    ]
    agent = _scripted_agent(turns, tmp_path, kb)
    path = await agent.build("make a workflow that greets a person")
    assert Path(path).exists()

    from neurosurfer.graph.workflow.package import load_package
    pkg = load_package(Path(path))
    assert pkg.name == "greeter"
    assert [n.id for n in pkg.graph.nodes] == ["greet"]


async def test_agent_declares_blocked_raises_infeasible(tmp_path, kb):
    turns = [
        ("", [("declare_blocked", {"reason": "requires prod credentials"})]),
        ("Cannot build this.", []),
    ]
    agent = _scripted_agent(turns, tmp_path, kb)
    with pytest.raises(WorkflowInfeasible, match="prod credentials"):
        await agent.build("hack the mainframe")


async def test_agent_neither_registered_nor_blocked_errors(tmp_path, kb):
    turns = [("I did nothing useful.", [])]
    agent = _scripted_agent(turns, tmp_path, kb)
    with pytest.raises(RuntimeError, match="without registering"):
        await agent.build("do something")


# ── A/B harness ─────────────────────────────────────────────────────────────────

async def test_harness_compares_builders(tmp_path, kb):
    reg = WorkflowRegistry(workflows_dir=tmp_path / "registry")

    async def good_builder(intent: str) -> str:
        session = BuildSession(intent=intent, staging_root=tmp_path / "staging",
                               registry=reg, knowledge=kb,
                               verification_mode="off")
        session.name = "hb"
        session.nodes = [{"id": "n", "kind": "function", "callable": f"{FN}._shout"}]
        ok, msg = session.register()
        assert ok, msg
        return session.registered_path

    async def bad_builder(intent: str) -> str:
        raise RuntimeError("nope")

    results = await run_harness(
        ["intent one"], {"good": good_builder, "bad": bad_builder})
    by = {c.builder: c for c in results}
    assert by["good"].ok and by["good"].node_count == 1
    assert by["good"].validation_ok is True
    assert not by["bad"].ok and "nope" in by["bad"].error

    report = render_report(results)
    assert "| good |" in report and "1/1" in report and "0/1" in report
    assert "Failures" in report and "nope" in report


# ── the terminal contract: a workflow, or a reason ─────────────────────────────


async def test_a_design_that_cannot_run_is_a_refusal_not_a_crash(tmp_path, kb):
    """A build that stops short with an unrunnable design reports *why*.

    It used to raise `RuntimeError` carrying whatever the model said last — on
    one `gpt-5-mini` run, a paragraph asking the *user* which of two fixes to
    apply. The outcome of a build must not depend on how articulate the model was
    when it ran out of turns: no workflow plus a reason is the same outcome as a
    deliberate refusal, so it is the same exception.

    Here the graph names an output node that does not exist — the gate refuses
    it, and the gate's message is what the caller gets.
    """
    turns = [
        ("", [("set_workflow", {"name": "broken", "description": "d"})]),
        ("", [("add_node", {"node": {
            "id": "a", "kind": "function", "callable": f"{FN}._shout",
        }})]),
        ("", [("add_node", {"node": {
            "id": "b", "kind": "function", "callable": f"{FN}._shout",
            "depends_on": ["a"],
        }})]),
        ("", [("set_outputs", {"outputs": ["b"]})]),
        # Break it — `b` is gone but still declared as the result — and then stop
        # calling tools so the loop ends with a design that cannot run.
        ("", [("remove_node", {"id": "b"})]),
        ("I think that is everything.", []),
    ] + [("Still nothing to add.", [])] * 8      # outlast the nudges

    agent = _scripted_agent(turns, tmp_path, kb)
    with pytest.raises(WorkflowInfeasible) as exc:
        await agent.build("build me something")

    message = str(exc.value)
    assert "could not produce a runnable workflow" in message
    assert "not proof the request is impossible" in message, (
        "it must not claim the request was impossible — only that the build stopped"
    )
    # The gate's own words, so the reader has something to act on.
    assert "validation failed" in message.lower()


async def test_building_nothing_at_all_is_still_an_error(tmp_path, kb):
    """The one case that stays a `RuntimeError`. No design means the agent did
    not work, which is a fault here — not a statement about the request."""
    turns = [("I am not going to build anything.", [])] * 10

    agent = _scripted_agent(turns, tmp_path, kb)
    with pytest.raises(RuntimeError) as exc:
        await agent.build("build me something")

    assert not isinstance(exc.value, WorkflowInfeasible)
    assert "built nothing" in str(exc.value)
