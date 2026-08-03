"""Verification is keyed to what it verified, not to a flag.

Verifying re-runs the whole graph — the expensive half of a build. The session
stores a fingerprint of the graph (plus the tools authored so far) alongside each
result, so a repeat `test_workflow` on an unchanged design is answered from the
record instead of paid for again, while any real change stales it.

Hermetic: function-node workflows, with `ScriptedProvider` playing the deriver and
judge. The script is the assertion — a turn left unconsumed means a run that did
not happen, and a script that runs dry means one that should not have.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neurosurfer.architect.agent import BuildSession, architect_tools
from neurosurfer.architect.knowledge import KnowledgeBase
from neurosurfer.graph.workflow.registry import WorkflowRegistry
from neurosurfer.tools.base import AutoApproveIOHandler, ToolContext

from ..fakes import ScriptedProvider

FN = "tests.architect.test_architect_verify_cache"


def _double(x=None, **kwargs):
    return (int(x) if x is not None else 0) * 2


@pytest.fixture(scope="module")
def kb():
    return KnowledgeBase()


def _plan_json(**test_inputs) -> str:
    return json.dumps({
        "criteria": [{"id": "doubles", "description": "output is twice the input"}],
        "test_inputs": test_inputs,
    })


def _judge_json(passed: bool) -> str:
    return json.dumps({
        "verdicts": [{"id": "doubles", "passed": passed, "reason": "checked"}],
        "diagnosis": "" if passed else "the d node does not double",
        "suggestions": "" if passed else "fix the d node",
    })


def _session(tmp_path: Path, kb, provider) -> BuildSession:
    s = BuildSession(
        intent="double a number",
        staging_root=tmp_path / "staging",
        registry=WorkflowRegistry(workflows_dir=tmp_path / "registry"),
        knowledge=kb,
        provider=provider,
        verification_mode="required",
    )
    s.name = "doubler"
    s.inputs = [{"name": "x", "type": "integer", "required": True}]
    s.nodes = [{"id": "d", "kind": "function", "callable": f"{FN}._double"}]
    s.outputs = ["d"]
    return s


def _tool(session, name):
    return next(t for t in architect_tools(session) if t.name == name)


def _ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, io=AutoApproveIOHandler())


# ── the win ─────────────────────────────────────────────────────────────────────

async def test_repeat_test_does_not_re_run_the_graph(tmp_path, kb):
    """Two `test_workflow` calls, one script entry each for derive and judge.

    If the second call re-ran anything it would pop a turn that isn't there and the
    scripted provider would hand back an empty response, failing the assertions.
    """
    provider = ScriptedProvider(turns=[(_plan_json(x=3), []), (_judge_json(True), [])])
    session = _session(tmp_path, kb, provider)
    ctx = _ctx(tmp_path)

    first = await _tool(session, "test_workflow").run({}, ctx)
    assert not first.is_error and "VERIFICATION PASSED" in first.content
    assert provider.calls == 2          # derive + judge
    assert session.graph_runs == 1

    second = await _tool(session, "test_workflow").run({}, ctx)
    assert not second.is_error and "VERIFICATION PASSED" in second.content
    assert "Not re-run" in second.content
    assert provider.calls == 2          # unchanged: nothing was asked of the model
    assert session.graph_runs == 1      # and the graph was not executed again


async def test_a_failed_verification_is_also_reused(tmp_path, kb):
    """Re-rolling a judge on an unchanged graph invites retrying past a verdict
    instead of fixing the design, so a failure is cached exactly like a pass."""
    provider = ScriptedProvider(turns=[(_plan_json(x=3), []), (_judge_json(False), [])])
    session = _session(tmp_path, kb, provider)
    ctx = _ctx(tmp_path)

    first = await _tool(session, "test_workflow").run({}, ctx)
    assert first.is_error and "VERIFICATION FAILED" in first.content

    second = await _tool(session, "test_workflow").run({}, ctx)
    assert second.is_error, "a cached failure must stay on the error channel"
    assert "Not re-run" in second.content
    assert provider.calls == 2
    assert session.graph_runs == 1


async def test_a_no_op_edit_does_not_stale_the_verification(tmp_path, kb):
    """Writing a node back with the values it already has is a common model move.
    It changes nothing, so it must not cost a full re-verification."""
    provider = ScriptedProvider(turns=[(_plan_json(x=3), []), (_judge_json(True), [])])
    session = _session(tmp_path, kb, provider)
    ctx = _ctx(tmp_path)

    await _tool(session, "test_workflow").run({}, ctx)
    assert session.last_verification is not None

    same = await _tool(session, "update_node").run(
        {"id": "d", "patch": {"callable": f"{FN}._double"}}, ctx)
    assert not same.is_error
    assert session.last_verification is not None, "an identical write is not a change"

    again = await _tool(session, "test_workflow").run({}, ctx)
    assert "Not re-run" in again.content
    assert session.graph_runs == 1


async def test_reverting_an_edit_restores_the_verification(tmp_path, kb):
    """The fingerprint describes the graph, not the history of how it got there."""
    provider = ScriptedProvider(turns=[(_plan_json(x=3), []), (_judge_json(True), [])])
    session = _session(tmp_path, kb, provider)
    ctx = _ctx(tmp_path)

    await _tool(session, "test_workflow").run({}, ctx)
    assert session.last_verification is not None

    await _tool(session, "update_node").run({"id": "d", "patch": {"writes": "out"}}, ctx)
    assert session.last_verification is None, "a real change stales it"

    # Put it back exactly as it was.
    await _tool(session, "remove_node").run({"id": "d"}, ctx)
    await _tool(session, "add_node").run(
        {"node": {"id": "d", "kind": "function", "callable": f"{FN}._double"}}, ctx)
    assert session.last_verification is not None


# ── and it still stales when it should ──────────────────────────────────────────

async def test_a_real_edit_forces_a_re_run(tmp_path, kb):
    provider = ScriptedProvider(turns=[
        (_plan_json(x=3), []), (_judge_json(True), []),   # first verification
        (_judge_json(True), []),                          # second: judge only
    ])
    session = _session(tmp_path, kb, provider)
    ctx = _ctx(tmp_path)

    await _tool(session, "test_workflow").run({}, ctx)
    await _tool(session, "update_node").run({"id": "d", "patch": {"writes": "out"}}, ctx)
    assert session.last_verification is None

    again = await _tool(session, "test_workflow").run({}, ctx)
    assert "Not re-run" not in again.content
    assert session.graph_runs == 2, "the changed graph was actually re-executed"
    # The acceptance plan is derived once per build and reused, so only the judge ran.
    assert provider.calls == 3


async def test_changing_the_output_set_stales_it(tmp_path, kb):
    """`set_outputs` changes what the judge is shown, so it must count as a change."""
    provider = ScriptedProvider(turns=[(_plan_json(x=3), []), (_judge_json(True), [])])
    session = _session(tmp_path, kb, provider)
    session.nodes.append({"id": "e", "kind": "function", "callable": f"{FN}._double",
                          "depends_on": ["d"]})
    ctx = _ctx(tmp_path)

    await _tool(session, "test_workflow").run({}, ctx)
    assert session.last_verification is not None

    await _tool(session, "set_outputs").run({"outputs": ["d", "e"]}, ctx)
    assert session.last_verification is None


async def test_authoring_a_tool_stales_it(tmp_path, kb):
    """The graph text can be identical while meaning something different: a node may
    name a tool that did not exist when the verification ran."""
    provider = ScriptedProvider(turns=[(_plan_json(x=3), []), (_judge_json(True), [])])
    session = _session(tmp_path, kb, provider)
    ctx = _ctx(tmp_path)

    await _tool(session, "test_workflow").run({}, ctx)
    assert session.last_verification is not None

    session.authored_tools.append("newly_authored")
    assert session.last_verification is None


async def test_different_test_inputs_are_a_different_test(tmp_path, kb):
    provider = ScriptedProvider(turns=[
        (_plan_json(x=3), []), (_judge_json(True), []),   # default inputs
        (_judge_json(True), []),                          # explicit inputs → re-run
    ])
    session = _session(tmp_path, kb, provider)
    ctx = _ctx(tmp_path)

    await _tool(session, "test_workflow").run({}, ctx)
    assert session.graph_runs == 1

    other = await _tool(session, "test_workflow").run({"test_inputs": {"x": 9}}, ctx)
    assert "Not re-run" not in other.content
    assert session.graph_runs == 2

    # ...and that second result is itself cached under its own inputs.
    repeat = await _tool(session, "test_workflow").run({"test_inputs": {"x": 9}}, ctx)
    assert "Not re-run" in repeat.content
    assert session.graph_runs == 2


# ── the cost is a number now ────────────────────────────────────────────────────

async def test_the_report_states_how_many_graph_runs_it_cost(tmp_path, kb):
    provider = ScriptedProvider(turns=[(_plan_json(x=3), []), (_judge_json(True), [])])
    session = _session(tmp_path, kb, provider)

    res = await _tool(session, "test_workflow").run({}, _ctx(tmp_path))
    assert "1 graph run" in res.content
    assert session.last_report.graph_runs == 1


async def test_branch_cases_are_counted_in_the_run_total(tmp_path, kb):
    """Each extra case re-runs the whole graph; the count must say so."""
    plan = json.dumps({
        "criteria": [{"id": "doubles", "description": "output is twice the input"}],
        "test_inputs": {"x": 3},
        "extra_cases": [{"label": "big", "test_inputs": {"x": 100}}],
    })
    provider = ScriptedProvider(turns=[(plan, []), (_judge_json(True), [])])
    session = _session(tmp_path, kb, provider)
    # Two nodes so the first run cannot reach full coverage and short-circuit the case.
    session.nodes.append({"id": "e", "kind": "function", "callable": f"{FN}._double",
                          "when": "inputs.x > 50"})
    session.outputs = ["d", "e"]

    res = await _tool(session, "test_workflow").run({}, _ctx(tmp_path))
    assert session.last_report.graph_runs == 2
    assert "2 graph runs" in res.content
