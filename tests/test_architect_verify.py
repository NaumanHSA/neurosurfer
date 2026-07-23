"""Phase 5 — closed-loop verification: engine, test_workflow tool, register gating.

All hermetic: function-node workflows (no LLM inside the workflow) + the
ScriptedProvider playing the deriver/judge roles with canned JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neurosurfer.architect import ArchitectAgent
from neurosurfer.architect.agent import (
    AcceptancePlan,
    BuildSession,
    architect_tools,
    derive_acceptance,
    verify_workflow,
)
from neurosurfer.architect.knowledge import KnowledgeBase
from neurosurfer.graph.workflow.registry import WorkflowRegistry
from neurosurfer.tools.base import AutoApproveIOHandler, ToolContext

from .fakes import ScriptedProvider

FN = "tests.test_architect_verify"


def _double(x=None, **kwargs):
    return (int(x) if x is not None else 0) * 2


def _boom(**kwargs):
    raise RuntimeError("kaboom")


@pytest.fixture(scope="module")
def kb():
    return KnowledgeBase()


def _session(tmp_path: Path, kb, provider=None, mode="encouraged") -> BuildSession:
    s = BuildSession(
        intent="double a number",
        staging_root=tmp_path / "staging",
        registry=WorkflowRegistry(workflows_dir=tmp_path / "registry"),
        knowledge=kb,
        provider=provider,
        verification_mode=mode,
    )
    s.name = "doubler"
    s.inputs = [{"name": "x", "type": "integer", "required": True}]
    s.nodes = [{"id": "d", "kind": "function", "callable": f"{FN}._double"}]
    s.outputs = ["d"]
    return s


def _ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, io=AutoApproveIOHandler())


def _plan_json(**test_inputs) -> str:
    return json.dumps({
        "criteria": [
            {"id": "doubles", "description": "output is exactly twice the input"},
        ],
        "test_inputs": test_inputs,
    })


def _judge_json(passed: bool, reason="checked") -> str:
    return json.dumps({
        "verdicts": [{"id": "doubles", "passed": passed, "reason": reason}],
        "diagnosis": "" if passed else "the d node does not double",
        "suggestions": "" if passed else "fix the d node's callable",
    })


# ── engine ──────────────────────────────────────────────────────────────────────

async def test_derive_acceptance_parses_and_fails_safe():
    provider = ScriptedProvider(turns=[(_plan_json(x=21), [])])
    plan = await derive_acceptance(provider, "double a number", "name: doubler")
    assert [c.id for c in plan.criteria] == ["doubles"]
    assert plan.test_inputs == {"x": 21}

    # Garbage output → fail-safe single criterion from the intent.
    provider = ScriptedProvider(turns=[("not json at all", [])])
    plan = await derive_acceptance(provider, "double a number", "name: doubler")
    assert len(plan.criteria) == 1 and plan.criteria[0].id == "fulfils_intent"


async def test_verify_passes_on_clean_run_and_passing_judge(tmp_path, kb):
    provider = ScriptedProvider(turns=[(_judge_json(True), [])])
    session = _session(tmp_path, kb)
    session.stage()
    plan = AcceptancePlan.model_validate(json.loads(_plan_json(x=4)))
    report = await verify_workflow(
        provider, intent=session.intent,
        package_dir=session.staging_root / session.name, plan=plan,
    )
    assert report.run_ok and report.passed
    assert report.verdicts[0]["passed"] is True
    d_node = next(n for n in report.node_summaries if n["id"] == "d")
    assert d_node["output"] == "8"


async def test_verify_fails_closed_when_judge_omits_criterion(tmp_path, kb):
    provider = ScriptedProvider(turns=[(json.dumps({"verdicts": []}), [])])
    session = _session(tmp_path, kb)
    session.stage()
    plan = AcceptancePlan.model_validate(json.loads(_plan_json(x=4)))
    report = await verify_workflow(
        provider, intent=session.intent,
        package_dir=session.staging_root / session.name, plan=plan,
    )
    assert report.run_ok and not report.passed
    assert "did not rule" in report.verdicts[0]["reason"]


async def test_verify_crashed_run_skips_judge_and_diagnoses(tmp_path, kb):
    provider = ScriptedProvider(turns=[])  # judge must NOT be called
    session = _session(tmp_path, kb)
    session.nodes = [{"id": "d", "kind": "function", "callable": f"{FN}._boom"}]
    session.stage()
    plan = AcceptancePlan.model_validate(json.loads(_plan_json(x=4)))
    report = await verify_workflow(
        provider, intent=session.intent,
        package_dir=session.staging_root / session.name, plan=plan,
    )
    assert not report.run_ok and not report.passed
    assert "kaboom" in report.diagnosis
    assert provider.calls == 0  # no judge on a crashed run


# ── test_workflow tool + staleness + gating ────────────────────────────────────

def _tool(session, name):
    return next(t for t in architect_tools(session) if t.name == name)


async def test_tool_passes_and_register_required_gates(tmp_path, kb):
    provider = ScriptedProvider(turns=[(_plan_json(x=3), []), (_judge_json(True), [])])
    session = _session(tmp_path, kb, provider=provider, mode="required")
    ctx = _ctx(tmp_path)

    # Required mode: register refuses before any verification.
    refused = await _tool(session, "register_workflow").run({}, ctx)
    assert refused.is_error and "requires a passing" in refused.content

    res = await _tool(session, "test_workflow").run({}, ctx)
    assert not res.is_error and "VERIFICATION PASSED" in res.content
    assert session.last_verification[0] is True

    ok = await _tool(session, "register_workflow").run({}, ctx)
    assert not ok.is_error and session.registered_path


async def test_tool_failure_blocks_register_and_edit_stales(tmp_path, kb):
    provider = ScriptedProvider(turns=[(_plan_json(x=3), []), (_judge_json(False), [])])
    session = _session(tmp_path, kb, provider=provider, mode="required")
    ctx = _ctx(tmp_path)

    res = await _tool(session, "test_workflow").run({}, ctx)
    assert res.is_error and "VERIFICATION FAILED" in res.content
    assert "fix the d node" in res.content  # suggestions surfaced to the agent

    refused = await _tool(session, "register_workflow").run({}, ctx)
    assert refused.is_error and "FAILED" in refused.content

    # Editing a node invalidates whatever verification state existed.
    session.last_verification = (True, "stale")
    upd = await _tool(session, "update_node").run(
        {"id": "d", "patch": {"writes": "out"}}, ctx)
    assert not upd.is_error
    assert session.last_verification is None


async def test_tool_not_offered_when_verification_off(tmp_path, kb):
    session = _session(tmp_path, kb, mode="off")
    names = {t.name for t in architect_tools(session)}
    assert "test_workflow" not in names
    # And register does not demand verification in off/encouraged modes.
    ok, _ = session.register()
    assert ok


async def test_tool_requires_structural_validity_first(tmp_path, kb):
    provider = ScriptedProvider(turns=[])
    session = _session(tmp_path, kb, provider=provider)
    session.nodes = [{"id": "d", "kind": "function", "callable": f"{FN}._double",
                      "depends_on": ["ghost"]}]
    res = await _tool(session, "test_workflow").run({}, _ctx(tmp_path))
    assert res.is_error and "not structurally valid" in res.content


# ── full agent loop with required verification (scripted) ──────────────────────

async def test_agent_closed_loop_required(tmp_path, kb):
    turns = [
        # agent turns and in-tool completions pop from the same script, in order:
        ("", [("set_workflow", {
            "name": "doubler", "description": "doubles x",
            "inputs": [{"name": "x", "type": "integer", "required": True}],
            "outputs": ["d"],
        })]),
        ("", [("add_node", {"node": {"id": "d", "kind": "function",
                                     "callable": f"{FN}._double"}})]),
        ("", [("test_workflow", {})]),
        (_plan_json(x=5), []),        # consumed by derive_acceptance
        (_judge_json(True), []),      # consumed by the judge
        ("", [("register_workflow", {})]),
        ("Done — registered.", []),
    ]
    agent = ArchitectAgent(
        ScriptedProvider(turns=turns),
        registry=WorkflowRegistry(workflows_dir=tmp_path / "registry"),
        staging_root=tmp_path / "staging",
        knowledge=kb,
        verify="required",
    )
    path = await agent.build("double a number")
    assert Path(path).exists()
