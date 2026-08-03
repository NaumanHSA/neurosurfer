"""The Planner: a plan is an artifact (Architect V3, Phase 3).

The planner is a single structured call, so it is tested the way a parser is —
feed it what a real model actually emits (preambles, code fences, missing fields,
invented kinds) and assert it comes back with something the builder can follow.

The load-bearing field is `is_external`. Everything else in a plan is a
convenience; that one is what stops "read the file" becoming a `base` node.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neurosurfer.architect.plan import PlanStep, WorkflowPlan
from neurosurfer.architect.planner import PlannerAgent, resolve_plan

from ..fakes import ScriptedProvider


def _plan_json(**over) -> str:
    payload = {
        "name": "file_report",
        "description": "Read a file and report on it.",
        "inputs": [{"name": "file_path", "type": "string", "required": True,
                    "description": "Path to the file"}],
        "steps": [
            {"id": "read_file", "intent": "Read the file's contents.",
             "kind_hint": "tool", "depends_on": [], "produces": "file text",
             "is_external": True, "needed_capability": "read a file from disk"},
            {"id": "analyse", "intent": "Analyse the content deeply.",
             "kind_hint": "base", "depends_on": ["read_file"],
             "produces": "insights", "is_external": False},
            {"id": "report", "intent": "Write a brief report.",
             "kind_hint": "base", "depends_on": ["analyse"],
             "produces": "the report", "is_external": False},
        ],
        "outputs": ["report"],
        "open_questions": [],
    }
    payload.update(over)
    return json.dumps(payload)


# ── parsing what models actually emit ───────────────────────────────────────────

async def test_a_clean_plan_parses():
    plan = await PlannerAgent(ScriptedProvider([(_plan_json(), [])])).plan("read a file")
    assert plan.name == "file_report"
    assert [s.id for s in plan.steps] == ["read_file", "analyse", "report"]
    assert [s.id for s in plan.external_steps] == ["read_file"]
    assert plan.outputs == ["report"]


async def test_a_preamble_and_a_code_fence_survive():
    """Small models do not honour "STRICT JSON only"."""
    noisy = (
        "Thinking… I should return an object like {\"steps\": []} first.\n"
        "```json\n" + _plan_json() + "\n```\nHope that helps!"
    )
    plan = await PlannerAgent(ScriptedProvider([(noisy, [])])).plan("read a file")
    assert len(plan.steps) == 3, "must take the LAST parseable object, not the example"


async def test_unparseable_output_retries_once_then_succeeds():
    provider = ScriptedProvider([("no json here", []), (_plan_json(), [])])
    plan = await PlannerAgent(provider).plan("read a file")
    assert len(plan.steps) == 3
    assert provider.calls == 2


async def test_two_failures_degrade_to_no_plan_rather_than_raising():
    """An unparseable plan should cost the build its planning advantage, not
    its life — the builder still works the way it did before this existed."""
    provider = ScriptedProvider([("nope", []), ("still nope", [])])
    plan = await PlannerAgent(provider).plan("read a file")
    assert plan.steps == []
    assert provider.calls == 2


# ── normalisation: what models get slightly wrong ───────────────────────────────

async def test_invented_kinds_fall_back_to_base():
    bad = _plan_json(steps=[{"id": "a", "intent": "x", "kind_hint": "megatron"}])
    plan = await PlannerAgent(ScriptedProvider([(bad, [])])).plan("x")
    assert plan.steps[0].kind_hint == "base"


async def test_duplicate_and_empty_ids_are_repaired():
    bad = _plan_json(steps=[
        {"id": "a", "intent": "one"},
        {"id": "a", "intent": "two"},
        {"id": "", "intent": "three"},
    ])
    plan = await PlannerAgent(ScriptedProvider([(bad, [])])).plan("x")
    assert len({s.id for s in plan.steps}) == 3


async def test_dangling_depends_on_is_dropped():
    bad = _plan_json(steps=[
        {"id": "a", "intent": "one"},
        {"id": "b", "intent": "two", "depends_on": ["a", "ghost", "b"]},
    ])
    plan = await PlannerAgent(ScriptedProvider([(bad, [])])).plan("x")
    assert plan.step("b").depends_on == ["a"]


async def test_a_named_capability_makes_a_step_external_whatever_the_flag_says():
    """Models write the prose and forget the boolean far more often than the
    reverse."""
    bad = _plan_json(steps=[{"id": "a", "intent": "read it", "is_external": False,
                             "needed_capability": "read a file from disk"}])
    plan = await PlannerAgent(ScriptedProvider([(bad, [])])).plan("x")
    assert plan.steps[0].is_external


async def test_missing_outputs_fall_back_to_the_last_step():
    bad = _plan_json(outputs=[])
    plan = await PlannerAgent(ScriptedProvider([(bad, [])])).plan("x")
    assert plan.outputs == ["report"]


async def test_outputs_naming_a_ghost_step_are_dropped():
    plan = await PlannerAgent(ScriptedProvider([(_plan_json(outputs=["ghost"]), [])])).plan("x")
    assert plan.outputs == ["report"]


# ── the resolution pass ─────────────────────────────────────────────────────────

def test_resolution_attaches_a_real_tool_to_an_external_step():
    plan = WorkflowPlan(steps=[
        PlanStep(id="read_file", intent="Read it.", is_external=True,
                 needed_capability="read a file from disk"),
        PlanStep(id="analyse", intent="Analyse it.", depends_on=["read_file"]),
    ])
    resolved = resolve_plan(plan)
    step = resolved.step("read_file")
    assert step.resolution["status"] == "have"
    assert step.tool_names == ["read_file"]
    assert step.resolved


def test_internal_steps_are_resolved_by_definition():
    plan = WorkflowPlan(steps=[PlanStep(id="a", intent="Summarise it.")])
    assert resolve_plan(plan).step("a").resolved
    assert plan.unresolved_steps == []


def test_a_capability_nothing_provides_leaves_the_step_unresolved(monkeypatch):
    from neurosurfer.mcp import registry as mcp_registry

    mcp_registry._cache.clear()
    monkeypatch.setattr(mcp_registry, "registry_get",
                        lambda *a, **k: {"servers": []})
    plan = WorkflowPlan(steps=[
        PlanStep(id="mail", intent="Read my inbox.", is_external=True,
                 needed_capability="read an email inbox"),
    ])
    resolved = resolve_plan(plan)
    assert not resolved.step("mail").resolved
    assert [s.id for s in resolved.unresolved_steps] == ["mail"]


def test_composition_marked_external_is_corrected(monkeypatch):
    """The planner sometimes flags "draft an email" as external. The ladder
    knows better, and the plan should end up honest."""
    plan = WorkflowPlan(steps=[
        PlanStep(id="draft", intent="Draft the reply.", is_external=True,
                 needed_capability="draft an email"),
    ])
    resolved = resolve_plan(plan)
    step = resolved.step("draft")
    assert not step.is_external
    assert step.needed_capability == ""
    assert step.resolved


def test_a_failing_lookup_does_not_kill_the_plan(monkeypatch):
    import neurosurfer.architect.capability as cap

    monkeypatch.setattr(cap, "resolve_capability",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    plan = WorkflowPlan(steps=[
        PlanStep(id="a", intent="x", is_external=True, needed_capability="y"),
    ])
    resolved = resolve_plan(plan)
    assert resolved.step("a").resolution["status"] == "none"
    assert not resolved.step("a").resolved


# ── rendering ───────────────────────────────────────────────────────────────────

def test_render_shows_the_capability_and_the_tool_that_answers_it():
    plan = resolve_plan(WorkflowPlan(name="x", steps=[
        PlanStep(id="read_file", intent="Read it.", kind_hint="tool",
                 is_external=True, needed_capability="read a file from disk"),
    ]))
    rendered = plan.render()
    assert "read a file from disk" in rendered
    assert "[have]" in rendered
    assert "`read_file`" in rendered


# ── the agent stops at the plan ─────────────────────────────────────────────────

@pytest.fixture()
def kb():
    from neurosurfer.architect.knowledge import KnowledgeBase

    return KnowledgeBase()


async def test_a_build_blocks_at_the_plan_without_writing_a_node(tmp_path: Path, kb,
                                                                 monkeypatch):
    """The cheapest place to learn a build cannot happen."""
    from neurosurfer.architect import ArchitectAgent, WorkflowInfeasible
    from neurosurfer.graph.workflow.registry import WorkflowRegistry
    from neurosurfer.mcp import registry as mcp_registry

    mcp_registry._cache.clear()
    monkeypatch.setattr(mcp_registry, "registry_get", lambda *a, **k: {"servers": []})

    plan = json.dumps({
        "name": "mail", "description": "Read mail.",
        "inputs": [],
        "steps": [{"id": "inbox", "intent": "Read my inbox.", "kind_hint": "react",
                   "is_external": True, "needed_capability": "read an email inbox"}],
        "outputs": ["inbox"],
    })
    agent = ArchitectAgent(
        ScriptedProvider([(plan, [])]),
        registry=WorkflowRegistry(workflows_dir=tmp_path / "registry"),
        staging_root=tmp_path / "staging",
        knowledge=kb,
        verify="off",
    )
    with pytest.raises(WorkflowInfeasible) as excinfo:
        await agent.build("read my email")

    assert "read an email inbox" in str(excinfo.value)
    assert agent.session.nodes == [], "nothing should have been built"
    assert excinfo.value.requirements[0]["step"] == "inbox"


async def test_an_approver_can_override_a_blocked_plan(tmp_path: Path, kb, monkeypatch):
    from neurosurfer.architect import ArchitectAgent
    from neurosurfer.graph.workflow.registry import WorkflowRegistry
    from neurosurfer.mcp import registry as mcp_registry

    mcp_registry._cache.clear()
    monkeypatch.setattr(mcp_registry, "registry_get", lambda *a, **k: {"servers": []})

    plan = json.dumps({
        "name": "mail", "steps": [
            {"id": "inbox", "intent": "Read my inbox.", "is_external": True,
             "needed_capability": "read an email inbox"}],
        "outputs": ["inbox"],
    })
    asked: list = []

    agent = ArchitectAgent(
        # After the plan, the builder gets one turn and stops — enough to prove
        # the build was allowed to start.
        ScriptedProvider([(plan, []), ("thinking about it", [])]),
        registry=WorkflowRegistry(workflows_dir=tmp_path / "registry"),
        staging_root=tmp_path / "staging",
        knowledge=kb,
        verify="off",
        approve_plan=lambda p: (asked.append(p) or True),
        max_turns=1,
    )
    with pytest.raises(RuntimeError, match="finished without registering"):
        await agent.build("read my email")
    assert len(asked) == 1, "the approver was asked"
    assert agent.session.plan is not None


async def test_the_plan_reaches_the_builder_prompt(tmp_path: Path, kb):
    """The single biggest weak-model lever: the builder is TOLD which tool to
    attach rather than asked to remember one."""
    from neurosurfer.architect import ArchitectAgent

    plan = resolve_plan(WorkflowPlan(name="x", steps=[
        PlanStep(id="read_file", intent="Read it.", kind_hint="tool",
                 is_external=True, needed_capability="read a file from disk"),
    ]))
    prompt = ArchitectAgent._render_prompt("read a file", None, plan)
    assert "BUILD THIS PLAN" in prompt
    assert "node `read_file`: use tools ['read_file']" in prompt


def test_no_plan_leaves_the_prompt_as_it_was():
    from neurosurfer.architect import ArchitectAgent

    prompt = ArchitectAgent._render_prompt("do a thing", None, None)
    assert "BUILD THIS PLAN" not in prompt


# ── the plan API ────────────────────────────────────────────────────────────────

@pytest.fixture()
def plan_client(tmp_path: Path, monkeypatch):
    """A gateway whose architect provider returns one canned plan."""
    from fastapi.testclient import TestClient

    from neurosurfer.app.server.architect_builds.manager import ArchitectManager
    from neurosurfer.app.server.gateway import NeurosurferServer
    from neurosurfer.graph.workflow.registry import WorkflowRegistry
    from neurosurfer.mcp import registry as mcp_registry

    mcp_registry._cache.clear()
    monkeypatch.setattr(mcp_registry, "registry_get", lambda *a, **k: {"servers": []})

    server = NeurosurferServer(app_name="test", api_keys=None)
    server.architect_manager = ArchitectManager(
        ScriptedProvider([(_plan_json(), [])]),
        registry=WorkflowRegistry(workflows_dir=tmp_path / "registry"),
    )
    return TestClient(server.create_app())


def test_plan_endpoint_returns_steps_and_their_resolutions(plan_client):
    body = plan_client.post("/v1/architect/plans", json={"intent": "read a file"}).json()

    steps = body["plan"]["steps"]
    assert [s["id"] for s in steps] == ["read_file", "analyse", "report"]
    external = next(s for s in steps if s["is_external"])
    assert external["resolution"]["status"] == "have"
    assert external["resolution"]["tools"][0]["name"] == "read_file"
    assert body["unresolved"] == []
    assert "read a file from disk" in body["rendered"]


def test_plan_endpoint_needs_an_intent(plan_client):
    assert plan_client.post("/v1/architect/plans", json={}).status_code == 422


def test_plan_endpoint_does_not_register_anything(plan_client):
    """Planning is meant to be the cheap question — it must not stage or register
    a package as a side effect."""
    plan_client.post("/v1/architect/plans", json={"intent": "read a file"})
    names = {w["name"] for w in plan_client.get("/v1/workflows").json()["workflows"]}
    assert "file_report" not in names
    assert plan_client.get("/v1/architect/builds").json()["builds"] == []


# ── kind_hint must be consistent with is_external ───────────────────────────────

@pytest.mark.parametrize("kind", ["tool", "react"])
async def test_tool_kinds_on_an_internal_step_are_demoted(kind):
    """`tool` and `react` exist to call tools. On a step that reaches nothing
    they describe a node with an empty toolbelt — which Phase 1 rejects."""
    bad = _plan_json(steps=[{"id": "a", "intent": "Summarise it.",
                             "kind_hint": kind, "is_external": False}])
    plan = await PlannerAgent(ScriptedProvider([(bad, [])])).plan("x")
    assert plan.steps[0].kind_hint == "base"


@pytest.mark.parametrize("kind", ["tool", "react"])
async def test_tool_kinds_survive_on_an_external_step(kind):
    good = _plan_json(steps=[{"id": "a", "intent": "Read it.", "kind_hint": kind,
                              "is_external": True,
                              "needed_capability": "read a file from disk"}])
    plan = await PlannerAgent(ScriptedProvider([(good, [])])).plan("x")
    assert plan.steps[0].kind_hint == kind


async def test_control_flow_kinds_are_preserved():
    """The planner owns the branching decision now — normalisation must not
    quietly flatten a router into a base node."""
    branching = _plan_json(steps=[
        {"id": "triage", "intent": "Route by urgency.", "kind_hint": "router"},
        {"id": "escalate", "intent": "Escalate it.", "depends_on": ["triage"]},
        {"id": "standard", "intent": "Reply normally.", "depends_on": ["triage"]},
    ], outputs=["escalate", "standard"])
    plan = await PlannerAgent(ScriptedProvider([(branching, [])])).plan("x")
    assert plan.step("triage").kind_hint == "router"
    assert plan.step("escalate").depends_on == ["triage"]


# ── approve and edit ────────────────────────────────────────────────────────────

def _agent(tmp_path: Path, kb, turns, **kw):
    from neurosurfer.architect import ArchitectAgent
    from neurosurfer.graph.workflow.registry import WorkflowRegistry

    return ArchitectAgent(
        ScriptedProvider(turns),
        registry=WorkflowRegistry(workflows_dir=tmp_path / "registry"),
        staging_root=tmp_path / "staging",
        knowledge=kb, verify="off", max_turns=1, **kw,
    )


async def test_a_healthy_plan_is_still_reviewed(tmp_path: Path, kb):
    """The gate is not only for broken plans — a wrong step costs one edit here
    and a whole rebuild once it is nodes."""
    seen: list = []
    agent = _agent(tmp_path, kb, [(_plan_json(), []), ("...", [])],
                   approve_plan=lambda p: (seen.append(p) or p))
    with pytest.raises(RuntimeError, match="finished without registering"):
        await agent.build("read a file")
    assert len(seen) == 1
    assert not seen[0].unresolved_steps, "this plan was fine, and was reviewed anyway"


async def test_rejecting_the_plan_builds_nothing(tmp_path: Path, kb):
    from neurosurfer.architect import WorkflowInfeasible

    agent = _agent(tmp_path, kb, [(_plan_json(), [])],
                   approve_plan=lambda _p: None)
    with pytest.raises(WorkflowInfeasible, match="not accepted"):
        await agent.build("read a file")
    assert agent.session.nodes == []


async def test_an_edited_plan_is_built_and_re_resolved(tmp_path: Path, kb):
    """The reviewer changed a step's capability, so the old lookup is void."""
    def _edit(plan):
        edited = plan.model_dump(mode="json")
        edited["steps"] = [s for s in edited["steps"] if s["id"] != "analyse"]
        edited["steps"].append({
            "id": "save_it", "intent": "Write the report out.", "kind_hint": "tool",
            "depends_on": ["report"], "is_external": True,
            "needed_capability": "write a file to disk",
        })
        return edited

    agent = _agent(tmp_path, kb, [(_plan_json(), []), ("...", [])],
                   approve_plan=_edit)
    with pytest.raises(RuntimeError, match="finished without registering"):
        await agent.build("read a file")

    plan = agent.session.plan
    assert [s.id for s in plan.steps] == ["read_file", "report", "save_it"]
    # The added step was looked up, not taken on trust.
    assert plan.step("save_it").tool_names == ["write_file"]
    # And the removed step's dangling edge went with it.
    assert plan.step("report").depends_on == []


async def test_an_unusable_edit_falls_back_to_the_original(tmp_path: Path, kb):
    agent = _agent(tmp_path, kb, [(_plan_json(), []), ("...", [])],
                   approve_plan=lambda _p: {"steps": "not a list"})
    with pytest.raises(RuntimeError, match="finished without registering"):
        await agent.build("read a file")
    assert [s.id for s in agent.session.plan.steps] == ["read_file", "analyse", "report"]


async def test_accepting_an_unresolvable_plan_is_an_explicit_override(
    tmp_path: Path, kb, monkeypatch
):
    """The reviewer was shown exactly which steps and what they'd need."""
    from neurosurfer.mcp import registry as mcp_registry

    mcp_registry._cache.clear()
    monkeypatch.setattr(mcp_registry, "registry_get", lambda *a, **k: {"servers": []})

    plan = json.dumps({"name": "mail", "steps": [
        {"id": "inbox", "intent": "Read my inbox.", "is_external": True,
         "needed_capability": "read an email inbox"}], "outputs": ["inbox"]})
    agent = _agent(tmp_path, kb, [(plan, []), ("...", [])],
                   approve_plan=lambda p: p)
    # It proceeds to the builder rather than blocking — the human said so.
    with pytest.raises(RuntimeError, match="finished without registering"):
        await agent.build("read my email")
    assert agent.session.plan.unresolved_steps


async def test_no_reviewer_still_blocks_an_unresolvable_plan(tmp_path: Path, kb,
                                                             monkeypatch):
    from neurosurfer.architect import WorkflowInfeasible
    from neurosurfer.mcp import registry as mcp_registry

    mcp_registry._cache.clear()
    monkeypatch.setattr(mcp_registry, "registry_get", lambda *a, **k: {"servers": []})

    plan = json.dumps({"name": "mail", "steps": [
        {"id": "inbox", "intent": "Read my inbox.", "is_external": True,
         "needed_capability": "read an email inbox"}], "outputs": ["inbox"]})
    agent = _agent(tmp_path, kb, [(plan, [])])
    with pytest.raises(WorkflowInfeasible, match="read an email inbox"):
        await agent.build("read my email")


# ── building a plan somebody else made ──────────────────────────────────────────

async def test_a_supplied_plan_skips_planning(tmp_path: Path, kb):
    """How a plan reviewed in one request gets built by the next one without
    being re-invented in between."""
    provider_turns = [("...", [])]           # no planning call in the script
    agent = _agent(tmp_path, kb, provider_turns)
    supplied = {
        "name": "supplied", "steps": [
            {"id": "read_file", "intent": "Read it.", "kind_hint": "tool",
             "is_external": True, "needed_capability": "read a file from disk"}],
        "outputs": ["read_file"],
    }
    with pytest.raises(RuntimeError, match="finished without registering"):
        await agent.build("read a file", plan=supplied)

    assert agent.session.plan.name == "supplied"
    # Re-resolved on arrival: a resolution that travelled over HTTP describes
    # whatever the catalog looked like when it was made, not now.
    assert agent.session.plan.step("read_file").tool_names == ["read_file"]


async def test_a_supplied_plan_must_have_steps(tmp_path: Path, kb):
    agent = _agent(tmp_path, kb, [])
    with pytest.raises(ValueError, match="at least one step"):
        await agent.build("x", plan={"name": "empty", "steps": []})


def test_the_build_record_follows_an_edited_plan():
    """Found live: the record captured the plan once, so a build whose plan was
    edited kept showing the version the model proposed — attributing the user's
    design to the agent and hiding the step they added."""
    from neurosurfer.app.server.architect_builds.manager import ArchitectManager
    from neurosurfer.app.server.architect_builds.store import BuildRecord

    class _Agent:
        def __init__(self) -> None:
            self.session = type("S", (), {"plan": None})()

    rec, agent = BuildRecord(intent="x"), _Agent()
    agent.session.plan = WorkflowPlan(name="p", steps=[PlanStep(id="a", intent="one")])
    ArchitectManager._capture_plan(rec, agent)
    assert [s["id"] for s in rec.plan["steps"]] == ["a"]

    # Unchanged: no second event.
    ArchitectManager._capture_plan(rec, agent)
    assert sum(1 for e in rec.events if e["type"] == "plan") == 1

    agent.session.plan = WorkflowPlan(
        name="p", steps=[PlanStep(id="a", intent="one"), PlanStep(id="b", intent="two")]
    )
    ArchitectManager._capture_plan(rec, agent)
    assert [s["id"] for s in rec.plan["steps"]] == ["a", "b"]
    assert sum(1 for e in rec.events if e["type"] == "plan") == 2
