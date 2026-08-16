"""Plan↔graph coverage and the design review (Architect V3, Phase 4).

Two mechanisms with deliberately different teeth, and the difference is the point:

- **Coverage** is decidable — a planned step either has a node or it doesn't — so
  it refuses. A step that quietly vanishes is part of the user's request going
  missing, invisible in a graph that otherwise validates.
- **The review** is a judgement, so it warns. Blocking a build on one weak model's
  opinion of another weak model's output is how V2's `verify="required"` stalled a
  9B model.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neurosurfer.architect.plan import PlanStep, WorkflowPlan
from neurosurfer.architect.review import ReviewReport, review_workflow

from ..fakes import ScriptedProvider

FN = "tests.architect.test_architect_review"


def _shout(**kwargs):
    return "hello"


@pytest.fixture(scope="module")
def kb():
    from neurosurfer.architect.knowledge import KnowledgeBase

    return KnowledgeBase()


@pytest.fixture()
def ctx(tmp_path: Path):
    from neurosurfer.tools.base import AutoApproveIOHandler, ToolContext

    return ToolContext(cwd=tmp_path, io=AutoApproveIOHandler())


def _session(tmp_path: Path, kb, *, provider=None, review="off"):
    from neurosurfer.architect.agent import BuildSession
    from neurosurfer.graph.workflow.registry import WorkflowRegistry

    return BuildSession(
        intent="do the thing",
        staging_root=tmp_path / "staging",
        registry=WorkflowRegistry(workflows_dir=tmp_path / "registry"),
        knowledge=kb, provider=provider, review_mode=review,
        # These exercise coverage and the review pass, not verification.
        verification_mode="off",
    )


def _tool(session, name):
    from neurosurfer.architect.agent import architect_tools

    return next(t for t in architect_tools(session) if t.name == name)


def _plan(*ids: str) -> WorkflowPlan:
    return WorkflowPlan(
        name="p",
        steps=[PlanStep(id=i, intent=f"do {i}") for i in ids],
        outputs=[ids[-1]],
    )


async def _build(session, ctx, *node_ids: str, **kw):
    await _tool(session, "set_workflow").run({"name": "cov"}, ctx)
    for nid in node_ids:
        await _tool(session, "add_node").run(
            {"node": {"id": nid, "kind": "function", "callable": f"{FN}._shout"},
             **kw}, ctx)


# ── coverage ────────────────────────────────────────────────────────────────────

async def test_matching_ids_pair_without_the_model_saying_anything(tmp_path, kb, ctx):
    """The builder is told to keep the plan's ids and overwhelmingly does;
    inferring that means it only has to speak up when it did something unusual."""
    s = _session(tmp_path, kb)
    s.plan = _plan("a", "b")
    await _build(s, ctx, "a", "b")
    missing, extra = s.plan_coverage()
    assert missing == [] and extra == []


async def test_a_dropped_step_refuses_registration(tmp_path, kb, ctx):
    s = _session(tmp_path, kb)
    s.plan = _plan("a", "b")
    await _build(s, ctx, "a")
    await _tool(s, "set_outputs").run({"outputs": ["a"]}, ctx)

    result = await _tool(s, "register_workflow").run({}, ctx)
    assert result.is_error
    assert "b — do b" in result.content
    assert "drop_plan_step" in result.content
    assert s.registered_path is None


async def test_dropping_a_step_explicitly_unblocks_it(tmp_path, kb, ctx):
    s = _session(tmp_path, kb)
    s.plan = _plan("a", "b")
    await _build(s, ctx, "a")
    await _tool(s, "set_outputs").run({"outputs": ["a"]}, ctx)

    dropped = await _tool(s, "drop_plan_step").run(
        {"step_id": "b", "reason": "node 'a' already covers it"}, ctx)
    assert not dropped.is_error
    assert s.dropped_steps["b"] == "node 'a' already covers it"

    result = await _tool(s, "register_workflow").run({}, ctx)
    assert not result.is_error, result.content


async def test_drop_needs_a_real_step_and_a_reason(tmp_path, kb, ctx):
    s = _session(tmp_path, kb)
    s.plan = _plan("a")
    drop = _tool(s, "drop_plan_step")
    assert (await drop.run({"step_id": "ghost", "reason": "x"}, ctx)).is_error
    assert (await drop.run({"step_id": "a", "reason": "  "}, ctx)).is_error


async def test_a_renamed_node_is_paired_by_citation(tmp_path, kb, ctx):
    s = _session(tmp_path, kb)
    s.plan = _plan("fetch_data")
    await _build(s, ctx, "load_the_data", plan_step_id="fetch_data")
    assert s.step_for_node("load_the_data") == "fetch_data"
    assert s.plan_coverage() == ([], [])


async def test_citing_a_step_that_does_not_exist_warns(tmp_path, kb, ctx):
    s = _session(tmp_path, kb)
    s.plan = _plan("a")
    result = await _tool(s, "add_node").run(
        {"node": {"id": "a", "kind": "function", "callable": f"{FN}._shout"},
         "plan_step_id": "ghost"}, ctx)
    assert "not a step in the plan" in result.content
    # …and the node still lands, paired by its own id.
    assert s.step_for_node("a") == "a"


async def test_an_unplanned_node_is_reported_but_does_not_block(tmp_path, kb, ctx):
    """The builder may legitimately split a step; judging that is the review's
    job, not a structural gate's."""
    s = _session(tmp_path, kb)
    s.plan = _plan("a")
    await _build(s, ctx, "a", "extra")
    await _tool(s, "set_outputs").run({"outputs": ["a", "extra"]}, ctx)

    missing, extra = s.plan_coverage()
    assert missing == [] and extra == ["extra"]
    assert not (await _tool(s, "register_workflow").run({}, ctx)).is_error


async def test_no_plan_means_no_coverage_check(tmp_path, kb, ctx):
    s = _session(tmp_path, kb)
    await _build(s, ctx, "a")
    await _tool(s, "set_outputs").run({"outputs": ["a"]}, ctx)
    assert s.plan_coverage() == ([], [])
    assert not (await _tool(s, "register_workflow").run({}, ctx)).is_error


async def test_progress_is_reported_against_the_plan(tmp_path, kb, ctx):
    said: list[str] = []
    s = _session(tmp_path, kb)
    s.notify = said.append
    s.plan = _plan("a", "b", "c")
    await _build(s, ctx, "a", "b")
    assert "[1/3 steps]" in said[1]
    assert "[2/3 steps]" in said[2]


# ── the review ──────────────────────────────────────────────────────────────────

def _review_json(ok=True, issues=(), summary="Does the thing.") -> str:
    return json.dumps({"ok": ok, "summary": summary, "issues": list(issues)})


async def test_a_clean_review_parses():
    report = await review_workflow(
        ScriptedProvider([(_review_json(), [])]),
        intent="x", graph_yaml="nodes: []")
    assert report.ok and not report.issues
    assert "Does the thing" in report.render()


async def test_issues_are_read_and_rendered():
    raw = _review_json(ok=False, issues=[
        {"node": "summarise", "problem": "It summarises when the request asked for "
         "the raw quotes", "fix": "Change its goal to extract quotes verbatim"},
    ])
    report = await review_workflow(ScriptedProvider([(raw, [])]),
                                   intent="x", graph_yaml="y")
    assert not report.ok
    assert report.issues[0].node == "summarise"
    assert "extract quotes verbatim" in report.render()


async def test_issues_override_a_contradictory_ok_flag():
    """Models set ok:true and then list problems often enough to need a rule:
    the issues are the evidence, the flag is a summary of it."""
    raw = _review_json(ok=True, issues=[{"problem": "the output is never produced"}])
    report = await review_workflow(ScriptedProvider([(raw, [])]),
                                   intent="x", graph_yaml="y")
    assert not report.ok


async def test_an_unusable_answer_is_inconclusive_not_approval():
    report = await review_workflow(ScriptedProvider([("no json", [])]),
                                   intent="x", graph_yaml="y")
    assert report.inconclusive
    assert "INCONCLUSIVE" in report.render()


async def test_a_reviewer_outage_does_not_fail_the_build():
    class _Down:
        async def complete(self, **_kw):
            raise RuntimeError("upstream is down")

    report = await review_workflow(_Down(), intent="x", graph_yaml="y")
    assert report.inconclusive and report.ok


# ── the review at the register gate ─────────────────────────────────────────────

async def test_warn_mode_registers_and_says_what_it_found(tmp_path, kb, ctx):
    raw = _review_json(ok=False, issues=[
        {"node": "a", "problem": "answers a narrower question", "fix": "widen it"}])
    s = _session(tmp_path, kb, provider=ScriptedProvider([(raw, [])]), review="warn")
    await _build(s, ctx, "a")
    await _tool(s, "set_outputs").run({"outputs": ["a"]}, ctx)

    result = await _tool(s, "register_workflow").run({}, ctx)
    assert not result.is_error, "a judgement must not block a build"
    assert s.registered_path is not None
    assert "narrower question" in result.content


async def test_required_mode_refuses_until_the_design_changes(tmp_path, kb, ctx):
    raw = _review_json(ok=False, issues=[{"problem": "wrong question"}])
    s = _session(tmp_path, kb, provider=ScriptedProvider([(raw, [])]),
                 review="required")
    await _build(s, ctx, "a")
    await _tool(s, "set_outputs").run({"outputs": ["a"]}, ctx)

    result = await _tool(s, "register_workflow").run({}, ctx)
    assert result.is_error and "wrong question" in result.content
    assert s.registered_path is None


async def test_a_review_is_reused_for_an_unchanged_graph(tmp_path, kb, ctx):
    provider = ScriptedProvider([(_review_json(), [])])
    s = _session(tmp_path, kb, provider=provider, review="warn")
    await _build(s, ctx, "a")
    await _tool(s, "set_outputs").run({"outputs": ["a"]}, ctx)

    await _tool(s, "register_workflow").run({}, ctx)
    await _tool(s, "register_workflow").run({}, ctx)
    assert provider.calls == 1, "the reviewer must not be asked twice about one graph"


async def test_editing_the_graph_stales_the_review(tmp_path, kb, ctx):
    s = _session(tmp_path, kb, provider=ScriptedProvider([]), review="off")
    await _build(s, ctx, "a")
    s.review = (s.graph_fingerprint(), ReviewReport(ok=True))
    assert s.last_review is not None

    await _tool(s, "add_node").run(
        {"node": {"id": "b", "kind": "function", "callable": f"{FN}._shout",
                  "depends_on": ["a"]}}, ctx)
    assert s.last_review is None, "a changed design has not been reviewed"


async def test_the_free_gates_run_before_the_reviewer_is_paid(tmp_path, kb, ctx):
    """A graph about to be refused for a missing step must not cost a review."""
    provider = ScriptedProvider([])          # any call would raise IndexError
    s = _session(tmp_path, kb, provider=provider, review="warn")
    s.plan = _plan("a", "b")
    await _build(s, ctx, "a")
    await _tool(s, "set_outputs").run({"outputs": ["a"]}, ctx)

    result = await _tool(s, "register_workflow").run({}, ctx)
    assert result.is_error and "b — do b" in result.content
    assert provider.calls == 0


async def test_review_off_never_calls_the_reviewer(tmp_path, kb, ctx):
    provider = ScriptedProvider([])
    s = _session(tmp_path, kb, provider=provider, review="off")
    await _build(s, ctx, "a")
    await _tool(s, "set_outputs").run({"outputs": ["a"]}, ctx)
    assert not (await _tool(s, "register_workflow").run({}, ctx)).is_error
    assert provider.calls == 0


def test_the_build_record_carries_the_review():
    from neurosurfer.app.server.architect_builds.manager import ArchitectManager
    from neurosurfer.app.server.architect_builds.store import BuildRecord

    class _Agent:
        def __init__(self) -> None:
            self.session = type("S", (), {"last_review": None})()

    rec, agent = BuildRecord(intent="x"), _Agent()
    ArchitectManager._capture_review(rec, agent)
    assert rec.review is None                       # nothing to say yet

    agent.session.last_review = ReviewReport(
        ok=False, issues=[], summary="s")
    ArchitectManager._capture_review(rec, agent)
    assert rec.review["summary"] == "s"
    assert [e["type"] for e in rec.events] == ["review"]


# ── the registry must match what the session says ───────────────────────────────


async def test_an_edit_after_registering_reaches_the_registry(tmp_path, kb, ctx):
    """A fix applied after `register_workflow` must not be thrown away.

    The defect this covers, from a real `gpt-5.6` transcript: with `review="warn"`
    the reviewer found that a node titled the *article* where the request asked for
    a title of the *summary*. Registration had already written the package, the
    tool reported the finding as advisory, the model dutifully patched the node —
    and the registered `graph.yaml` still carried the flawed prompt, because
    `register()` snapshots to disk and later edits live only in the session.

    The registered artifact is the deliverable. It tracks the session, or the
    review is advice nobody can act on.
    """
    import yaml

    s = _session(tmp_path, kb)
    await _build(s, ctx, "a")
    await _tool(s, "set_outputs").run({"outputs": ["a"]}, ctx)
    assert not (await _tool(s, "register_workflow").run({}, ctx)).is_error

    # The model reads the review and fixes the node — after registering. This is
    # `update_node`, which is exactly what the transcript shows it calling.
    res = await _tool(s, "update_node").run(
        {"id": "a", "patch": {"description": "the fix the reviewer asked for"}}, ctx)
    assert not res.is_error, res.content

    changed, msg = s.sync_registration()
    assert changed, msg

    on_disk = yaml.safe_load((Path(s.registered_path) / "graph.yaml").read_text())
    assert on_disk["nodes"][0]["description"] == "the fix the reviewer asked for"


async def test_registering_an_unchanged_design_rewrites_nothing(tmp_path, kb, ctx):
    """Only a *changed* design is re-saved. Same fingerprint, no write — so the
    terminal path does not rewrite the package on every build that ends cleanly."""
    s = _session(tmp_path, kb)
    await _build(s, ctx, "a")
    await _tool(s, "set_outputs").run({"outputs": ["a"]}, ctx)
    await _tool(s, "register_workflow").run({}, ctx)

    changed, msg = s.sync_registration()
    assert not changed and msg == ""


async def test_a_warned_review_does_not_tell_the_model_it_may_finish(tmp_path, kb, ctx):
    """The message that caused the defect said both things at once.

    `register()` ends with "The build is complete — you may finish now", and the
    review note appended "consider fixing and re-registering". A real transcript
    has a model take the shorter road: it patched the node the reviewer named and
    then stopped, leaving the flaw in the registered copy. One instruction, and
    `sync_registration` is what makes the fix land.
    """
    raw = _review_json(ok=False, issues=[
        {"node": "a", "problem": "titles the article, not the summary",
         "fix": "say summary"}])
    s = _session(tmp_path, kb, provider=ScriptedProvider([(raw, [])]), review="warn")
    await _build(s, ctx, "a")
    await _tool(s, "set_outputs").run({"outputs": ["a"]}, ctx)

    result = await _tool(s, "register_workflow").run({}, ctx)
    assert not result.is_error, result.content
    assert "you may finish now" not in result.content
    assert "titles the article" in result.content
    # And a clean review keeps the original ending — nothing to act on.
    s2 = _session(tmp_path / "b", kb, provider=ScriptedProvider([(_review_json(), [])]),
                  review="warn")
    await _build(s2, ctx, "a")
    await _tool(s2, "set_outputs").run({"outputs": ["a"]}, ctx)
    clean = await _tool(s2, "register_workflow").run({}, ctx)
    assert "you may finish now" in clean.content
