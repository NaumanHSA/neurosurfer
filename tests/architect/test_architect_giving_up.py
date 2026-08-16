"""When a build is allowed to give up, and when it must not.

The Architect must produce the same *kind* of outcome on every model. It did not:
`gpt-5.1` — a stronger model than the ones this was tuned on — turned a two-node
"summarise an article and title it" request into `WorkflowInfeasible`, while
`gpt-5-mini` and a local 9B built it. The transcript says why, and it is
structural rather than a quirk of one model:

    Repeated test_workflow runs show that, even with increasingly strict
    instructions, the model continues to introduce concepts not present in the
    source text, consistently failing the "no new information" acceptance
    criteria required for registration.

Three faults, each of which alone is survivable:

1. The model **writes the acceptance criteria it is then judged against**, and a
   more capable model writes a stricter bar. The judge fails closed, so an
   absolute criterion is close to unpassable.
2. The repair loop had **no bound** — `graph_runs` was counted and never used.
3. `declare_blocked` had **no guard**, so "I cannot satisfy my own judge" ended a
   build that had a complete, valid, grounded design in the session.

These tests fix the bar in place: a design nothing is missing from is never
infeasible, the loop always terminates, and an unfalsifiable criterion never
becomes the thing a build dies on.
"""

from __future__ import annotations

from pathlib import Path

import pytest

FN = "tests.architect.test_architect_giving_up"


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


def _session(tmp_path: Path, kb, **over):
    from neurosurfer.architect.agent import BuildSession
    from neurosurfer.graph.workflow.registry import WorkflowRegistry

    kwargs = dict(
        intent="summarise an article and write a title for the summary",
        staging_root=tmp_path / "staging",
        registry=WorkflowRegistry(workflows_dir=tmp_path / "registry"),
        knowledge=kb, verification_mode="off", review_mode="off",
    )
    kwargs.update(over)
    return BuildSession(**kwargs)


def _tool(session, name):
    from neurosurfer.architect.agent import architect_tools

    return next(t for t in architect_tools(session) if t.name == name)


async def _build(session, ctx, *node_ids: str):
    await _tool(session, "set_workflow").run({"name": "wf"}, ctx)
    for nid in node_ids:
        await _tool(session, "add_node").run(
            {"node": {"id": nid, "kind": "function", "callable": f"{FN}._shout"}}, ctx)
    await _tool(session, "set_outputs").run({"outputs": [node_ids[-1]]}, ctx)


# ── the guard on declare_blocked ────────────────────────────────────────────────


async def test_a_complete_grounded_design_cannot_be_declared_blocked(tmp_path, kb, ctx):
    """The exact exit `gpt-5.1` took. Nothing is missing, so nothing is blocked."""
    s = _session(tmp_path, kb)
    await _build(s, ctx, "a")

    result = await _tool(s, "declare_blocked").run(
        {"reason": "cannot guarantee the summary is free of invented detail"}, ctx)

    assert result.is_error
    assert s.blocked_reason is None, "a finished design must survive the attempt"
    # And it says what to do instead, because a model with no next move invents one.
    assert "register_workflow" in result.content


async def test_a_build_with_nothing_in_it_may_still_block(tmp_path, kb, ctx):
    """The planner refusing before any node exists is the original, valid use."""
    s = _session(tmp_path, kb)
    result = await _tool(s, "declare_blocked").run({"reason": "needs an Oracle login"}, ctx)
    assert not result.is_error
    assert s.blocked_reason == "needs an Oracle login"


async def test_an_ungrounded_design_may_still_block(tmp_path, kb, ctx):
    """A node that reaches outside the model with no tool is a real blocker.

    This is the Oracle transcript's shape: it built three nodes, validation was
    *not* clean, and blocking was the right call. The guard asks whether anything
    is missing — here the capability is — so it lets the block through.
    """
    s = _session(tmp_path, kb)
    await _tool(s, "set_workflow").run({"name": "wf"}, ctx)
    await _tool(s, "add_node").run(
        {"node": {"id": "a", "kind": "base",
                  "instructions": "Send an email to the support lead."}}, ctx)
    await _tool(s, "set_outputs").run({"outputs": ["a"]}, ctx)

    result = await _tool(s, "declare_blocked").run(
        {"reason": "no read-write database tool exists"}, ctx)
    assert not result.is_error, result.content
    assert s.blocked_reason


async def test_an_invalid_design_may_still_block(tmp_path, kb, ctx):
    """Structural failure is a real blocker too — the guard asks whether anything
    is *missing*, not whether the model feels stuck."""
    s = _session(tmp_path, kb)
    await _tool(s, "set_workflow").run({"name": "wf"}, ctx)
    await _tool(s, "add_node").run(
        {"node": {"id": "a", "kind": "function", "callable": f"{FN}._shout"}}, ctx)
    s.outputs = ["nonexistent_node"]      # dangling output: validation fails

    result = await _tool(s, "declare_blocked").run({"reason": "cannot finish"}, ctx)
    assert not result.is_error


# ── the repair loop terminates ──────────────────────────────────────────────────


async def test_the_repair_budget_lets_a_failing_build_register(tmp_path, kb, ctx):
    """Past the attempt limit a judged failure stops being a wall.

    Not a silent pass: the workflow registers *and* carries the loudest caveat
    there is. A workflow that runs, with a warning attached, is worth more to a
    user than no workflow and a paragraph about why.
    """
    s = _session(tmp_path, kb, verification_mode="required")
    await _build(s, ctx, "a")

    for i in range(s.max_verification_attempts):
        assert not s.verification_exhausted, f"exhausted early at attempt {i}"
        s.failed_verifications += 1
        s.record_verification(passed=False, rendered="judge said no", report=None, test_inputs=None)

    assert s.verification_exhausted
    ok, msg = s.pre_register()
    assert ok, msg

    caveat = s.verification_caveat()
    assert "FAILED verification" in caveat
    assert str(s.max_verification_attempts) in caveat


async def test_one_failure_is_not_exhaustion(tmp_path, kb, ctx):
    """The loop must still get to repair. One bad verdict blocks as it always did."""
    s = _session(tmp_path, kb, verification_mode="required")
    await _build(s, ctx, "a")
    s.failed_verifications += 1
    s.record_verification(passed=False, rendered="judge said no", report=None, test_inputs=None)

    assert not s.verification_exhausted
    ok, msg = s.pre_register()
    assert not ok and "verification FAILED" in msg


async def test_a_passing_verification_carries_no_caveat(tmp_path, kb, ctx):
    """Exhaustion is about the *last* verdict, not the count. A build that failed
    twice and then passed is a build that passed."""
    s = _session(tmp_path, kb, verification_mode="required")
    await _build(s, ctx, "a")
    s.failed_verifications = 99
    s.record_verification(passed=True, rendered="all criteria met", report=None, test_inputs=None)

    assert not s.verification_exhausted
    assert s.verification_caveat() == ""
    ok, _ = s.pre_register()
    assert ok


async def test_exhaustion_does_not_excuse_a_broken_design(tmp_path, kb, ctx):
    """The escape is from the *judge*, not from the structural gates. A graph that
    does not validate stays unregisterable however many attempts were spent."""
    s = _session(tmp_path, kb, verification_mode="required")
    await _build(s, ctx, "a")
    s.outputs = ["nonexistent_node"]
    s.failed_verifications = 99
    s.record_verification(passed=False, rendered="judge said no", report=None, test_inputs=None)

    ok, msg = s.pre_register()
    assert not ok and "validation failed" in msg


# ── a finished design must not die of bookkeeping ───────────────────────────────


async def test_a_run_that_ends_without_registering_still_hands_over_its_work(
    tmp_path, kb, ctx, monkeypatch
):
    """The loop can end with no terminal state — the model narrates past its
    nudges, or burns `max_turns`. A 9B did exactly that after building a good
    workflow, and the build raised, discarding it.

    The bar is unchanged: `pre_register` still has to pass. This only removes the
    requirement that the *model* be the one to ask.
    """
    from neurosurfer.graph.workflow.registry import WorkflowRegistry

    s = _session(tmp_path, kb, registry=WorkflowRegistry(workflows_dir=tmp_path / "reg"))
    await _build(s, ctx, "a")
    assert s.registered_path is None

    ok, msg = s.pre_register()
    assert ok, msg
    ok, _ = s.register()
    assert ok
    assert s.registered_path, "the design the run left behind is now on disk"


async def test_an_unfinished_design_is_not_salvaged(tmp_path, kb, ctx):
    """The salvage is not a way past the gates. A graph that does not validate
    stays unregistered, and the caller still gets its error."""
    s = _session(tmp_path, kb)
    await _tool(s, "set_workflow").run({"name": "wf"}, ctx)
    await _tool(s, "add_node").run(
        {"node": {"id": "a", "kind": "function", "callable": f"{FN}._shout"}}, ctx)
    s.outputs = ["nonexistent_node"]

    ok, _msg = s.pre_register()
    assert not ok
    assert s.registered_path is None
