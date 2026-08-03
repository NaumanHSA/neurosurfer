"""V3 Phase 5a — verification fixtures: the workflow gets a world to run in.

The complaint this closes: a workflow that reads a file was verified by putting
the string "Sample input text for testing the workflow end to end." into its
`file_path`, so it failed on a file nobody had created. These tests pin that a
real file gets made, that the run happens where the file is, and that a source
path with nothing behind it fails loudly and says what to add.

Hermetic: `tool`-kind nodes calling the `read_file` builtin (no LLM inside the
workflow) + the ScriptedProvider playing the deriver/judge with canned JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neurosurfer.architect.agent import (
    AcceptancePlan,
    BuildSession,
    Fixture,
    architect_tools,
    derive_acceptance,
    verify_workflow,
)
from neurosurfer.architect.agent.verify import _backfill_inputs, _path_role
from neurosurfer.architect.knowledge import KnowledgeBase
from neurosurfer.graph.workflow.registry import WorkflowRegistry
from neurosurfer.tools.base import AutoApproveIOHandler, ToolContext

from ..fakes import ScriptedProvider

ARTICLE = (
    "The city council voted 7-2 on Tuesday to fund the harbour redevelopment.\n"
    "Opponents argued the projected tourism revenue rests on optimistic\n"
    "assumptions about ferry traffic that the consultants never modelled.\n"
)
SETUP = f"from pathlib import Path\nPath('article.txt').write_text({ARTICLE!r})\n"


@pytest.fixture(scope="module")
def kb():
    return KnowledgeBase()


def _reader_session(tmp_path: Path, kb, provider=None, mode="encouraged") -> BuildSession:
    """A workflow that really opens a file — the shape from the bug report."""
    s = BuildSession(
        intent="read a file and report what is in it",
        staging_root=tmp_path / "staging",
        registry=WorkflowRegistry(workflows_dir=tmp_path / "registry"),
        knowledge=kb,
        provider=provider,
        verification_mode=mode,
    )
    s.name = "reader"
    s.inputs = [{"name": "file_path", "type": "string", "required": True}]
    s.nodes = [{"id": "read", "kind": "tool", "tools": ["read_file"],
                "tool_args": {"path": "{file_path}"}}]
    s.outputs = ["read"]
    return s


def _ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, io=AutoApproveIOHandler())


def _judge_json(passed: bool) -> str:
    return json.dumps({
        "verdicts": [{"id": "reports_content", "passed": passed, "reason": "checked"}],
        "diagnosis": "" if passed else "the node did not read the file",
        "suggestions": "" if passed else "attach read_file",
    })


def _plan(**kw) -> AcceptancePlan:
    return AcceptancePlan.model_validate({
        "criteria": [{"id": "reports_content", "description": "output reflects the file"}],
        **kw,
    })


# ── which inputs are paths ──────────────────────────────────────────────────────

@pytest.mark.parametrize(("spec", "expected"), [
    ({"name": "file_path", "type": "string"}, "source"),
    ({"name": "path", "type": "string"}, "source"),
    ({"name": "input_dir", "type": "string"}, "source"),
    ({"name": "filename", "type": "string"}, "source"),
    ({"name": "doc", "type": "path"}, "source"),
    ({"name": "output_path", "type": "string"}, "sink"),
    ({"name": "dest_file", "type": "string"}, "sink"),
    ({"name": "report_path", "type": "string"}, "sink"),
    # Not paths at all — the false positives that would cost a needless fixture.
    ({"name": "article", "type": "string"}, None),
    ({"name": "topic", "type": "string"}, None),
    ({"name": "count", "type": "integer"}, None),
    ({"name": "paths", "type": "array"}, None),
])
def test_path_role_classifies(spec, expected):
    assert _path_role(spec) == expected


def test_backfill_skips_source_paths_but_still_fills_everything_else():
    inputs: dict = {}
    _backfill_inputs(inputs, [
        {"name": "file_path", "type": "string", "required": True},
        {"name": "output_path", "type": "string", "required": True},
        {"name": "topic", "type": "string", "required": True},
        {"name": "count", "type": "integer", "required": True},
    ])
    # The lie is gone: nothing is invented for a file the workflow will open.
    assert "file_path" not in inputs
    # A sink is written by the run, and everything else is fine to fabricate.
    assert inputs["output_path"] and inputs["topic"] and inputs["count"] == 3


# ── deriving fixtures ───────────────────────────────────────────────────────────

async def test_derive_acceptance_picks_up_fixtures():
    provider = ScriptedProvider(turns=[(json.dumps({
        "criteria": [{"id": "c", "description": "d"}],
        "test_inputs": {"file_path": "article.txt"},
        "fixtures": {"setup": SETUP, "creates": ["article.txt"]},
    }), [])])
    plan = await derive_acceptance(
        provider, "read a file", "name: reader",
        declared_inputs=[{"name": "file_path", "type": "string", "required": True}],
    )
    assert plan.fixtures and plan.fixtures.creates == ["article.txt"]
    assert plan.test_inputs == {"file_path": "article.txt"}
    assert "article.txt" in plan.render()


async def test_derive_acceptance_without_fixtures_stays_none():
    provider = ScriptedProvider(turns=[(json.dumps({
        "criteria": [{"id": "c", "description": "d"}],
        "test_inputs": {"topic": "harbours"},
    }), [])])
    plan = await derive_acceptance(provider, "write about a topic", "name: writer")
    assert plan.fixtures is None


# ── the headline: a workflow that reads a file, verified against a real one ─────

async def test_verify_runs_against_a_fixture_file(tmp_path, kb):
    provider = ScriptedProvider(turns=[(_judge_json(True), [])])
    session = _reader_session(tmp_path, kb)
    session.stage()

    report = await verify_workflow(
        provider, intent=session.intent,
        package_dir=session.staging_root / session.name,
        plan=_plan(test_inputs={"file_path": "article.txt"},
                   fixtures={"setup": SETUP, "creates": ["article.txt"]}),
        declared_inputs=session.inputs,
    )

    assert report.run_ok and report.passed, report.render()
    assert report.fixtures_created == ["article.txt"]
    # The node really opened the file the fixture wrote.
    read = next(n for n in report.node_summaries if n["id"] == "read")
    assert "harbour redevelopment" in str(read.get("output"))
    assert "Ran against fixture files: article.txt" in report.render()


async def test_sandbox_is_thrown_away_after_the_run(tmp_path, kb):
    provider = ScriptedProvider(turns=[(_judge_json(True), [])])
    session = _reader_session(tmp_path, kb)
    session.stage()
    before = set(Path(tmp_path).rglob("article.txt"))

    await verify_workflow(
        provider, intent=session.intent,
        package_dir=session.staging_root / session.name,
        plan=_plan(test_inputs={"file_path": "article.txt"},
                   fixtures={"setup": SETUP, "creates": ["article.txt"]}),
        declared_inputs=session.inputs,
    )
    # Nothing leaked into the staging tree or the repo.
    assert set(Path(tmp_path).rglob("article.txt")) == before
    assert not Path("article.txt").exists()


# ── the failures, each naming its fix ───────────────────────────────────────────

async def test_source_path_with_no_fixture_fails_before_running(tmp_path, kb):
    provider = ScriptedProvider(turns=[])  # neither runner nor judge may be reached
    session = _reader_session(tmp_path, kb)
    session.stage()

    report = await verify_workflow(
        provider, intent=session.intent,
        package_dir=session.staging_root / session.name,
        plan=_plan(test_inputs={"file_path": "article.txt"}),
        declared_inputs=session.inputs,
    )

    assert not report.passed and not report.run_ok
    assert report.graph_runs == 0          # failed before spending a run
    assert provider.calls == 0             # and before paying the judge
    assert "`file_path`" in report.diagnosis
    # Names the input so the caller can repair it, rather than only describing
    # the failure — see test_tool_re_derives_a_missing_fixture_and_recovers.
    assert report.missing_fixture_for == ["file_path"]


async def test_broken_fixture_script_is_diagnosed(tmp_path, kb):
    provider = ScriptedProvider(turns=[])
    session = _reader_session(tmp_path, kb)
    session.stage()

    report = await verify_workflow(
        provider, intent=session.intent,
        package_dir=session.staging_root / session.name,
        plan=_plan(test_inputs={"file_path": "article.txt"},
                   fixtures={"setup": "raise SystemExit('nope')",
                             "creates": ["article.txt"]}),
        declared_inputs=session.inputs,
    )
    assert not report.passed and report.graph_runs == 0
    assert "fixture setup script failed" in report.diagnosis
    assert provider.calls == 0


async def test_one_fixture_file_and_one_input_are_paired_up(tmp_path, kb):
    """The model writes the fixture and forgets to reference it — seen live.

    On gpt-4o-mini: a setup script creating `input_file.txt`, and `test_inputs`
    with no `file_path` at all. One file, one input wanting one; pairing them is
    not a guess, and it beats spending a model turn to be told the obvious.
    """
    provider = ScriptedProvider(turns=[(_judge_json(True), [])])
    session = _reader_session(tmp_path, kb)
    session.stage()

    report = await verify_workflow(
        provider, intent=session.intent,
        package_dir=session.staging_root / session.name,
        # No `file_path` at all, and the fixture calls the file something else.
        plan=_plan(test_inputs={},
                   fixtures={"setup": SETUP.replace("article.txt", "input_file.txt"),
                             "creates": ["input_file.txt"]}),
        declared_inputs=session.inputs,
    )
    assert report.run_ok and report.passed, report.render()
    read = next(n for n in report.node_summaries if n["id"] == "read")
    assert "harbour redevelopment" in str(read.get("output"))
    # Disclosed, because the acceptance plan still shows what the model wrote.
    assert report.paired_inputs == {"file_path": "input_file.txt"}
    assert "repointed at the fixture's file: input_file.txt" in report.render()


async def test_a_placeholder_path_is_repointed_at_the_real_fixture(tmp_path, kb):
    """Seen live: `test_inputs: {file_path: "path/to/sample/file.txt"}` next to a
    fixture that created `sample_article.txt`. The invented path resolves to
    nothing, so it counts as unsatisfied and gets repointed."""
    provider = ScriptedProvider(turns=[(_judge_json(True), [])])
    session = _reader_session(tmp_path, kb)
    session.stage()

    report = await verify_workflow(
        provider, intent=session.intent,
        package_dir=session.staging_root / session.name,
        plan=_plan(test_inputs={"file_path": "path/to/sample/file.txt"},
                   fixtures={"setup": SETUP, "creates": ["article.txt"]}),
        declared_inputs=session.inputs,
    )
    assert report.run_ok and report.passed, report.render()
    assert report.paired_inputs == {"file_path": "article.txt"}


async def test_ambiguous_fixture_files_are_not_guessed(tmp_path, kb):
    """Two files and one unsatisfied input is a guess, so it stays a failure."""
    provider = ScriptedProvider(turns=[])
    session = _reader_session(tmp_path, kb)
    session.stage()

    report = await verify_workflow(
        provider, intent=session.intent,
        package_dir=session.staging_root / session.name,
        plan=_plan(test_inputs={},
                   fixtures={"setup": "from pathlib import Path\n"
                                      "Path('a.txt').write_text('a')\n"
                                      "Path('b.txt').write_text('b')\n",
                             "creates": ["a.txt", "b.txt"]}),
        declared_inputs=session.inputs,
    )
    assert not report.passed and report.graph_runs == 0
    assert report.missing_fixture_for == ["file_path"]
    assert provider.calls == 0


async def test_a_fixture_may_write_into_a_declared_subdirectory(tmp_path, kb):
    """The rig makes room for what the fixture declared it would create.

    Straight from a live run: the setup script did
    `open('input/text_analysis.txt', 'w')`, which raises FileNotFoundError in a
    fresh sandbox because `input/` does not exist. It cost the whole build.
    """
    provider = ScriptedProvider(turns=[(_judge_json(True), [])])
    session = _reader_session(tmp_path, kb)
    session.stage()

    report = await verify_workflow(
        provider, intent=session.intent,
        package_dir=session.staging_root / session.name,
        plan=_plan(test_inputs={"file_path": "input/article.txt"},
                   fixtures={"setup": SETUP.replace("article.txt", "input/article.txt"),
                             "creates": ["input/article.txt"]}),
        declared_inputs=session.inputs,
    )
    assert report.run_ok and report.passed, report.render()
    read = next(n for n in report.node_summaries if n["id"] == "read")
    assert "harbour redevelopment" in str(read.get("output"))


async def test_an_undeclared_subdirectory_still_fails_as_a_rig_problem(tmp_path, kb):
    """Only declared paths get their parents made — an undeclared one is on the script."""
    provider = ScriptedProvider(turns=[])
    session = _reader_session(tmp_path, kb)
    session.stage()

    report = await verify_workflow(
        provider, intent=session.intent,
        package_dir=session.staging_root / session.name,
        plan=_plan(test_inputs={"file_path": "nope/article.txt"},
                   fixtures={"setup": "open('nope/article.txt', 'w').write('x')",
                             "creates": []}),
        declared_inputs=session.inputs,
    )
    assert not report.passed and report.fixture_setup_failed
    assert report.fixture_problem
    assert provider.calls == 0


async def test_an_unmet_creates_entry_is_reported_not_fatal(tmp_path, kb):
    """Models list node outputs in `creates`; that must not fail a good fixture.

    Seen on gpt-4o-mini: `creates: [test_document.txt, analysis_results, report]`
    — one real file and two of the workflow's own outputs. The file the workflow
    opens was there, so the run should happen.
    """
    provider = ScriptedProvider(turns=[(_judge_json(True), [])])
    session = _reader_session(tmp_path, kb)
    session.stage()

    report = await verify_workflow(
        provider, intent=session.intent,
        package_dir=session.staging_root / session.name,
        plan=_plan(test_inputs={"file_path": "article.txt"},
                   fixtures={"setup": SETUP,
                             "creates": ["article.txt", "analysis_results", "report"]}),
        declared_inputs=session.inputs,
    )
    assert report.run_ok and report.passed, report.render()
    # Reported, so a misdeclared fixture is still visible to whoever reads this.
    assert any("analysis_results" in c for c in report.fixtures_created)


# ── through the agent's own tool ────────────────────────────────────────────────

async def test_test_workflow_tool_derives_and_uses_a_fixture(tmp_path, kb):
    """End to end over the toolbelt: derive → fixture → run → judge → register."""
    derived = json.dumps({
        "criteria": [{"id": "reports_content", "description": "output reflects the file"}],
        "test_inputs": {"file_path": "article.txt"},
        "fixtures": {"setup": SETUP, "creates": ["article.txt"]},
    })
    provider = ScriptedProvider(turns=[(derived, []), (_judge_json(True), [])])
    session = _reader_session(tmp_path, kb, provider=provider, mode="required")
    ctx = _ctx(tmp_path)

    def belt(name):
        return next(t for t in architect_tools(session) if t.name == name)

    res = await belt("test_workflow").run({}, ctx)
    assert not res.is_error, res.content
    assert "VERIFICATION PASSED" in res.content
    assert "Ran against fixture files: article.txt" in res.content

    ok = await belt("register_workflow").run({}, ctx)
    assert not ok.is_error, ok.content


async def test_tool_re_derives_a_missing_fixture_and_recovers(tmp_path, kb):
    """A first plan with no fixture is repaired in code, not handed to the model.

    Observed on gpt-4o-mini before this existed: told to "add a fixture", the
    builder — whose tools only edit graphs — declared the whole build blocked.
    """
    no_fixture = json.dumps({
        "criteria": [{"id": "reports_content", "description": "output reflects the file"}],
        "test_inputs": {"file_path": "article.txt"},
    })
    with_fixture = json.dumps({
        "criteria": [{"id": "reports_content", "description": "output reflects the file"}],
        "test_inputs": {"file_path": "article.txt"},
        "fixtures": {"setup": SETUP, "creates": ["article.txt"]},
    })
    provider = ScriptedProvider(turns=[
        (no_fixture, []),        # first derivation forgets the fixture
        (with_fixture, []),      # re-derivation, told it is mandatory
        (_judge_json(True), []),  # judge on the run that then succeeds
    ])
    session = _reader_session(tmp_path, kb, provider=provider, mode="required")
    ctx = _ctx(tmp_path)

    res = await next(
        t for t in architect_tools(session) if t.name == "test_workflow"
    ).run({}, ctx)

    assert not res.is_error, res.content
    assert "VERIFICATION PASSED" in res.content
    assert "Ran against fixture files: article.txt" in res.content
    # The repaired plan is what got cached, so a later call does not repeat it.
    assert session.acceptance_plan.fixtures.creates == ["article.txt"]


async def test_two_bad_fixtures_stop_at_the_rig_and_never_reach_the_graph(tmp_path, kb):
    """One retry, then it is declared a harness problem — not handed to the model.

    Both live failures came from the model being told about a fixture it could not
    fix: once it declared the build blocked, once it began adding `setup_fixtures`
    nodes to the workflow and looped on the resulting validation errors. So a
    second rig failure says, in as many words, that the graph is not at fault.
    """
    unpairable = json.dumps({
        "criteria": [{"id": "reports_content", "description": "output reflects the file"}],
        "test_inputs": {},
        # Two files, one input wanting one — too ambiguous to pair up.
        "fixtures": {"setup": "from pathlib import Path\n"
                              "Path('a.txt').write_text('a')\n"
                              "Path('b.txt').write_text('b')\n",
                     "creates": ["a.txt", "b.txt"]},
    })
    provider = ScriptedProvider(turns=[(unpairable, []), (unpairable, [])])
    session = _reader_session(tmp_path, kb, provider=provider, mode="required")
    ctx = _ctx(tmp_path)

    res = await next(
        t for t in architect_tools(session) if t.name == "test_workflow"
    ).run({}, ctx)

    assert res.is_error
    assert provider.calls == 2       # derived, re-derived once, never judged
    assert session.fixture_retry_used
    assert "TEST HARNESS could not be set up" in res.content
    assert "Do NOT add nodes to create test files" in res.content
    assert "no change to the graph will fix it" in res.content

    # Registration is allowed — refusing would deadlock on a rig fault — but the
    # workflow is recorded as unverified.
    assert session.verification_unavailable
    reg = await next(
        t for t in architect_tools(session) if t.name == "register_workflow"
    ).run({}, ctx)
    assert not reg.is_error, reg.content
    assert "UNVERIFIED" in reg.content


async def test_the_fixture_retry_happens_only_once(tmp_path, kb):
    """A second test_workflow call must not buy a second re-derivation."""
    no_fixture = json.dumps({
        "criteria": [{"id": "reports_content", "description": "output reflects the file"}],
        "test_inputs": {"file_path": "article.txt"},
    })
    provider = ScriptedProvider(turns=[(no_fixture, []), (no_fixture, [])])
    session = _reader_session(tmp_path, kb, provider=provider, mode="required")
    ctx = _ctx(tmp_path)

    def belt():
        return next(t for t in architect_tools(session) if t.name == "test_workflow")

    await belt().run({}, ctx)
    assert provider.calls == 2 and session.fixture_retry_used

    # Second call: the graph is unchanged, so this is answered from the cache and
    # costs nothing at all.
    await belt().run({}, ctx)
    assert provider.calls == 2


def test_fixture_is_falsy_when_empty():
    assert not Fixture()
    assert not Fixture(setup="   ", creates=["a.txt"])
    assert Fixture(setup="pass")
