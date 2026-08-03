"""V3 Phase 5b — steps this machine cannot exercise, run as stubs and declared.

A workflow whose Gmail step needs an MCP server nobody installed cannot be tested
on that step. Refusing to run any of it teaches nothing; quietly passing it is
worse. So the step runs against a stub, the rest of the graph is really tested,
and the headline says how much of the graph the verdict actually covers.

Hermetic throughout: `tool` nodes naming a tool that does not exist, plus the
ScriptedProvider as the judge.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from neurosurfer.architect.agent import AcceptancePlan, BuildSession, verify_workflow
from neurosurfer.architect.knowledge import KnowledgeBase
from neurosurfer.graph.workflow.package import load_package
from neurosurfer.graph.workflow.registry import WorkflowRegistry
from neurosurfer.graph.workflow.runner import (
    NOT_EXERCISED_MARK,
    NotExercisedTool,
    WorkflowRunner,
)
from neurosurfer.tools.base import AutoApproveIOHandler, ToolContext

from ..fakes import ScriptedProvider

MISSING = "gmail_read_inbox"   # no such tool on any machine running these tests


@pytest.fixture(scope="module")
def kb():
    return KnowledgeBase()


def _gmail_session(tmp_path: Path, kb, provider=None) -> BuildSession:
    """Two steps: one needs a tool nobody has, one is a plain LLM node."""
    s = BuildSession(
        intent="read my inbox and summarise what needs a reply",
        staging_root=tmp_path / "staging",
        registry=WorkflowRegistry(workflows_dir=tmp_path / "registry"),
        knowledge=kb, provider=provider,
    )
    s.name = "inbox_triage"
    s.inputs = [{"name": "since", "type": "string", "required": True}]
    s.nodes = [
        {"id": "fetch", "kind": "tool", "tools": [MISSING], "tool_args": {}},
        {"id": "summarise", "kind": "function", "depends_on": ["fetch"],
         "callable": "tests.architect.test_architect_external_steps._count_chars"},
    ]
    s.outputs = ["summarise"]
    return s


def _count_chars(**kwargs) -> str:
    """Stands in for an LLM step: proves the node downstream of a stub still ran."""
    return f"downstream ran over {len(str(kwargs.get('fetch', '')))} chars"


def _judge_json(passed: bool) -> str:
    return json.dumps({
        "verdicts": [{"id": "triaged", "passed": passed, "reason": "checked"}],
        "diagnosis": "" if passed else "the inbox step never really ran",
        "suggestions": "" if passed else "install a Gmail MCP server",
    })


def _plan(**kw) -> AcceptancePlan:
    return AcceptancePlan.model_validate({
        "criteria": [{"id": "triaged", "description": "important mail is surfaced"}],
        "test_inputs": {"since": "yesterday"},
        **kw,
    })


# ── the stub itself ─────────────────────────────────────────────────────────────

async def test_not_exercised_tool_succeeds_and_marks_its_output(tmp_path):
    tool = NotExercisedTool("gmail_read", "no MCP server is connected")
    res = await tool.run({"anything": 1, "at": "all"}, ToolContext(
        cwd=tmp_path, io=AutoApproveIOHandler()))
    # `ok`, not error: the point is that the rest of the graph keeps going.
    assert not res.is_error
    assert NOT_EXERCISED_MARK in res.content
    assert "gmail_read" in res.content and "no MCP server is connected" in res.content


def _write_pkg(tmp_path: Path) -> Path:
    pkg_dir = tmp_path / "wf"
    pkg_dir.mkdir()
    (pkg_dir / "workflow.yaml").write_text(yaml.dump(
        {"name": "wf", "version": "1.0.0", "entrypoint": "graph.yaml"}))
    (pkg_dir / "graph.yaml").write_text(yaml.dump({
        "name": "wf",
        "nodes": [{"id": "fetch", "kind": "tool", "tools": [MISSING], "tool_args": {}}],
        "outputs": ["fetch"],
    }))
    return pkg_dir


class _Dummy:
    model = "dummy"
    capabilities = type("C", (), {"context_window": 8192, "max_output_tokens": 512})()


def test_runner_still_refuses_a_missing_tool_by_default(tmp_path):
    """A real run must fail loudly — stubbing is a verification affordance only."""
    runner = WorkflowRunner(_Dummy(), cwd=tmp_path)
    with pytest.raises(ValueError, match="not registered"):
        runner.run(load_package(_write_pkg(tmp_path)), {})


def test_an_mcp_tool_whose_server_has_no_credentials_cannot_be_tested(monkeypatch):
    """The case 5b is actually about: connected, so it validates; no secret, so
    calling it would fail with an auth error partway through a run."""
    from neurosurfer.config.mcp import McpServerConfig
    from neurosurfer.graph.workflow.runner import uncredentialed_reason

    class _FakeMcpTool:
        is_mcp = True
        name = "gmail_read"
        server_name = "gmail"

    cfg = McpServerConfig(name="gmail", command="npx",
                          env={"GMAIL_TOKEN": "${GMAIL_TOKEN}"})

    class _Store:
        @staticmethod
        def default():
            return type("S", (), {"get": staticmethod(lambda _n: cfg)})()

    monkeypatch.setattr("neurosurfer.config.mcp.McpStore", _Store)

    monkeypatch.delenv("GMAIL_TOKEN", raising=False)
    reason = uncredentialed_reason(_FakeMcpTool())
    assert reason and "GMAIL_TOKEN" in reason and "gmail" in reason

    # Supplied → testable for real, no stub.
    monkeypatch.setenv("GMAIL_TOKEN", "sekrit")
    assert uncredentialed_reason(_FakeMcpTool()) is None


def test_a_non_mcp_tool_is_never_treated_as_uncredentialed():
    from neurosurfer.graph.workflow.runner import uncredentialed_reason
    from neurosurfer.tools.registry import all_tools

    read_file = {t.name: t for t in all_tools()}["read_file"]
    assert uncredentialed_reason(read_file) is None


def test_runner_stubs_a_missing_tool_when_asked(tmp_path):
    runner = WorkflowRunner(_Dummy(), cwd=tmp_path, stub_missing_tools=True)
    result = runner.run(load_package(_write_pkg(tmp_path)), {})

    assert runner.stubbed_tools == {MISSING}
    assert result.nodes["fetch"].error is None
    assert NOT_EXERCISED_MARK in str(result.nodes["fetch"].raw_output)


# ── verification declares what it could not cover ───────────────────────────────

async def test_verification_is_partial_and_says_so_in_the_headline(tmp_path, kb):
    provider = ScriptedProvider(turns=[(_judge_json(True), [])])
    session = _gmail_session(tmp_path, kb)
    session.stage()

    report = await verify_workflow(
        provider, intent=session.intent,
        package_dir=session.staging_root / session.name,
        plan=_plan(), declared_inputs=session.inputs,
    )

    assert report.run_ok
    assert report.partial
    assert report.not_exercised == ["fetch"]
    assert report.stubbed_tools == [MISSING]

    headline = report.render().splitlines()[0]
    assert "PARTIAL" in headline and "1 step" in headline
    body = report.render()
    assert "NOT exercised" in body and MISSING in body

    # The rest of the graph really ran — that is the whole reason to stub.
    downstream = next(n for n in report.node_summaries if n["id"] == "summarise")
    assert downstream["status"] == "ok"
    assert "downstream ran over" in str(downstream["output"])


async def test_the_judge_is_told_which_outputs_are_stubs(tmp_path, kb):
    """Otherwise it scores fabricated data as though the step had worked."""
    provider = ScriptedProvider(turns=[(_judge_json(False), [])])
    session = _gmail_session(tmp_path, kb)
    session.stage()

    await verify_workflow(
        provider, intent=session.intent,
        package_dir=session.staging_root / session.name,
        plan=_plan(), declared_inputs=session.inputs,
    )

    judge_prompt = provider.prompts[-1]
    assert "did NOT run for real" in judge_prompt
    assert "fetch" in judge_prompt


async def test_partial_verification_registers_but_says_so(tmp_path, kb):
    """Phase 5c: blocking here would make the workflow unregisterable on the very
    machine that cannot exercise it. It registers, and the message names what has
    never been proven.

    The verification is recorded directly: a graph naming a tool that does not
    exist is turned away by the Phase 1 capability gate long before verification,
    so the only way to reach a partial run through the toolbelt is a tool that
    disappeared *after* validation — which is precisely what this stands in for.
    """
    from neurosurfer.architect.agent.verify import VerificationReport

    session = _gmail_session(tmp_path, kb)
    # A plain graph that validates; the partiality is in the recorded report.
    session.nodes = [{"id": "fetch", "kind": "function",
                      "callable": "tests.architect.test_architect_external_steps._count_chars"}]
    session.outputs = ["fetch"]
    session.stage()
    assert session.verification_mode == "required"   # the new default

    report = VerificationReport(
        passed=True, run_ok=True, graph_runs=1,
        not_exercised=["fetch"], stubbed_tools=[MISSING],
    )
    session.record_verification(
        passed=True, rendered=report.render(), report=report, test_inputs=None)

    ok, msg = session.register()
    assert ok, msg
    assert "PARTIAL verification" in msg
    assert "fetch" in msg and "never been proven" in msg


async def test_a_broken_test_rig_registers_as_unverified_rather_than_stalling(tmp_path, kb):
    """A fixture failure is not a verdict on the graph, and must not deadlock.

    Under `required`, refusing here would stall the build forever on a problem the
    workflow cannot cause and the builder cannot fix — the same trap that made a
    partial verification registerable.
    """
    from neurosurfer.architect.agent.verify import VerificationReport

    session = _gmail_session(tmp_path, kb)
    session.nodes = [{"id": "fetch", "kind": "function",
                      "callable": "tests.architect.test_architect_external_steps._count_chars"}]
    session.outputs = ["fetch"]
    session.stage()

    report = VerificationReport(
        passed=False, run_ok=False, graph_runs=0,
        missing_fixture_for=["file_path"],
        diagnosis="The workflow reads `file_path`, and no test fixture created it.",
    )
    assert report.fixture_problem
    session.record_verification(
        passed=False, rendered=report.render(), report=report, test_inputs=None)

    assert session.verification_unavailable
    ok, msg = session.register()
    assert ok, msg
    assert "UNVERIFIED" in msg
    assert "nothing has ever run it end to end" in msg


async def test_a_genuine_verification_failure_still_blocks(tmp_path, kb):
    """The rig escape must not become a way past a real failing verification."""
    from neurosurfer.architect.agent.verify import VerificationReport

    session = _gmail_session(tmp_path, kb)
    session.nodes = [{"id": "fetch", "kind": "function",
                      "callable": "tests.architect.test_architect_external_steps._count_chars"}]
    session.outputs = ["fetch"]
    session.stage()

    report = VerificationReport(
        passed=False, run_ok=True, graph_runs=1,
        verdicts=[{"id": "triaged", "passed": False, "reason": "wrong output"}],
    )
    assert not report.fixture_problem
    session.record_verification(
        passed=False, rendered=report.render(), report=report, test_inputs=None)

    assert not session.verification_unavailable
    ok, msg = session.register()
    assert not ok and "verification FAILED" in msg


async def test_a_full_verification_registers_without_a_caveat(tmp_path, kb):
    from neurosurfer.architect.agent.verify import VerificationReport

    session = _gmail_session(tmp_path, kb)
    session.nodes = [{"id": "fetch", "kind": "function",
                      "callable": "tests.architect.test_architect_external_steps._count_chars"}]
    session.outputs = ["fetch"]
    session.stage()

    report = VerificationReport(passed=True, run_ok=True, graph_runs=1)
    session.record_verification(
        passed=True, rendered=report.render(), report=report, test_inputs=None)

    ok, msg = session.register()
    assert ok and "PARTIAL" not in msg


async def test_a_fully_exercised_graph_is_not_partial(tmp_path, kb):
    """The common case pays nothing for this: no stubs, no partial banner."""
    provider = ScriptedProvider(turns=[(_judge_json(True), [])])
    session = _gmail_session(tmp_path, kb)
    # Same graph with the unavailable step removed.
    session.nodes = [{"id": "summarise", "kind": "function",
                      "callable": "tests.architect.test_architect_external_steps._count_chars"}]
    session.outputs = ["summarise"]
    session.stage()

    report = await verify_workflow(
        provider, intent=session.intent,
        package_dir=session.staging_root / session.name,
        plan=_plan(), declared_inputs=session.inputs,
    )
    assert report.passed and not report.partial
    assert report.not_exercised == []
    assert "PARTIAL" not in report.render()
