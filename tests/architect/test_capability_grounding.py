"""Capability grounding — can a node do what it says? (Architect V3, Phase 1)

The two failures that motivated this are reproduced verbatim as `test_regression_*`:
a workflow whose first node is called `read_file`, is a `base` node, and has no
tool; and one whose "agents" are `react` nodes with empty toolbelts. Both used to
validate clean and register.

Everything here is deterministic — no model in the loop. That is the point: these
checks have to hold on a weak model precisely because the model has no say in them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from neurosurfer.graph import Graph, GraphNode
from neurosurfer.graph.workflow.capability import (
    node_action_text,
    suspected_capability,
    toolless_react_node,
)
from neurosurfer.graph.workflow.package import WorkflowPackage
from neurosurfer.graph.workflow.schema import WorkflowManifest
from neurosurfer.graph.workflow.validate import validate_package


def _pkg(nodes: list[GraphNode], outputs: list[str], tmp_path: Path) -> WorkflowPackage:
    graph = Graph(name="t", nodes=nodes, outputs=outputs)
    return WorkflowPackage(manifest=WorkflowManifest(name="t"), graph=graph, path=tmp_path)


def _issues(report, *kinds: str) -> list:
    return [i for i in report.errors + report.warnings + report.gaps if i.kind in kinds]


# ── the structural half: a react node with no tools ─────────────────────────────

def test_toolless_react_is_an_error(tmp_path):
    nodes = [GraphNode(id="a", kind="react", goal="Do the thing.")]
    report = validate_package(_pkg(nodes, ["a"], tmp_path))
    assert not report.ok
    issues = _issues(report, "capability")
    assert len(issues) == 1
    assert issues[0].node_id == "a"
    assert "no tools" in issues[0].message


def test_react_with_tools_is_fine(tmp_path):
    nodes = [GraphNode(id="a", kind="react", tools=["read_file"],
                       goal="Read the file and report what is in it.")]
    report = validate_package(_pkg(nodes, ["a"], tmp_path))
    assert report.ok
    assert not _issues(report, "capability", "capability_gap")


def test_toolless_react_inside_a_body_is_caught(tmp_path):
    """Body nodes really run, so they are really checked."""
    inner = GraphNode(id="worker", kind="react", goal="Handle {item}.")
    nodes = [GraphNode(id="fan", kind="map", over="inputs.items", body=[inner])]
    report = validate_package(_pkg(nodes, ["fan"], tmp_path))
    assert not report.ok
    assert [i.node_id for i in _issues(report, "capability")] == ["worker"]


def test_toolless_react_helper():
    assert toolless_react_node(GraphNode(id="a", kind="react"))
    assert not toolless_react_node(GraphNode(id="a", kind="react", tools=["read_file"]))
    assert not toolless_react_node(GraphNode(id="a", kind="base"))


# ── the lexical half: a base node describing an external action ─────────────────

@pytest.mark.parametrize(
    ("goal", "label"),
    [
        ("Read the contents of the file located at {file_path}.", "read a file from disk"),
        ("Get the raw content of the file for analysis.", "read a file from disk"),
        ("Load the CSV and pull out the totals.", "read a file from disk"),
        ("List the directory to find every report.", "list a directory"),
        ("Save the summary to a file for later.", "write a file to disk"),
        ("Search the web for recent coverage.", "search the web"),
        ("Fetch the URL and pull out the title.", "fetch a URL or call an API"),
        ("Check the inbox for unread emails.", "read an email inbox"),
        ("Monitor Gmail for anything urgent.", "read an email inbox"),
        ("Send a text message when something urgent lands.",
         "send a message or notification"),
        ("Run the command and capture its output.", "run a shell command"),
        ("Query the database for open tickets.", "query a database"),
    ],
)
def test_external_actions_are_detected(goal, label):
    found = suspected_capability(goal)
    assert found is not None, f"missed: {goal}"
    assert found.label == label


@pytest.mark.parametrize(
    "goal",
    [
        # Pure text work. None of this reaches outside the prompt.
        "Summarise the article in exactly three sentences.",
        "Write a catchy title for the summary.",
        "Perform a deep analysis of the content and extract key insights.",
        "Create a summary report from the insights gathered.",
        "Draft a polite reply to the customer's email.",
        "Classify the ticket as urgent or routine.",
        "Read the summary and judge whether it is accurate.",
        "Turn the topic into a tight research scope: what to find out.",
        # Provenance, not an instruction — the verb sits behind a noun.
        "Analyze the content read from the file.",
        "Score the data fetched from the API earlier.",
        # A gerund noun phrase describing input, not an action to take.
        "Given the emails retrieved upstream, pick the important ones.",
    ],
)
def test_pure_llm_work_is_not_flagged(goal):
    assert suspected_capability(goal) is None, f"false positive: {goal}"


def test_expected_result_is_not_scanned():
    """It describes the output's shape. "List of unread emails" is a noun phrase."""
    node = GraphNode(
        id="pick",
        kind="base",
        goal="Choose the important ones.",
        expected_result="List of important unread emails.",
    )
    assert suspected_capability(*node_action_text(node)) is None


def test_node_id_alone_is_enough_signal():
    """The motivating failure named the node `read_file` and said nothing else useful."""
    node = GraphNode(id="read_file", kind="base", goal="Do it.")
    found = suspected_capability(*node_action_text(node))
    assert found is not None and found.label == "read a file from disk"


def test_gap_is_a_warning_not_an_error(tmp_path):
    """A hand-written package must not be rejected on a string match."""
    nodes = [GraphNode(id="read_file", kind="base",
                       goal="Get the raw content of the file for analysis.")]
    report = validate_package(_pkg(nodes, ["read_file"], tmp_path))
    assert report.ok                      # package-level: still valid
    gaps = _issues(report, "capability_gap")
    assert len(gaps) == 1
    assert "read_file" in (gaps[0].suggestion or "")   # names the tool that would fix it


def test_node_holding_a_tool_is_never_flagged(tmp_path):
    nodes = [GraphNode(id="load", kind="tool", tools=["read_file"],
                       tool_args={"path": "{file_path}"})]
    report = validate_package(_pkg(nodes, ["load"], tmp_path))
    assert not _issues(report, "capability", "capability_gap")


def test_suggestion_names_no_tool_when_none_exists(tmp_path):
    nodes = [GraphNode(id="a", kind="base", goal="Check the inbox for unread mail.")]
    report = validate_package(_pkg(nodes, ["a"], tmp_path))
    gap = _issues(report, "capability_gap")[0]
    assert "MCP" in (gap.suggestion or "")   # the honest answer: nothing built in


# ── rule 4: a wired-up node may already have been handed its data ───────────────

def test_upstream_dependency_forgives_a_source_capability(tmp_path):
    """The live gpt-4o-mini failure: 'Analyze the content of the file' downstream
    of the node that read it is correct as written."""
    nodes = [
        GraphNode(id="read_file", kind="tool", tools=["read_file"],
                  tool_args={"path": "/tmp/x"}, writes="file_contents"),
        GraphNode(id="analyze_content", kind="base", depends_on=["read_file"],
                  purpose="Analyze the content of the file.",
                  goal="Perform a deep analysis of the file contents."),
    ]
    report = validate_package(_pkg(nodes, ["analyze_content"], tmp_path))
    assert not _issues(report, "capability_gap")


def test_no_upstream_still_flags_a_source_capability(tmp_path):
    """Same text, nothing feeding it — now there is nowhere for the content to
    have come from."""
    nodes = [GraphNode(id="analyze_content", kind="base",
                       purpose="Analyze the content of the file.")]
    report = validate_package(_pkg(nodes, ["analyze_content"], tmp_path))
    assert len(_issues(report, "capability_gap")) == 1


def test_upstream_never_forgives_a_sink(tmp_path):
    """No upstream node can send an SMS on your behalf."""
    nodes = [
        GraphNode(id="pick", kind="base", goal="Choose the urgent ones."),
        GraphNode(id="notify_user", kind="base", depends_on=["pick"],
                  goal="Send a text message when an important email is found."),
    ]
    report = validate_package(_pkg(nodes, ["notify_user"], tmp_path))
    gaps = _issues(report, "capability_gap")
    assert [g.node_id for g in gaps] == ["notify_user"]


@pytest.mark.parametrize(
    ("goal", "is_source"),
    [
        ("Read the contents of the file.", True),
        ("Search the web for coverage.", True),
        ("Check the inbox for unread mail.", True),
        ("Send a text message to the on-call engineer.", False),
        ("Save the summary to a file.", False),
        ("Run the command and capture output.", False),
    ],
)
def test_source_and_sink_classification(goal, is_source):
    found = suspected_capability(goal)
    assert found is not None and found.is_source is is_source


# ── regressions: the two workflows from the bug report ──────────────────────────

def test_regression_file_analysis_report(tmp_path):
    """Screenshot A: three base nodes, the first one 'reads' a file with no tool."""
    nodes = [
        GraphNode(id="read_file", kind="base",
                  purpose="Read the contents of the file located at {file_path}.",
                  goal="Get the raw content of the file for analysis.",
                  expected_result="The content of the file as a string."),
        GraphNode(id="analyze_content", kind="base", depends_on=["read_file"],
                  purpose="Analyze the content read from the file.",
                  goal="Perform a deep analysis of the content and extract key insights."),
        GraphNode(id="generate_report", kind="base", depends_on=["analyze_content"],
                  goal="Create a summary report from the insights gathered."),
    ]
    report = validate_package(_pkg(nodes, ["generate_report"], tmp_path))
    gaps = _issues(report, "capability_gap")
    # Exactly the one node that lies — not the two downstream ones that merely
    # mention the file it was supposed to have read.
    assert [g.node_id for g in gaps] == ["read_file"]


def test_regression_gmail_monitoring_workflow(tmp_path):
    """Screenshot B: 'REACT AGENT' nodes with empty toolbelts."""
    nodes = [
        GraphNode(id="monitor_gmail", kind="react",
                  goal="Check for unread emails to identify important communications."),
        GraphNode(id="filter_important_emails", kind="base", depends_on=["monitor_gmail"],
                  goal="Filter unread emails based on provided criteria."),
        GraphNode(id="notify_user", kind="react", depends_on=["filter_important_emails"],
                  goal="Send a text message notification when an important email is found."),
        GraphNode(id="draft_reply", kind="react", depends_on=["filter_important_emails"],
                  goal="Create a draft reply for each important email identified."),
    ]
    report = validate_package(_pkg(nodes, ["notify_user", "draft_reply"], tmp_path))
    assert not report.ok, "this workflow must not be registerable"
    flagged = {i.node_id for i in _issues(report, "capability")}
    assert flagged == {"monitor_gmail", "notify_user", "draft_reply"}


# ── the registration gate: a build must answer for every gap ────────────────────

@pytest.fixture()
def session(tmp_path: Path):
    # The registration gate belongs to the build *session*, which arrives with
    # `architect/agent/` in plan 01 Phase 4. Everything above this line is the
    # grounding rules themselves and runs now; these five are the gate that
    # consumes them. Skipped at the fixture so the reason sits with the cause
    # rather than being repeated on five tests.
    pytest.importorskip(
        "neurosurfer.architect.agent",
        reason="BuildSession arrives in plan 01 Phase 4 — the rules it gates are already covered above",
    )
    from neurosurfer.architect.agent import BuildSession

    from neurosurfer.architect.knowledge import KnowledgeBase
    from neurosurfer.graph.workflow.registry import WorkflowRegistry

    return BuildSession(
        intent="read a file and report on it",
        staging_root=tmp_path / "staging",
        registry=WorkflowRegistry(workflows_dir=tmp_path / "registry"),
        knowledge=KnowledgeBase(),
        # The subject here is the capability gate, not verification.
        verification_mode="off",
    )


@pytest.fixture()
def ctx(tmp_path: Path):
    from neurosurfer.tools.base import AutoApproveIOHandler, ToolContext

    return ToolContext(cwd=tmp_path, io=AutoApproveIOHandler())


def _belt(session, name):
    from neurosurfer.architect.agent import architect_tools

    return next(t for t in architect_tools(session) if t.name == name)


async def _stage_ungrounded(session, ctx):
    await _belt(session, "set_workflow").run(
        {"name": "reader",
         "inputs": [{"name": "file_path", "type": "string", "required": True}]}, ctx)
    await _belt(session, "add_node").run(
        {"node": {"id": "read_file", "kind": "base",
                  "purpose": "Read the contents of the file located at {file_path}.",
                  "goal": "Get the raw content of the file for analysis."}}, ctx)
    await _belt(session, "set_outputs").run({"outputs": ["read_file"]}, ctx)


async def test_register_refuses_an_ungrounded_node(session, ctx):
    await _stage_ungrounded(session, ctx)

    ok, text = session.validate()
    assert ok, "the package itself is still valid — the gap is a warning"

    refused = await _belt(session, "register_workflow").run({}, ctx)
    assert refused.is_error
    assert "read_file" in refused.content
    assert session.registered_path is None
    # Prescriptive, per the weak-model principle: it names the next call.
    assert "update_node" in refused.content
    assert "declare_blocked" in refused.content


async def test_attaching_a_tool_unblocks_registration(session, ctx):
    await _stage_ungrounded(session, ctx)
    await _belt(session, "update_node").run(
        {"id": "read_file",
         "patch": {"kind": "tool", "tools": ["read_file"],
                   "tool_args": {"path": "{file_path}"}}}, ctx)

    ok = await _belt(session, "register_workflow").run({}, ctx)
    assert not ok.is_error, ok.content
    assert session.registry.exists("reader")


async def test_acknowledging_unblocks_and_is_recorded(session, ctx):
    await _stage_ungrounded(session, ctx)
    ack = await _belt(session, "acknowledge_capability").run(
        {"node_id": "read_file", "reason": "the text arrives from an upstream node"}, ctx)
    assert not ack.is_error
    assert session.acknowledged_capabilities["read_file"]

    ok = await _belt(session, "register_workflow").run({}, ctx)
    assert not ok.is_error, ok.content


async def test_acknowledge_needs_a_real_node_and_a_reason(session, ctx):
    await _stage_ungrounded(session, ctx)
    ack = _belt(session, "acknowledge_capability")
    assert (await ack.run({"node_id": "ghost", "reason": "x"}, ctx)).is_error
    assert (await ack.run({"node_id": "read_file", "reason": "  "}, ctx)).is_error


async def test_toolless_react_blocks_registration_as_a_hard_error(session, ctx):
    await _belt(session, "set_workflow").run({"name": "agenty"}, ctx)
    await _belt(session, "add_node").run(
        {"node": {"id": "monitor_gmail", "kind": "react",
                  "goal": "Check for unread emails to identify important ones."}}, ctx)
    refused = await _belt(session, "register_workflow").run({}, ctx)
    assert refused.is_error and "no tools" in refused.content
    # And acknowledging must NOT get past it — this one is structural.
    await _belt(session, "acknowledge_capability").run(
        {"node_id": "monitor_gmail", "reason": "trust me"}, ctx)
    still = await _belt(session, "register_workflow").run({}, ctx)
    assert still.is_error and "no tools" in still.content
