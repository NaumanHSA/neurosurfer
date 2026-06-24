from __future__ import annotations

from pathlib import Path

import pytest

from neurosurfer.agents import events
from neurosurfer.agents.agentic_loop import AgenticLoop
from neurosurfer.agents.runtime.permissions import Guardrails
from neurosurfer.llm.providers.anthropic import AnthropicProvider
from neurosurfer.llm.providers.openai import OpenAICompatProvider
from neurosurfer.tools import default_pool

from .fakes import (
    FakeAnthropicClient,
    FakeOpenAIClient,
    ScriptedIO,
    ScriptedProvider,
)


def make_real_provider(kind: str, turns):
    if kind == "anthropic":
        p = AnthropicProvider(api_key="test", model="claude-opus-4-8")
        p._client = FakeAnthropicClient(list(turns))  # type: ignore[attr-defined]
        return p
    p = OpenAICompatProvider(
        base_url="http://x/v1", api_key="t", model="local", context_window=200_000
    )
    p._client = FakeOpenAIClient(list(turns))  # type: ignore[attr-defined]
    return p


def build_agent(provider, cwd: Path, *, mode="default", guardrails=None, io=None):
    return AgenticLoop(
        provider=provider,
        tools=default_pool(),
        system_prompt="Do the task.",
        guardrails=guardrails or Guardrails(write_scope=["docs/"]),
        io=io or ScriptedIO(),
        cwd=cwd,
        mode=mode,
    )


async def drive(agent: AgenticLoop, user_input: str) -> list[events.Event]:
    out = []
    async for ev in agent.run(user_input):
        out.append(ev)
    return out


@pytest.mark.parametrize("provider_kind", ["scripted", "anthropic", "openai"])
@pytest.mark.asyncio
async def test_full_scripted_task_terminates(tmp_path, provider_kind):
    (tmp_path / "a.txt").write_text("hello source\n")
    turns = [
        ("Reading the source.", [("read_file", {"path": "a.txt"})]),
        ("Writing docs.", [("write_file", {"path": "docs/out.md", "content": "# Docs\n"})]),
        ("All done.", [("finish", {"summary": "wrote docs", "status": "success"})]),
    ]
    if provider_kind == "scripted":
        provider = ScriptedProvider(turns)
    else:
        provider = make_real_provider(provider_kind, turns)

    agent = build_agent(provider, tmp_path)
    evs = await drive(agent, "document this repo")

    finished = [e for e in evs if isinstance(e, events.RunFinished)]
    assert finished and finished[-1].status == "success"
    assert (tmp_path / "docs" / "out.md").read_text() == "# Docs\n"
    # Tools were executed and reported.
    tool_done = [e for e in evs if isinstance(e, events.ToolFinished)]
    assert {e.name for e in tool_done} == {"read_file", "write_file", "finish"}


@pytest.mark.asyncio
async def test_write_outside_scope_is_denied(tmp_path):
    turns = [
        ("Writing outside scope.", [("write_file", {"path": "secret.md", "content": "x"})]),
        ("Done.", [("finish", {"summary": "tried"})]),
    ]
    agent = build_agent(ScriptedProvider(turns), tmp_path, guardrails=Guardrails(write_scope=["docs/"]))
    evs = await drive(agent, "go")
    denied = [
        e for e in evs if isinstance(e, events.ToolFinished) and e.name == "write_file"
    ][0]
    assert denied.result.is_error and "scope" in denied.result.content.lower()
    assert not (tmp_path / "secret.md").exists()
    # The user was asked (escalation), and declined.
    assert agent.io.write_requests  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_write_outside_scope_allow_once(tmp_path):
    """'once' allows a single out-of-scope write; the next one asks again."""
    out = tmp_path / "elsewhere"
    out.mkdir()
    turns = [
        ("Writing 1.", [("write_file", {"path": str(out / "a.md"), "content": "A"})]),
        ("Writing 2.", [("write_file", {"path": str(out / "b.md"), "content": "B"})]),
        ("Done.", [("finish", {"summary": "wrote both"})]),
    ]
    io = ScriptedIO(write_choice="once")
    g = Guardrails(write_scope=["docs/"])
    agent = build_agent(ScriptedProvider(turns), tmp_path, guardrails=g, io=io)
    evs = await drive(agent, "go")
    writes = [e for e in evs if isinstance(e, events.ToolFinished) and e.name == "write_file"]
    assert all(not w.result.is_error for w in writes)
    assert (out / "a.md").read_text() == "A" and (out / "b.md").read_text() == "B"
    # 'once' never widens scope → every out-of-scope write is re-prompted.
    assert len(io.write_requests) == 2
    assert g.write_scope == ["docs/"]


@pytest.mark.asyncio
async def test_write_outside_scope_always_widens_and_persists(tmp_path):
    """'always' allows + adds the folder to write_scope, so siblings don't re-ask."""
    out = tmp_path / "elsewhere"
    out.mkdir()
    persisted: list[str] = []
    turns = [
        ("Writing 1.", [("write_file", {"path": str(out / "a.md"), "content": "A"})]),
        ("Writing 2.", [("write_file", {"path": str(out / "b.md"), "content": "B"})]),
        ("Done.", [("finish", {"summary": "wrote both"})]),
    ]
    io = ScriptedIO(write_choice="always")
    g = Guardrails(write_scope=["docs/"])
    agent = AgenticLoop(
        provider=ScriptedProvider(turns),
        tools=default_pool(),
        system_prompt="Do the task.",
        guardrails=g,
        io=io,
        cwd=tmp_path,
        persist_scope=persisted.append,
    )
    evs = await drive(agent, "go")
    writes = [e for e in evs if isinstance(e, events.ToolFinished) and e.name == "write_file"]
    assert all(not w.result.is_error for w in writes)
    # Asked once; the folder was added to the live scope and the persist hook fired.
    assert len(io.write_requests) == 1
    assert any(str(out) in s for s in g.write_scope)
    assert persisted and str(out) in persisted[0]


@pytest.mark.asyncio
async def test_plan_gate_blocks_then_allows(tmp_path):
    turns = [
        # In plan mode, this write must be denied.
        ("Trying to write early.", [("write_file", {"path": "docs/a.md", "content": "early"})]),
        # Present a plan — approved by ScriptedIO.
        ("Here is the plan.", [("present_plan", {"plan": "1. write docs"})]),
        # Now in default mode, the write succeeds.
        ("Writing now.", [("write_file", {"path": "docs/a.md", "content": "final"})]),
        ("Done.", [("finish", {"summary": "ok"})]),
    ]
    agent = build_agent(
        ScriptedProvider(turns),
        tmp_path,
        mode="plan",
        guardrails=Guardrails(write_scope=["docs/"]),
        io=ScriptedIO(approve_plan=True),
    )
    evs = await drive(agent, "go")

    writes = [e for e in evs if isinstance(e, events.ToolFinished) and e.name == "write_file"]
    assert writes[0].result.is_error  # blocked in plan mode
    assert not writes[1].result.is_error  # allowed after approval
    assert any(isinstance(e, events.ModeChanged) for e in evs)
    assert (tmp_path / "docs" / "a.md").read_text() == "final"


@pytest.mark.asyncio
async def test_shell_gate_denied(tmp_path):
    turns = [
        ("Running a command.", [("run_command", {"command": "echo hi"})]),
        ("Done.", [("finish", {"summary": "ok"})]),
    ]
    agent = build_agent(
        ScriptedProvider(turns),
        tmp_path,
        guardrails=Guardrails(write_scope=["docs/"], shell_policy="gated"),
        io=ScriptedIO(approve_shell=False),
    )
    evs = await drive(agent, "go")
    sh = [e for e in evs if isinstance(e, events.ToolFinished) and e.name == "run_command"][0]
    assert sh.result.is_error and "declined" in sh.result.content.lower()


@pytest.mark.asyncio
async def test_path_deny_blocks_read(tmp_path):
    (tmp_path / ".env").write_text("SECRET=1")
    turns = [
        ("Reading env.", [("read_file", {"path": ".env"})]),
        ("Done.", [("finish", {"summary": "ok"})]),
    ]
    agent = build_agent(ScriptedProvider(turns), tmp_path)
    evs = await drive(agent, "go")
    rd = [e for e in evs if isinstance(e, events.ToolFinished) and e.name == "read_file"][0]
    assert rd.result.is_error and "denied" in rd.result.content.lower()


@pytest.mark.asyncio
async def test_parallel_reads_scheduled(tmp_path):
    (tmp_path / "a.txt").write_text("A")
    (tmp_path / "b.txt").write_text("B")
    turns = [
        (
            "Reading both.",
            [("read_file", {"path": "a.txt"}), ("read_file", {"path": "b.txt"})],
        ),
        ("Done.", [("finish", {"summary": "ok"})]),
    ]
    agent = build_agent(ScriptedProvider(turns), tmp_path)
    evs = await drive(agent, "go")
    reads = [e for e in evs if isinstance(e, events.ToolFinished) and e.name == "read_file"]
    assert len(reads) == 2 and all(not r.result.is_error for r in reads)


@pytest.mark.asyncio
async def test_write_between_reads_executes_in_order(tmp_path):
    """Tool dispatch ordering: a non-concurrency-safe tool (write) between two reads
    must run *after* the first read and *before* the second — the all-reads-first
    split would let the second read see stale data."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "x.md").write_text("v1")
    turns = [
        (
            "Read, rewrite, re-read.",
            [
                ("read_file", {"path": "docs/x.md"}),
                ("write_file", {"path": "docs/x.md", "content": "v2"}),
                ("read_file", {"path": "docs/x.md"}),
            ],
        ),
        ("Done.", [("finish", {"summary": "ok"})]),
    ]
    agent = build_agent(ScriptedProvider(turns), tmp_path, guardrails=Guardrails(write_scope=["docs/"]))
    evs = await drive(agent, "go")
    reads = [e for e in evs if isinstance(e, events.ToolFinished) and e.name == "read_file"]
    assert len(reads) == 2
    assert "v1" in reads[0].result.content  # first read saw original
    assert "v2" in reads[1].result.content  # second read saw the write (correct ordering)


@pytest.mark.asyncio
async def test_max_turns_guard(tmp_path):
    # Never calls finish — should stop at max_turns.
    turns = [("loop", [("read_file", {"path": "a.txt"})]) for _ in range(10)]
    (tmp_path / "a.txt").write_text("x")
    agent = build_agent(
        ScriptedProvider(turns), tmp_path, guardrails=Guardrails(write_scope=["docs/"], max_turns=3)
    )
    evs = await drive(agent, "go")
    finished = [e for e in evs if isinstance(e, events.RunFinished)][-1]
    assert finished.status == "max_turns"
    assert agent.turns == 3
