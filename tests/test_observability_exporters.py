"""Trace-exporter tests — config detection, the event→lifecycle mapping, the
live ``base._tap`` wiring through a real agent run, and fail-soft guarantees.

No network: a :class:`MemoryExporter` records the lifecycle and the agent is
driven by ``ScriptedProvider``.
"""

from __future__ import annotations

import pytest

from neurosurfer.agents import events
from neurosurfer.agents.oneshot import Agent
from neurosurfer.agents.runtime.permissions import Guardrails
from neurosurfer.config.observability import detect_exporters_from_env
from neurosurfer.llm.types import Usage
from neurosurfer.observability.context import TraceContext
from neurosurfer.observability.exporters import (
    MemoryExporter,
    configure_exporters,
    get_active_exporters,
    register_exporter,
    reset_exporters,
)
from neurosurfer.observability.exporters.base import TraceExporter
from neurosurfer.observability.exporters.stream import TraceStreamObserver
from neurosurfer.tools import default_pool

from .fakes import ScriptedIO, ScriptedProvider


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_exporters()
    yield
    reset_exporters()


# ── env detection ───────────────────────────────────────────────────────────
def test_env_detection_off_by_default():
    assert detect_exporters_from_env({}) == []


def test_env_detection_langfuse_and_otel():
    assert detect_exporters_from_env(
        {"LANGFUSE_PUBLIC_KEY": "p", "LANGFUSE_SECRET_KEY": "s"}
    ) == ["langfuse"]
    assert detect_exporters_from_env({"OTEL_EXPORTER_OTLP_ENDPOINT": "http://x"}) == ["otel"]


def test_env_explicit_override_forces_off():
    env = {"NEUROSURFER_EXPORTERS": "none", "LANGFUSE_PUBLIC_KEY": "p", "LANGFUSE_SECRET_KEY": "s"}
    assert detect_exporters_from_env(env) == []


# ── observer maps the event stream onto exporter hooks ──────────────────────
def test_observer_lifecycle_mapping():
    mem = MemoryExporter()
    ctx = TraceContext(metadata={"agent_type": "ReactAgent"})
    obs = TraceStreamObserver(ctx, [mem], model="m", name="ReactAgent.run")
    obs.start(input="hi")
    from neurosurfer.tools.base import ToolResult

    for ev in [
        events.ToolStarted(id="c1", name="calc", args={"x": 1}),
        events.ToolFinished(id="c1", name="calc", result=ToolResult(content="42")),
        events.TurnCompleted(usage=Usage(input_tokens=100, output_tokens=20), stop_reason="tool_use"),
        events.TextDelta("The answer "),
        events.TextDelta("is 42."),
        events.TurnCompleted(usage=Usage(input_tokens=130, output_tokens=8), stop_reason="end_turn"),
        events.RunFinished(status="completed", report="The answer is 42."),
    ]:
        obs.handle(ev)
    obs.close()

    assert mem.hooks() == [
        "run_start", "tool_start", "tool_finish", "turn", "turn", "run_finish", "flush"
    ]
    turns = mem.of("turn")
    assert (turns[0]["input_tokens"], turns[0]["output_tokens"]) == (100, 20)
    assert turns[1]["output"] == "The answer is 42."
    assert mem.of("run_finish")[0]["output"] == "The answer is 42."


# ── live wiring: a real agent run drives the registered exporter ────────────
def _oneshot(provider, cwd):
    return Agent(
        provider=provider,
        tools=default_pool(),
        system_prompt="Answer.",
        guardrails=Guardrails(write_scope=["**"]),
        io=ScriptedIO(),
        cwd=cwd,
    )


@pytest.mark.asyncio
async def test_agent_run_emits_trace(tmp_path):
    (tmp_path / "a.txt").write_text("hello\n")
    mem = MemoryExporter()
    register_exporter(mem)

    turns = [
        ("", [("read_file", {"path": "a.txt"})]),  # tool round
        ("The file says hello.", []),               # synthesis
    ]
    agent = _oneshot(ScriptedProvider(turns), tmp_path)
    _ = [ev async for ev in agent.run("read a.txt")]

    hooks = mem.hooks()
    assert hooks[0] == "run_start"
    assert "tool_start" in hooks and "tool_finish" in hooks
    assert mem.of("tool_start")[0]["name"] == "read_file"
    # ScriptedProvider reports usage per turn → at least one turn with token counts.
    turns_rec = mem.of("turn")
    assert turns_rec and turns_rec[0]["input_tokens"] == 10
    rf = mem.of("run_finish")
    assert rf and rf[0]["status"] == "completed"
    assert hooks[-1] == "flush"


@pytest.mark.asyncio
async def test_bad_exporter_never_breaks_the_run(tmp_path):
    class BadExporter(TraceExporter):
        name = "bad"

        def on_run_start(self, *a, **k):
            raise RuntimeError("boom")

        def on_turn(self, *a, **k):
            raise RuntimeError("boom")

        def on_run_finish(self, *a, **k):
            raise RuntimeError("boom")

    mem = MemoryExporter()
    register_exporter(BadExporter())
    register_exporter(mem)

    agent = _oneshot(ScriptedProvider([("Paris.", [])]), tmp_path)
    result = await agent.complete("capital of France?")

    assert result == "Paris."               # run unaffected by the raising exporter
    assert mem.hooks()[0] == "run_start"     # the good exporter still saw everything
    assert mem.of("run_finish")[0]["status"] == "completed"


def test_unknown_exporter_is_skipped_not_fatal():
    assert configure_exporters(["does-not-exist"]) == []
    assert get_active_exporters() == []
