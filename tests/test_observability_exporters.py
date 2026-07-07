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


# ── Phase 5: nesting + session grouping ─────────────────────────────────────
@pytest.mark.asyncio
async def test_run_nests_under_active_parent(tmp_path):
    """A run started inside another run inherits its trace and nests under its span."""
    from neurosurfer.observability.context import (
        TraceContext,
        pop_trace_context,
        push_trace_context,
    )

    mem = MemoryExporter()
    register_exporter(mem)

    parent = TraceContext(session_id="sess-1", metadata={"agent_type": "Parent"})
    token = push_trace_context(parent)              # simulate being inside a parent run
    try:
        agent = _oneshot(ScriptedProvider([("hi", [])]), tmp_path)
        await agent.complete("x")
    finally:
        pop_trace_context(token)

    rs = mem.of("run_start")[0]
    assert rs["trace_id"] == parent.trace_id          # same trace
    assert rs["parent_span_id"] == parent.span_id     # nested under the parent span
    assert rs["span_id"] != parent.span_id            # its own span
    assert rs["session_id"] == "sess-1"               # session inherited


@pytest.mark.asyncio
async def test_session_id_flows_to_trace(tmp_path):
    mem = MemoryExporter()
    register_exporter(mem)
    agent = Agent(
        provider=ScriptedProvider([("hi", [])]),
        tools=default_pool(),
        system_prompt="Answer.",
        guardrails=Guardrails(write_scope=["**"]),
        io=ScriptedIO(),
        cwd=tmp_path,
        session_id="conversation-42",
    )
    await agent.complete("x")
    assert mem.of("run_start")[0]["session_id"] == "conversation-42"


def test_context_var_isolated_after_run():
    """The ambient trace context is cleared once a run's observer closes."""
    from neurosurfer.observability.context import TraceContext, current_trace_context
    from neurosurfer.observability.exporters.stream import TraceStreamObserver

    assert current_trace_context() is None
    obs = TraceStreamObserver(TraceContext(), [MemoryExporter()], model="m", name="r")
    obs.start()
    assert current_trace_context() is not None        # published while running
    obs.close()
    assert current_trace_context() is None             # cleared after close
