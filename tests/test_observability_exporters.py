"""Trace-exporter tests — config detection, the event→lifecycle mapping, the
live ``base._tap`` wiring through a real agent run, fail-soft guarantees, and
(Phase 5) a graph workflow rendering as one nested trace.

No network: a :class:`MemoryExporter` records the lifecycle and agents/graphs
are driven by ``ScriptedProvider``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from neurosurfer.agents import events
from neurosurfer.agents.oneshot import Agent
from neurosurfer.agents.runtime.permissions import Guardrails
from neurosurfer.config.observability import detect_exporters_from_env
from neurosurfer.graph.workflow.package import load_package
from neurosurfer.graph.workflow.runner import WorkflowRunner
from neurosurfer.llm.types import Message, TextBlock, Usage
from neurosurfer.observability.context import TraceContext
from neurosurfer.observability.exporters import (
    MemoryExporter,
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
        events.TurnCompleted(
            usage=Usage(input_tokens=130, output_tokens=8),
            stop_reason="end_turn",
            output=Message(role="assistant", content=[TextBlock(text="The answer is 42.")]),
        ),
        events.RunFinished(status="completed", report="The answer is 42."),
    ]:
        obs.handle(ev)
    obs.close()

    assert mem.hooks() == [
        "run_start", "tool_start", "tool_finish", "turn", "turn", "run_finish", "flush"
    ]
    turns = mem.of("turn")
    assert (turns[0]["input_tokens"], turns[0]["output_tokens"]) == (100, 20)
    assert turns[1]["output"] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "The answer is 42."}],
    }
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


# ── Phase 5: a graph workflow renders as one nested trace ───────────────────
class TestWorkflowTraceNesting:
    """workflow → node → agent: three nested spans, all sharing one trace.

    Hierarchy exercised here::

        workflow:<name>          (root span, opened by WorkflowRunner/executor)
        └── node:<id>            (one span per graph node)
            └── <Agent>.run      (the node's agent, if it is an LLM node)

    The root + node spans come from ``traced_run``; each node's agent flows
    through ``base._tap`` and nests under its node span.
    """

    @staticmethod
    def _base_node_pkg(pkg_dir: Path) -> None:
        """A one-node workflow whose single node is a base (LLM) node."""
        pkg_dir.mkdir(parents=True, exist_ok=True)
        (pkg_dir / "workflow.yaml").write_text(
            yaml.dump({"name": pkg_dir.name, "version": "0.1.0", "entrypoint": "graph.yaml"}),
            encoding="utf-8",
        )
        graph = {
            "name": pkg_dir.name,
            "nodes": [{"id": "n", "kind": "base", "purpose": "answer"}],
            "outputs": ["n"],
        }
        (pkg_dir / "graph.yaml").write_text(yaml.dump(graph), encoding="utf-8")

    @staticmethod
    def _by_name(starts, prefix):
        return [s for s in starts if s["name"].startswith(prefix)]

    @classmethod
    def _one(cls, starts, prefix):
        matches = cls._by_name(starts, prefix)
        assert len(matches) == 1, f"expected one {prefix!r} span, got {len(matches)}"
        return matches[0]

    def test_workflow_node_agent_hierarchy(self, tmp_path):
        """workflow → node → agent: three nested spans, one trace."""
        reset_exporters()
        mem = MemoryExporter()
        register_exporter(mem)
        try:
            pkg_dir = tmp_path / "wf"
            self._base_node_pkg(pkg_dir)
            pkg = load_package(pkg_dir)
            WorkflowRunner(ScriptedProvider([("the answer", [])])).run(pkg, {"query": "hi"})
        finally:
            reset_exporters()

        starts = mem.of("run_start")
        wf = self._one(starts, "workflow:")
        node = self._one(starts, "node:")
        agent = self._one(starts, "Agent")  # OneShotAgent → "Agent.run"

        # workflow is the root; node nests under workflow; agent nests under node.
        assert wf["parent_span_id"] is None
        assert node["parent_span_id"] == wf["span_id"]
        assert agent["parent_span_id"] == node["span_id"]
        # all share one trace
        assert node["trace_id"] == wf["trace_id"] == agent["trace_id"]
        # every span opened is closed
        assert len(mem.of("run_finish")) == len(starts) == 3

        # V2: workflow + node spans carry the run's real I/O — previously they
        # showed null/undefined and only the nested agent had input/output.
        assert wf["input"] == {"query": "hi"}
        assert node["input"]["graph_inputs"] == {"query": "hi"}
        finishes = {f.get("span_id"): f for f in mem.of("run_finish")}
        assert "the answer" in str(finishes[node["span_id"]]["output"])
        assert finishes[wf["span_id"]]["output"] == {"n": finishes[node["span_id"]]["output"]}
