"""Trace-exporter tests — config detection, the event→lifecycle mapping, the
live ``base._tap`` wiring through a real agent run, fail-soft guarantees, and
(Phase 5) a graph workflow rendering as one nested trace.

No network: a :class:`MemoryExporter` records the lifecycle and agents/graphs
are driven by ``ScriptedProvider``.
"""

from __future__ import annotations

import logging
import time
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
from neurosurfer.observability.dispatch import drain
from neurosurfer.observability.exporters import (
    MemoryExporter,
    configure_exporters,
    get_active_exporters,
    register_exporter,
    reset_exporters,
)
from neurosurfer.observability.exporters.base import TraceExporter
from neurosurfer.observability.exporters.otel import _QuietSpanExporter, _root_cause
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
    assert drain(), "trace-export thread did not finish"

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
    assert drain(), "trace-export thread did not finish"

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
    assert drain(), "trace-export thread did not finish"

    assert result == "Paris."               # run unaffected by the raising exporter
    assert mem.hooks()[0] == "run_start"     # the good exporter still saw everything
    assert mem.of("run_finish")[0]["status"] == "completed"


# ── export never runs on the agent's thread ─────────────────────────────────
class TestExportIsOffThread:
    """A slow exporter must cost the run nothing.

    This is the guarantee that was missing: exporter hooks and `flush()` ran
    inline, so a run waited on the monitoring backend's network. With no
    collector listening that was ~8.2s *per flush* on Windows, and flushes
    happen at every run finish.
    """

    class _SlowExporter(TraceExporter):
        name = "slow"

        def __init__(self, delay: float = 0.2) -> None:
            self.delay = delay
            self.seen: list[str] = []

        def _slow(self, hook: str) -> None:
            time.sleep(self.delay)
            self.seen.append(hook)

        def on_run_start(self, *a, **k):
            self._slow("run_start")

        def on_turn(self, *a, **k):
            self._slow("turn")

        def on_run_finish(self, *a, **k):
            self._slow("run_finish")

        def flush(self):
            self._slow("flush")

    @pytest.mark.asyncio
    async def test_a_slow_exporter_does_not_slow_the_run(self, tmp_path):
        slow = self._SlowExporter(delay=0.2)
        register_exporter(slow)

        agent = _oneshot(ScriptedProvider([("Paris.", [])]), tmp_path)
        started = time.perf_counter()
        result = await agent.complete("capital of France?")
        elapsed = time.perf_counter() - started

        assert result == "Paris."
        # At least run_start + turn + run_finish + flush would be >= 0.8s inline.
        assert elapsed < 0.4, f"the run waited on the exporter ({elapsed:.2f}s)"

        assert drain(timeout=10), "trace-export thread did not finish"
        assert "run_start" in slow.seen and "flush" in slow.seen, "work still ran"

    def test_hooks_arrive_in_order(self):
        """One worker on a FIFO: exporters keep per-run dicts and would corrupt
        if `on_run_finish` overtook `on_run_start`."""
        mem = MemoryExporter()
        ctx = TraceContext(metadata={"agent_type": "ReactAgent"})
        obs = TraceStreamObserver(ctx, [mem], model="m", name="ReactAgent.run")
        obs.start(input="hi")
        for _ in range(50):
            obs.handle(events.TurnCompleted(
                usage=Usage(input_tokens=1, output_tokens=1), stop_reason="end_turn"
            ))
        obs.close()
        assert drain(timeout=10)

        hooks = mem.hooks()
        assert hooks[0] == "run_start"
        assert hooks[-1] == "flush"
        assert hooks[-2] == "run_finish"
        assert hooks.count("turn") == 50

    def test_submit_never_raises_and_drain_reports(self):
        from neurosurfer.observability import dispatch

        boom = []

        def _explode() -> None:
            boom.append(1)
            raise RuntimeError("exporter blew up on the worker thread")

        dispatch.submit(_explode)          # must not raise here
        assert dispatch.drain(timeout=10)  # worker survived it
        assert boom == [1]

        marker = []
        dispatch.submit(lambda: marker.append("still alive"))
        assert dispatch.drain(timeout=10)
        assert marker == ["still alive"], "one bad task killed the worker"


# ── an exporter that is named but not configured ────────────────────────────
class TestUnconfiguredExporterIsSkipped:
    """`NEUROSURFER_EXPORTERS=otel` with no endpoint used to build an exporter
    anyway, and the OTel SDK filled in `http://localhost:4318` — so an install
    that had pointed at no collector still opened one, and paid ~8.2s to learn
    nothing was there. Auto-detection never had this problem: it turns `otel`
    on *because* the endpoint is set. The explicit list now agrees."""

    def test_otel_without_an_endpoint_is_not_built(self, caplog, monkeypatch):
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
        with caplog.at_level(logging.WARNING, logger="neurosurfer.observability"):
            assert configure_exporters(["otel"]) == []
        assert "not configured" in caplog.records[0].getMessage()

    def test_otel_with_an_endpoint_is_built(self, monkeypatch):
        pytest.importorskip("opentelemetry.exporter.otlp.proto.http.trace_exporter")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:9")
        assert [e.name for e in configure_exporters(["otel"])] == ["otel"]

    def test_langfuse_without_keys_is_not_built(self, caplog, monkeypatch):
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        with caplog.at_level(logging.WARNING, logger="neurosurfer.observability"):
            assert configure_exporters(["langfuse"]) == []

    def test_exporters_needing_nothing_still_build(self):
        assert [e.name for e in configure_exporters(["memory"])] == ["memory"]

    def test_a_constructed_instance_bypasses_the_check(self):
        """`register_exporter` takes something the caller built deliberately."""
        mem = MemoryExporter()
        register_exporter(mem)
        assert mem in get_active_exporters()


# ── the OTLP collector that isn't there ─────────────────────────────────────
class TestUnreachableCollector:
    """A collector that is not running warns once instead of raising per batch.

    ``OTLPSpanExporter.export`` re-raises a transport error (it retries a
    ``ConnectionError`` exactly once), and ``BatchSpanProcessor`` catches that
    with ``logger.exception``. So a dead endpoint printed a full traceback for
    *every* batch, none of whose frames name neurosurfer — tutorial 03's router
    cell emits a span per node, and its own output was buried under stacks
    pointing at ``urllib3``.

    No collector is contacted here: the unit tests drive a fake inner exporter,
    and the wiring test aims at ``127.0.0.1:9`` (discard), which refuses
    immediately — the same dead port the offline suite already uses for
    ``NEUROSURFER_TEST_BASE_URL``.
    """

    class _DeadExporter:
        """Stands in for ``OTLPSpanExporter`` pointed at nothing."""

        _endpoint = "http://localhost:4318/v1/traces"

        def __init__(self) -> None:
            self.calls = 0

        def _boom(self) -> None:
            self.calls += 1
            try:
                raise ConnectionRefusedError("[WinError 10061] actively refused it")
            except ConnectionRefusedError as e:
                raise RuntimeError("Max retries exceeded with url: /v1/traces") from e

        def export(self, spans):
            self._boom()

        def force_flush(self, timeout_millis=30_000):
            self._boom()

        def shutdown(self):
            self._boom()

    def _quiet(self):
        inner = self._DeadExporter()
        return inner, _QuietSpanExporter(inner, failure_result="FAILURE")

    def test_the_collector_is_tried_exactly_once(self, caplog):
        """The cost of a failed attempt is the reason, not tidiness.

        ``force_flush`` blocks the calling thread and runs at every run finish;
        on Windows one failed attempt is ~8.2s (a closed port is not refused
        instantly, and ``localhost`` is tried on both ``::1`` and
        ``127.0.0.1``, then retried). Probing again per batch is what turned a
        two-second workflow into a seventy-second one.
        """
        inner, quiet = self._quiet()
        with caplog.at_level(logging.WARNING, logger="neurosurfer.observability"):
            assert quiet.export(["span"]) == "FAILURE"   # no exception escapes
            assert quiet.export(["span"]) == "FAILURE"
            assert quiet.export(["span"]) == "FAILURE"

        assert inner.calls == 1, "the network must be touched once, then never again"
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1, "one warning for the session, not one per batch"
        assert warnings[0].exc_info is None, "a warning, not a traceback"

    def test_the_warning_says_where_why_how_long_and_how_to_stop_it(self, caplog):
        _, quiet = self._quiet()
        with caplog.at_level(logging.WARNING, logger="neurosurfer.observability"):
            quiet.export(["span"])
        msg = caplog.records[0].getMessage()

        assert "http://localhost:4318/v1/traces" in msg          # where
        assert "actively refused it" in msg                      # why — the root cause
        assert "blocked the run for" in msg                      # what it cost
        assert "disabled for" in msg                             # what happens now
        assert "OTEL_EXPORTER_OTLP_ENDPOINT" in msg              # how to fix
        assert "NEUROSURFER_EXPORTERS=none" in msg               # how to turn off

    def test_flush_and_shutdown_survive_a_dead_collector(self, caplog):
        inner, quiet = self._quiet()
        with caplog.at_level(logging.WARNING, logger="neurosurfer.observability"):
            assert quiet.force_flush() is False
            quiet.shutdown()                                     # returns, does not raise
        assert inner.calls == 1, "the first failure disables the rest"
        assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1

    def test_a_disabled_exporter_stops_costing_anything(self):
        """The point of disabling: later flushes must not reach the network."""
        class Slow:
            _endpoint = "http://localhost:4318/v1/traces"

            def __init__(self):
                self.calls = 0

            def export(self, spans):
                self.calls += 1
                time.sleep(0.25)                     # stands in for the ~8.2s connect
                raise ConnectionRefusedError("refused")

        inner = Slow()
        quiet = _QuietSpanExporter(inner, failure_result="FAILURE")
        first = time.perf_counter()
        quiet.export(["span"])
        first = time.perf_counter() - first

        rest = time.perf_counter()
        for _ in range(20):
            quiet.export(["span"])
        rest = time.perf_counter() - rest

        assert inner.calls == 1
        assert first >= 0.25                          # the one attempt really happened
        assert rest < first, "twenty later exports must cost less than one attempt"

    def test_a_healthy_collector_is_untouched(self, caplog):
        class Live:
            def export(self, spans):
                return "SUCCESS"

            def force_flush(self, timeout_millis=30_000):
                return True

        quiet = _QuietSpanExporter(Live(), failure_result="FAILURE")
        with caplog.at_level(logging.WARNING, logger="neurosurfer.observability"):
            assert quiet.export(["span"]) == "SUCCESS"
            assert quiet.force_flush() is True
        assert caplog.records == [], "a working exporter must say nothing"

    def test_root_cause_follows_both_chains(self):
        """``raise … from`` sets ``__cause__``; a bare raise sets ``__context__``.

        The real chain uses both, so following only one stops at a message that
        says "Max retries exceeded" and never reaches the reason.
        """
        try:
            try:
                raise ConnectionRefusedError("refused")
            except ConnectionRefusedError as inner:
                raise OSError("wrapped") from inner
        except OSError as ctx:
            try:
                raise RuntimeError("outer")          # bare raise → __context__
            except RuntimeError as e:
                assert ctx is e.__context__
                assert _root_cause(e) == "ConnectionRefusedError: refused"

    def test_root_cause_survives_a_cycle(self):
        a = ValueError("a")
        b = ValueError("b")
        a.__cause__ = b
        b.__cause__ = a                                # pathological, but must terminate
        assert _root_cause(a) == "ValueError: b"

    def test_batch_processor_emits_no_traceback(self, caplog, monkeypatch):
        """The guarantee, through the real SDK: a run against a dead endpoint
        produces our one warning and nothing from ``BatchSpanProcessor``."""
        pytest.importorskip("opentelemetry.exporter.otlp.proto.http.trace_exporter")
        from neurosurfer.observability.exporters.otel import OtelExporter

        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:9")
        exporter = OtelExporter(service_name="test")
        ctx = TraceContext(metadata={"agent_type": "Agent"})

        with caplog.at_level(logging.WARNING):
            exporter.on_run_start(ctx, name="Agent.run", input="hi")
            exporter.on_run_finish(ctx, status="completed", output="there")
            exporter.flush()
            exporter.close()

        otel_records = [r for r in caplog.records if r.name.startswith("opentelemetry")]
        assert otel_records == [], f"SDK still logging: {[r.getMessage() for r in otel_records]}"
        ours = [r for r in caplog.records if r.name == "neurosurfer.observability"]
        assert len(ours) == 1 and "not reachable" in ours[0].getMessage()
        assert ours[0].exc_info is None


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
            assert drain(), "trace-export thread did not finish"
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
