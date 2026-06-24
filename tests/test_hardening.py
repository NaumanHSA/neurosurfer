"""Phase 9 — hardening tests: interrupt-persist + resume, cache audit.

- Interrupt + resume: the runner persists durable state each turn; a resume
  reloads task/inputs (from the transcript) + durable state and continues with
  the plan gate already satisfied.
- Prompt-cache audit (Anthropic): cache_control breakpoints are emitted on the
  system block and the last tool, and cache_read usage is parsed back.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from neurosurfer.agents import events
from neurosurfer.agents.context.durable_state import DurableState

from .fakes import ScriptedIO, ScriptedProvider


# ── interrupt persistence + resume ────────────────────────────────────────────
def _resumable_task(**kwargs):
    from neurosurfer.tasks.definition import TaskDefinition

    defaults: dict = dict(
        name="resumable",
        description="t",
        system_prompt="Do the task.",
        tools=["todo", "present_plan", "finish"],
        plan_required=True,
    )
    defaults.update(kwargs)
    return TaskDefinition(**defaults)


@pytest.mark.asyncio
async def test_runner_persists_durable_state(tmp_path: Path):
    """Each turn writes the durable snapshot to runs_state_dir/<run_id>.json."""
    from unittest.mock import patch

    from neurosurfer.config import Config, ObservabilityConfig, TasksConfig
    from neurosurfer.tasks.runner import TaskRunner

    # Turn 1 records a todo (mutates durable), turn 2 finishes.
    turns = [
        ("planning", [("todo", {"items": [{"content": "do x", "status": "pending"}]})]),
        ("done", [("finish", {"summary": "ok", "status": "success"})]),
    ]
    provider = ScriptedProvider(turns)
    cfg = Config(
        tasks=TasksConfig(dir=tmp_path / "tasks"),
        observability=ObservabilityConfig(state_dir=tmp_path / "state"),
    )
    cfg.ensure_dirs()

    with patch("neurosurfer.tasks.runner.build_provider", return_value=provider):
        runner = TaskRunner(cfg, cwd=tmp_path)
        task = _resumable_task(plan_required=False)
        async for _ in runner.run(task, {}, ScriptedIO(), run_id="abc123"):
            pass

    state_file = cfg.observability.runs_state_dir() / "abc123.json"
    assert state_file.exists()
    restored = DurableState.load(state_file)
    assert restored.todos and restored.todos[0]["content"] == "do x"


@pytest.mark.asyncio
async def test_resume_reloads_durable_and_skips_plan_gate(tmp_path: Path):
    """Resume injects prior durable state and starts in default (not plan) mode."""
    from unittest.mock import patch

    from neurosurfer.config import Config, ObservabilityConfig, TasksConfig
    from neurosurfer.tasks.runner import TaskRunner

    cfg = Config(
        tasks=TasksConfig(dir=tmp_path / "tasks"),
        observability=ObservabilityConfig(state_dir=tmp_path / "state"),
    )
    cfg.ensure_dirs()

    # Pre-existing durable state with an approved plan (as if interrupted mid-run).
    durable = DurableState()
    durable.set_plan("Approved Plan", "1. write docs")
    durable.set_todos([{"content": "write docs", "status": "in_progress"}])

    # On resume the agent should be free to write/finish immediately (no plan gate).
    turns = [("continuing", [("finish", {"summary": "resumed and finished", "status": "success"})])]
    provider = ScriptedProvider(turns)

    captured: dict = {}
    from neurosurfer.agents.agentic_loop import AgenticLoop as RealAgent

    def _spy_agent(*args, **kwargs):
        captured["mode"] = kwargs.get("mode")
        captured["durable"] = kwargs.get("durable")
        return RealAgent(*args, **kwargs)

    with patch("neurosurfer.tasks.runner.build_provider", return_value=provider), \
         patch("neurosurfer.tasks.runner.AgenticLoop", side_effect=_spy_agent):
        runner = TaskRunner(cfg, cwd=tmp_path)
        task = _resumable_task(plan_required=True)
        evs = [
            ev
            async for ev in runner.run(
                task, {}, ScriptedIO(approve_plan=True),
                run_id="run9", durable=durable, resume=True,
            )
        ]

    # Plan gate skipped because a plan was already approved.
    assert captured["mode"] == "default"
    # Prior durable state was injected (same object continues to be used).
    assert captured["durable"].plan_text == "1. write docs"
    assert any(isinstance(e, events.RunFinished) and e.status == "success" for e in evs)


def test_read_run_meta_and_list_resumable(tmp_path: Path):
    """CLI helpers recover task/inputs from a transcript and list saved runs."""
    from neurosurfer.app.cli.context import CLIContext
    from neurosurfer.app.cli.run import _read_run_meta, list_resumable
    from neurosurfer.config import Config, ObservabilityConfig, TasksConfig
    from neurosurfer.observability.transcript import EventTranscript

    cfg = Config(
        tasks=TasksConfig(dir=tmp_path / "tasks"),
        observability=ObservabilityConfig(state_dir=tmp_path / "state"),
    )
    cfg.ensure_dirs()
    ctx = CLIContext.create(cfg)

    with EventTranscript("runX", cfg.observability.transcripts_dir()) as t:
        t.record("run_start", task="code", inputs={"repo": "/tmp/x"}, cwd="/tmp/x")
    DurableState().save(cfg.observability.runs_state_dir() / "runX.json")

    assert _read_run_meta(ctx, "runX") == ("code", {"repo": "/tmp/x"})
    assert _read_run_meta(ctx, "missing") is None
    assert "runX" in list_resumable(ctx)


# ── prompt-cache audit (Anthropic) ────────────────────────────────────────────
def test_anthropic_cache_breakpoints_emitted():
    from neurosurfer.llm.providers.anthropic import _system_param, to_anthropic_tools
    from neurosurfer.llm.types import ToolSchema

    sys_param = _system_param("You are an agent.")
    assert sys_param is not None
    assert sys_param[0]["cache_control"] == {"type": "ephemeral"}

    tools = [
        ToolSchema(name="a", description="d", input_schema={"type": "object"}),
        ToolSchema(name="b", description="d", input_schema={"type": "object"}),
    ]
    rendered = to_anthropic_tools(tools)
    # Only the last tool carries the cache breakpoint.
    assert "cache_control" not in rendered[0]
    assert rendered[-1]["cache_control"] == {"type": "ephemeral"}


def test_anthropic_parses_cache_read_usage():
    from types import SimpleNamespace

    from neurosurfer.llm.providers.anthropic import AnthropicProvider

    p = AnthropicProvider(api_key="test", model="claude-opus-4-8")
    final = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="hi")],
        stop_reason="end_turn",
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            cache_read_input_tokens=1234,
            cache_creation_input_tokens=42,
        ),
    )
    resp = p._final_to_response(final)
    assert resp.usage.cache_read_input_tokens == 1234
    assert resp.usage.cache_creation_input_tokens == 42
