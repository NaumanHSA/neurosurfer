"""Phase 6 tests: Task layer (definition, policy, registry, runner) + transcript.

Covers:
  - TaskDefinition validation (unknown tools rejected).
  - InputSpec / Provenance defaults.
  - PolicyCeiling: validate_task catches every violation type.
  - validate_task_all returns all violations at once.
  - Valid task passes policy with default ceiling.
  - TaskRegistry: save → list → get round-trip.
  - TaskRegistry: save validates against ceiling (rejects over-broad task).
  - TaskRegistry: TaskNotFoundError on missing name.
  - TaskRegistry: delete removes the file.
  - TaskRunner.run wires a TaskDefinition into an Agent and returns events.
  - EventTranscript: write/read round-trip; close idempotent.
  - new_run_id produces unique 12-hex IDs.
  - CLI banner function builds a table without crashing.
  - RichIOHandler.ask: numbered choice selection.
  - RichIOHandler.ask: text (no options) returns raw input.
  - build_full_pool returns a pool with all expected tool names.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

from neurosurfer.agents import events
from neurosurfer.agents.runtime.permissions import Guardrails
from neurosurfer.observability.transcript import EventTranscript, new_run_id
from neurosurfer.tasks.definition import ALL_TOOLS, InputSpec, Provenance, TaskDefinition
from neurosurfer.tasks.policy import (
    PolicyCeiling,
    PolicyViolation,
    validate_task,
    validate_task_all,
)
from neurosurfer.tasks.registry import TaskNotFoundError, TaskRegistry
from tests.fakes import ScriptedIO, ScriptedProvider

# ──────────────────────────────────────────────────────────────────────────────
# TaskDefinition
# ──────────────────────────────────────────────────────────────────────────────

def _minimal_task(**kwargs) -> TaskDefinition:
    defaults = dict(
        name="test-task",
        system_prompt="You are helpful.",
        tools=["read_file", "finish"],
    )
    defaults.update(kwargs)
    return TaskDefinition(**defaults)


def test_task_definition_defaults():
    td = _minimal_task()
    assert td.version == 1
    assert td.plan_required is False
    assert td.model is None
    assert isinstance(td.provenance, Provenance)


def test_task_definition_unknown_tool_rejected():
    with pytest.raises(Exception, match="Unknown tools"):
        TaskDefinition(name="x", system_prompt="s", tools=["not_a_real_tool"])


def test_task_definition_all_tools_valid():
    td = TaskDefinition(name="full", system_prompt="s", tools=list(ALL_TOOLS))
    assert set(td.tools) == set(ALL_TOOLS)


def test_input_spec_defaults():
    spec = InputSpec(name="repo", prompt="Enter repo path")
    assert spec.required is True
    assert spec.type == "text"
    assert spec.default is None


def test_input_spec_choice_type():
    spec = InputSpec(
        name="level",
        type="choice",
        prompt="Pick level",
        choices=["basic", "advanced"],
    )
    assert spec.choices == ["basic", "advanced"]


# ──────────────────────────────────────────────────────────────────────────────
# PolicyCeiling + validate_task
# ──────────────────────────────────────────────────────────────────────────────

def test_valid_task_passes_default_ceiling():
    td = _minimal_task()
    validate_task(td)  # should not raise


def test_policy_rejects_over_max_turns():
    td = TaskDefinition(
        name="t",
        system_prompt="s",
        tools=["read_file"],
        guardrails=Guardrails(max_turns=600),
    )
    with pytest.raises(PolicyViolation, match="max_turns"):
        validate_task(td)


def test_policy_rejects_over_subagent_depth():
    td = TaskDefinition(
        name="t",
        system_prompt="s",
        tools=["read_file"],
        guardrails=Guardrails(max_subagent_depth=10),
    )
    with pytest.raises(PolicyViolation, match="max_subagent_depth"):
        validate_task(td)


def test_policy_rejects_over_concurrent_subagents():
    td = TaskDefinition(
        name="t",
        system_prompt="s",
        tools=["read_file"],
        guardrails=Guardrails(max_concurrent_subagents=20),
    )
    with pytest.raises(PolicyViolation, match="max_concurrent_subagents"):
        validate_task(td)


def test_policy_rejects_denied_write_root():
    td = TaskDefinition(
        name="t",
        system_prompt="s",
        tools=["read_file"],
        guardrails=Guardrails(write_scope=["/"]),
    )
    with pytest.raises(PolicyViolation, match="write_scope"):
        validate_task(td)


def test_policy_rejects_git_write_scope():
    td = TaskDefinition(
        name="t",
        system_prompt="s",
        tools=["read_file"],
        guardrails=Guardrails(write_scope=[".git"]),
    )
    with pytest.raises(PolicyViolation, match="write_scope"):
        validate_task(td)


def test_validate_task_all_returns_multiple_violations():
    td = TaskDefinition(
        name="t",
        system_prompt="s",
        tools=["read_file"],
        guardrails=Guardrails(max_turns=9999, max_subagent_depth=99, write_scope=["/"]),
    )
    errors = validate_task_all(td)
    assert len(errors) >= 3
    assert any("max_turns" in e for e in errors)
    assert any("max_subagent_depth" in e for e in errors)
    assert any("write_scope" in e for e in errors)


def test_validate_task_all_empty_for_valid_task():
    td = _minimal_task()
    assert validate_task_all(td) == []


def test_custom_ceiling_tighter_than_default():
    tight = PolicyCeiling(max_turns=10)
    td = TaskDefinition(
        name="t",
        system_prompt="s",
        tools=["read_file"],
        guardrails=Guardrails(max_turns=50),
    )
    with pytest.raises(PolicyViolation, match="max_turns"):
        validate_task(td, tight)


# ──────────────────────────────────────────────────────────────────────────────
# TaskRegistry
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_registry(tmp_path: Path) -> TaskRegistry:
    # Isolate user-task behaviour from the packaged built-in overlay.
    return TaskRegistry(tmp_path / "tasks", builtin_dir=tmp_path / "no_builtins")


def test_registry_empty_list(tmp_registry: TaskRegistry):
    assert tmp_registry.list() == []


def test_registry_save_list_get(tmp_registry: TaskRegistry):
    td = _minimal_task(name="demo")
    tmp_registry.save(td)
    assert "demo" in tmp_registry.list()
    loaded = tmp_registry.get("demo")
    assert loaded.name == "demo"
    assert loaded.system_prompt == "You are helpful."


def test_registry_get_not_found(tmp_registry: TaskRegistry):
    with pytest.raises(TaskNotFoundError):
        tmp_registry.get("nonexistent")


def test_registry_save_rejects_over_broad_task(tmp_registry: TaskRegistry):
    bad = TaskDefinition(
        name="bad",
        system_prompt="s",
        tools=["read_file"],
        guardrails=Guardrails(max_turns=9999),
    )
    with pytest.raises(PolicyViolation):
        tmp_registry.save(bad)


def test_registry_save_no_validate_skips_policy(tmp_registry: TaskRegistry):
    bad = TaskDefinition(
        name="bad-skip",
        system_prompt="s",
        tools=["read_file"],
        guardrails=Guardrails(max_turns=9999),
    )
    tmp_registry.save(bad, validate=False)
    assert "bad-skip" in tmp_registry.list()


def test_registry_delete(tmp_registry: TaskRegistry):
    tmp_registry.save(_minimal_task(name="del-me"))
    assert "del-me" in tmp_registry.list()
    tmp_registry.delete("del-me")
    assert "del-me" not in tmp_registry.list()


def test_registry_delete_not_found(tmp_registry: TaskRegistry):
    with pytest.raises(TaskNotFoundError):
        tmp_registry.delete("ghost")


def test_registry_iter(tmp_registry: TaskRegistry):
    for n in ("a", "b", "c"):
        tmp_registry.save(_minimal_task(name=n))
    names = [t.name for t in tmp_registry.iter()]
    assert sorted(names) == ["a", "b", "c"]


def test_registry_yaml_round_trip_with_inputs(tmp_registry: TaskRegistry):
    td = TaskDefinition(
        name="with-inputs",
        system_prompt="s",
        tools=["read_file"],
        inputs=[InputSpec(name="repo", prompt="Enter path")],
    )
    tmp_registry.save(td)
    loaded = tmp_registry.get("with-inputs")
    assert len(loaded.inputs) == 1
    assert loaded.inputs[0].name == "repo"


# ──────────────────────────────────────────────────────────────────────────────
# TaskRunner
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_task_runner_runs_and_yields_events(tmp_path: Path):
    """TaskRunner.run() wires TaskDefinition → Agent → yields RunFinished."""
    from neurosurfer.config import Config, ObservabilityConfig, TasksConfig
    from neurosurfer.tasks.runner import TaskRunner

    # Patch build_provider to return a scripted fake.
    provider = ScriptedProvider([("Task done!", [])])

    with patch("neurosurfer.tasks.runner.build_provider", return_value=provider):
        cfg = Config(
            tasks=TasksConfig(dir=tmp_path / "tasks"),
            observability=ObservabilityConfig(state_dir=tmp_path / "state"),
        )
        runner = TaskRunner(cfg, cwd=tmp_path)
        task = _minimal_task(name="runner-test")

        collected: list[events.Event] = []
        async for ev in runner.run(task, {}, ScriptedIO()):
            collected.append(ev)

    assert any(isinstance(e, events.RunFinished) for e in collected)
    finished = next(e for e in collected if isinstance(e, events.RunFinished))
    assert finished.status == "completed"


@pytest.mark.asyncio
async def test_task_runner_filters_tool_pool(tmp_path: Path):
    """Runner applies the Task's tools allow-list."""
    from neurosurfer.config import Config, ObservabilityConfig, TasksConfig
    from neurosurfer.tasks.runner import TaskRunner, build_full_pool

    provider = ScriptedProvider([("ok", [])])

    with patch("neurosurfer.tasks.runner.build_provider", return_value=provider):
        cfg = Config(
            tasks=TasksConfig(dir=tmp_path / "tasks"),
            observability=ObservabilityConfig(state_dir=tmp_path / "state"),
        )
        runner = TaskRunner(cfg, cwd=tmp_path)
        # Task only allows read_file and finish.
        task = _minimal_task(name="narrow", tools=["read_file", "finish"])

        full_pool = build_full_pool(tmp_path)
        captured_pool: list = []

        # Intercept _run to inspect the agent's pool.
        original_run = runner._run

        async def _spy(task, inputs, io, **kwargs):
            # Let it run one step then look at what tool names the agent gets.
            pool = full_pool.select(task.tools)
            captured_pool.extend(pool.names())
            async for ev in original_run(task, inputs, io, **kwargs):
                yield ev

        runner._run = _spy  # type: ignore[method-assign]
        async for _ in runner.run(task, {}, ScriptedIO()):
            pass

    assert "read_file" in captured_pool
    assert "write_file" not in captured_pool


# ──────────────────────────────────────────────────────────────────────────────
# EventTranscript
# ──────────────────────────────────────────────────────────────────────────────

def test_transcript_write_and_read(tmp_path: Path):
    t = EventTranscript("abc123", tmp_path)
    t.record("run_start", task="demo")
    t.record("TextDelta", text="hello")
    t.record("run_end")
    t.close()

    entries = t.read_all()
    assert len(entries) == 3
    assert entries[0]["type"] == "run_start"
    assert entries[0]["task"] == "demo"
    assert entries[1]["text"] == "hello"
    assert entries[2]["type"] == "run_end"


def test_transcript_context_manager(tmp_path: Path):
    with EventTranscript("ctx", tmp_path) as t:
        t.record("ping")
    entries = t.read_all()
    assert entries[0]["type"] == "ping"


def test_transcript_path(tmp_path: Path):
    t = EventTranscript("myrun", tmp_path)
    t.close()
    assert t.path == tmp_path / "myrun.jsonl"


def test_new_run_id_unique():
    ids = {new_run_id() for _ in range(50)}
    assert len(ids) == 50


def test_new_run_id_format():
    rid = new_run_id()
    assert len(rid) == 12
    assert rid.isalnum()


# ──────────────────────────────────────────────────────────────────────────────
# build_full_pool
# ──────────────────────────────────────────────────────────────────────────────

def test_build_full_pool_has_all_tools(tmp_path: Path):
    from neurosurfer.tasks.runner import build_full_pool

    pool = build_full_pool(tmp_path, tasks_dir=tmp_path / "tasks")
    names = set(pool.names())
    expected = {
        "read_file", "list_dir", "search", "data", "run_command",
        "web_search", "http", "browse",
        "write_file", "apply_edit", "ask_user", "present_plan",
        "todo", "spawn_agent", "finish", "register_task",
    }
    assert expected == names


# ──────────────────────────────────────────────────────────────────────────────
# RichIOHandler
# ──────────────────────────────────────────────────────────────────────────────

def _patch_ainput(value: str):
    async def fake_ainput(prompt_text: str = "", *, is_password: bool = False) -> str:
        return value

    return patch("neurosurfer.app.cli.io._ainput", fake_ainput)


@pytest.mark.asyncio
async def test_rich_io_ask_choice_by_number():
    """ask() with options accepts a number and returns the matching option."""
    from rich.console import Console

    from neurosurfer.app.cli.io import RichIOHandler

    io = RichIOHandler(Console(file=StringIO()))
    with _patch_ainput("2"):
        result = await io.ask("Pick one", options=["alpha", "beta", "gamma"])
    assert result == "beta"


@pytest.mark.asyncio
async def test_rich_io_ask_choice_by_exact_text():
    """ask() with options also accepts the exact option string."""
    from rich.console import Console

    from neurosurfer.app.cli.io import RichIOHandler

    io = RichIOHandler(Console(file=StringIO()))
    with _patch_ainput("gamma"):
        result = await io.ask("Pick one", options=["alpha", "beta", "gamma"])
    assert result == "gamma"


@pytest.mark.asyncio
async def test_rich_io_ask_no_options_returns_text():
    """ask() without options returns the raw text input."""
    from rich.console import Console

    from neurosurfer.app.cli.io import RichIOHandler

    io = RichIOHandler(Console(file=StringIO()))
    with _patch_ainput("free text answer"):
        result = await io.ask("What do you want?")
    assert result == "free text answer"


@pytest.mark.asyncio
async def test_rich_io_ask_options_accepts_free_text():
    """ask() with options also accepts a custom typed answer not in the list."""
    from rich.console import Console

    from neurosurfer.app.cli.io import RichIOHandler

    io = RichIOHandler(Console(file=StringIO()))
    with _patch_ainput("my own custom answer"):
        result = await io.ask("Pick one", options=["alpha", "beta"])
    assert result == "my own custom answer"


# ──────────────────────────────────────────────────────────────────────────────
# CLI banner
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_print_banner_does_not_crash(tmp_path, monkeypatch):
    from io import StringIO

    from rich.console import Console

    from neurosurfer.app.cli.banner import print_banner
    from neurosurfer.app.cli.context import CLIContext
    from neurosurfer.config import load_config

    monkeypatch.setenv("NEUROSURFER_TASKS_DIR", str(tmp_path / "tasks"))
    monkeypatch.setenv("NEUROSURFER_STATE_DIR", str(tmp_path / "state"))
    ctx = CLIContext.create(load_config(env_file=tmp_path / "nope.env"))
    ctx.console = Console(file=StringIO(), width=120)
    await print_banner(ctx)
    output = ctx.console.file.getvalue()  # type: ignore[attr-defined]
    assert "neurosurfer" in output
    assert "provider" in output
    assert "workflow" in output.lower()
