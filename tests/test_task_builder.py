"""Phase 8 tests: register_task tool + task_builder meta-agent.

Covers:
  - register_task saves a valid TaskDefinition to the user registry.
  - register_task rejects unknown tools (definition validation).
  - register_task rejects policy violations (ceiling enforcement).
  - register_task rejects denied write_scope roots.
  - register_task overwrites an existing task (refinement/versioning).
  - task_builder YAML is discoverable via TaskRegistry.
  - task_builder YAML passes policy validation.
  - task_builder has exactly the expected tool set.
  - task_builder YAML parses into a valid TaskDefinition.
  - system-prompt assembly wraps task body with base sections.
  - E2E: ScriptedProvider calls register_task → task in registry, passes policy.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from neurosurfer.tasks.definition import ALL_TOOLS, TaskDefinition
from neurosurfer.tasks.policy import validate_task
from neurosurfer.tasks.registry import TaskRegistry
from tests.fakes import ScriptedIO, ScriptedProvider

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_ctx(tasks_dir: Path):
    """Return a minimal ToolContext with a writable tasks_dir."""

    from neurosurfer.agents.runtime.permissions import Guardrails
    from neurosurfer.tools.base import ToolContext

    return ToolContext(cwd=tasks_dir, io=ScriptedIO(), guardrails=Guardrails())


def _make_args(**kwargs):
    """Build RegisterTaskArgs with sensible defaults."""
    from neurosurfer.app.tools.register_task import RegisterTaskArgs

    defaults = dict(
        name="my-task",
        description="A test task",
        system_prompt="Do the thing.",
        tools=["read_file", "ask_user", "finish"],
    )
    defaults.update(kwargs)
    return RegisterTaskArgs(**defaults)


# ──────────────────────────────────────────────────────────────────────────────
# register_task tool — unit tests
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_register_task_saves_to_registry(tmp_path: Path):
    from neurosurfer.app.tools.register_task import RegisterTaskTool

    tool = RegisterTaskTool(tmp_path / "tasks")
    ctx = _make_ctx(tmp_path)
    result = await tool.call(_make_args(), ctx)

    assert not result.is_error
    assert "my-task" in result.content
    assert (tmp_path / "tasks" / "my-task.yaml").exists()


@pytest.mark.asyncio
async def test_register_task_stored_yaml_is_loadable(tmp_path: Path):
    from neurosurfer.app.tools.register_task import RegisterTaskTool

    tool = RegisterTaskTool(tmp_path / "tasks")
    ctx = _make_ctx(tmp_path)
    await tool.call(_make_args(name="load-test", description="desc", system_prompt="sp"), ctx)

    registry = TaskRegistry(tmp_path / "tasks")
    td = registry.get("load-test")
    assert td.name == "load-test"
    assert td.description == "desc"
    assert td.system_prompt == "sp"


@pytest.mark.asyncio
async def test_register_task_drops_unknown_tools(tmp_path: Path):
    """Unknown tool names are filtered out; the task still registers."""
    from neurosurfer.app.tools.register_task import RegisterTaskTool

    tool = RegisterTaskTool(tmp_path / "tasks")
    ctx = _make_ctx(tmp_path)
    result = await tool.call(_make_args(tools=["read_file", "not_a_tool"]), ctx)

    assert not result.is_error
    td = TaskRegistry(tmp_path / "tasks").get("my-task")
    assert "not_a_tool" not in td.tools
    assert "read_file" in td.tools


@pytest.mark.asyncio
async def test_register_task_clamps_over_ceiling_max_turns(tmp_path: Path):
    """max_turns above the policy ceiling is clamped, not rejected."""
    from neurosurfer.app.tools.register_task import (
        RegisterTaskArgs,
        RegisterTaskTool,
        _GuardrailsInput,
    )

    tool = RegisterTaskTool(tmp_path / "tasks")
    ctx = _make_ctx(tmp_path)
    args = RegisterTaskArgs(
        name="clamped",
        description="d",
        system_prompt="s",
        tools=["finish"],
        guardrails=_GuardrailsInput(max_turns=9999),
    )
    result = await tool.call(args, ctx)

    assert not result.is_error
    td = TaskRegistry(tmp_path / "tasks").get("clamped")
    assert td.guardrails.max_turns == 500  # ceiling


@pytest.mark.asyncio
async def test_register_task_drops_denied_write_root(tmp_path: Path):
    """A '/' write scope is dropped (security), and the task still registers."""
    from neurosurfer.app.tools.register_task import (
        RegisterTaskArgs,
        RegisterTaskTool,
        _GuardrailsInput,
    )

    tool = RegisterTaskTool(tmp_path / "tasks")
    ctx = _make_ctx(tmp_path)
    args = RegisterTaskArgs(
        name="rootwriter",
        description="d",
        system_prompt="s",
        tools=["write_file", "finish"],
        guardrails=_GuardrailsInput(write_scope=["/", "out/"]),
    )
    result = await tool.call(args, ctx)

    assert not result.is_error
    td = TaskRegistry(tmp_path / "tasks").get("rootwriter")
    assert "/" not in td.guardrails.write_scope
    assert "out/" in td.guardrails.write_scope


@pytest.mark.asyncio
async def test_register_task_overwrites_existing(tmp_path: Path):
    from neurosurfer.app.tools.register_task import RegisterTaskTool

    tool = RegisterTaskTool(tmp_path / "tasks")
    ctx = _make_ctx(tmp_path)

    await tool.call(_make_args(name="overwrite-me", description="v1"), ctx)
    await tool.call(_make_args(name="overwrite-me", description="v2"), ctx)

    registry = TaskRegistry(tmp_path / "tasks")
    td = registry.get("overwrite-me")
    assert td.description == "v2"


@pytest.mark.asyncio
async def test_register_task_result_contains_run_hint(tmp_path: Path):
    from neurosurfer.app.tools.register_task import RegisterTaskTool

    tool = RegisterTaskTool(tmp_path / "tasks")
    result = await tool.call(_make_args(name="hint-task"), _make_ctx(tmp_path))

    assert not result.is_error
    assert "/task run hint-task" in result.content


@pytest.mark.asyncio
async def test_register_task_sets_provenance_created_by(tmp_path: Path):
    from neurosurfer.app.tools.register_task import RegisterTaskTool

    tool = RegisterTaskTool(tmp_path / "tasks")
    await tool.call(_make_args(name="prov-task"), _make_ctx(tmp_path))

    registry = TaskRegistry(tmp_path / "tasks")
    td = registry.get("prov-task")
    assert td.provenance.created_by == "task_builder"


@pytest.mark.asyncio
async def test_register_task_coerces_loose_input_fields(tmp_path: Path):
    """Loose input phrasing from a small model is normalised, not rejected."""
    from neurosurfer.app.tools.register_task import RegisterTaskArgs, RegisterTaskTool

    tool = RegisterTaskTool(tmp_path / "tasks")
    ctx = _make_ctx(tmp_path)
    # 'url' type and a 'question' key instead of 'prompt'.
    args = RegisterTaskArgs.model_validate({
        "name": "loose",
        "description": "d",
        "system_prompt": "s",
        "tools": ["read_file", "finish"],
        "inputs": [{"name": "repo", "type": "url", "question": "Which repo?"}],
    })
    result = await tool.call(args, ctx)

    assert not result.is_error
    td = TaskRegistry(tmp_path / "tasks").get("loose")
    assert td.inputs[0].type == "path_or_url"
    assert td.inputs[0].prompt == "Which repo?"


# ──────────────────────────────────────────────────────────────────────────────
# todo tool — lenient status / shape coercion (small-model robustness)
# ──────────────────────────────────────────────────────────────────────────────

def test_todo_coerces_status_aliases():
    from neurosurfer.tools.builtin.todo import TodoArgs

    args = TodoArgs.model_validate({"todos": [
        {"content": "a", "status": "done"},
        {"content": "b", "status": "in-progress"},
        {"content": "c", "status": "todo"},
    ]})
    assert [t.status for t in args.todos] == ["completed", "in_progress", "pending"]


def test_todo_accepts_bare_strings_and_alt_keys():
    from neurosurfer.tools.builtin.todo import TodoArgs

    # A bare list, items as strings or with a 'task' key.
    args = TodoArgs.model_validate(["write tests", {"task": "run them"}])
    assert args.todos[0].content == "write tests"
    assert args.todos[1].content == "run them"


# ──────────────────────────────────────────────────────────────────────────────
# task_builder builtin YAML
# ──────────────────────────────────────────────────────────────────────────────

def test_task_builder_is_discoverable():
    registry = TaskRegistry(Path("/tmp/nonexistent-user-tasks-8xyz"))
    assert "task_builder" in registry.list()


def test_task_builder_passes_policy():
    registry = TaskRegistry(Path("/tmp/nonexistent-user-tasks-8xyz"))
    td = registry.get("task_builder")
    validate_task(td)  # should not raise


def test_task_builder_has_expected_tools():
    registry = TaskRegistry(Path("/tmp/nonexistent-user-tasks-8xyz"))
    td = registry.get("task_builder")
    tool_set = set(td.tools)
    assert "ask_user" in tool_set
    assert "present_plan" in tool_set
    assert "register_task" in tool_set
    assert "finish" in tool_set
    # Must NOT include dangerous tools
    assert "run_command" not in tool_set
    assert "write_file" not in tool_set
    assert "apply_edit" not in tool_set


def test_task_builder_shell_policy_is_denied():
    registry = TaskRegistry(Path("/tmp/nonexistent-user-tasks-8xyz"))
    td = registry.get("task_builder")
    assert td.guardrails.shell_policy == "denied"


def test_task_builder_no_write_scope():
    registry = TaskRegistry(Path("/tmp/nonexistent-user-tasks-8xyz"))
    td = registry.get("task_builder")
    assert td.guardrails.write_scope == []


def test_task_builder_plan_not_required():
    """task_builder drives its own plan gate internally via present_plan."""
    registry = TaskRegistry(Path("/tmp/nonexistent-user-tasks-8xyz"))
    td = registry.get("task_builder")
    assert td.plan_required is False


def test_register_task_in_all_tools():
    assert "register_task" in ALL_TOOLS


# ──────────────────────────────────────────────────────────────────────────────
# System-prompt standardisation
# ──────────────────────────────────────────────────────────────────────────────

def test_standardised_system_prompt_contains_base_sections(tmp_path: Path):
    """TaskRunner wraps the task body with identity/tone/guardrails/env sections."""
    from neurosurfer.agents.runtime.permissions import Guardrails
    from neurosurfer.tasks.runner import _build_system_prompt

    task = TaskDefinition(
        name="t",
        system_prompt="Do the specific thing.",
        tools=["finish"],
    )
    prompt = _build_system_prompt(
        task,
        {},
        guardrails=Guardrails(),
        cwd=tmp_path,
        model="claude-opus-4-8",
    )

    assert "neurosurfer" in prompt          # BASE_IDENTITY
    assert "Do the specific thing." in prompt  # task body preserved
    assert "Guardrails" in prompt             # guardrail section
    assert "Environment" in prompt            # env section


def test_standardised_prompt_substitutes_output_dir(tmp_path: Path):
    from neurosurfer.agents.runtime.permissions import Guardrails
    from neurosurfer.tasks.runner import _build_system_prompt

    task = TaskDefinition(
        name="t",
        system_prompt="Write to <output_dir>index.md.",  # <output_dir> includes trailing slash
        tools=["write_file", "finish"],
    )
    prompt = _build_system_prompt(
        task,
        {"output_dir": "reports/"},
        guardrails=Guardrails(),
        cwd=tmp_path,
        model="scripted",
    )
    assert "reports/index.md" in prompt
    assert "<output_dir>" not in prompt


# ──────────────────────────────────────────────────────────────────────────────
# E2E: task_builder run via ScriptedProvider → task registered + runnable
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_task_builder_e2e_registers_task(tmp_path: Path):
    """Full e2e: ScriptedProvider calls register_task → task appears in registry.

    The scripted provider skips the interview phases and jumps straight to
    register_task (simulating a completed interview), then calls finish.
    """
    from neurosurfer.config import Config, ObservabilityConfig, TasksConfig
    from neurosurfer.agents import events
    from neurosurfer.tasks.registry import TaskRegistry
    from neurosurfer.tasks.runner import TaskRunner

    tasks_dir = tmp_path / "tasks"
    register_args = {
        "name": "e2e-task",
        "description": "Created by e2e test",
        "system_prompt": "Do the e2e thing.",
        "tools": ["read_file", "ask_user", "finish"],
        "guardrails": {"shell_policy": "denied", "write_scope": [], "max_turns": 50},
    }
    finish_args = {"summary": "Registered e2e-task successfully.", "status": "success"}

    provider = ScriptedProvider([
        # Turn 1: agent calls register_task (skips interview for scripted test)
        ("I'll register the task now.", [("register_task", register_args)]),
        # Turn 2: agent calls finish
        ("Done!", [("finish", finish_args)]),
    ])

    with patch("neurosurfer.tasks.runner.build_provider", return_value=provider):
        cfg = Config(
            tasks=TasksConfig(dir=tasks_dir),
            observability=ObservabilityConfig(state_dir=tmp_path / "state"),
        )
        registry = TaskRegistry(tasks_dir)
        # Run task_builder
        tb_task = registry.get("task_builder")
        runner = TaskRunner(cfg, cwd=tmp_path)

        collected: list[events.Event] = []
        async for ev in runner.run(tb_task, {}, ScriptedIO()):
            collected.append(ev)

    # The run must complete
    assert any(isinstance(e, events.RunFinished) for e in collected)

    # The new task must be in the user registry
    user_registry = TaskRegistry(tasks_dir)
    assert "e2e-task" in user_registry.list()

    # The registered task must pass policy
    td = user_registry.get("e2e-task")
    validate_task(td)
    assert td.description == "Created by e2e test"
    assert td.provenance.created_by == "task_builder"


@pytest.mark.asyncio
async def test_task_builder_e2e_clamps_over_ceiling_guardrails(tmp_path: Path):
    """Over-ceiling guardrails are sanitised at registration, not rejected."""
    from neurosurfer.config import Config, ObservabilityConfig, TasksConfig
    from neurosurfer.tasks.registry import TaskRegistry
    from neurosurfer.tasks.runner import TaskRunner

    tasks_dir = tmp_path / "tasks"
    over_args = {
        "name": "over-task",
        "description": "Over ceiling",
        "system_prompt": "s",
        "tools": ["finish"],
        "guardrails": {"shell_policy": "denied", "write_scope": [], "max_turns": 9999},
    }
    finish_args = {"summary": "Registered with clamped caps.", "status": "success"}

    provider = ScriptedProvider([
        ("Registering...", [("register_task", over_args)]),
        ("Done.", [("finish", finish_args)]),
    ])

    with patch("neurosurfer.tasks.runner.build_provider", return_value=provider):
        cfg = Config(
            tasks=TasksConfig(dir=tasks_dir),
            observability=ObservabilityConfig(state_dir=tmp_path / "state"),
        )
        registry = TaskRegistry(tasks_dir)
        tb_task = registry.get("task_builder")
        runner = TaskRunner(cfg, cwd=tmp_path)

        async for _ in runner.run(tb_task, {}, ScriptedIO()):
            pass

    user_registry = TaskRegistry(tasks_dir)
    assert "over-task" in user_registry.list()
    td = user_registry.get("over-task")
    validate_task(td)  # clamped → passes policy
    assert td.guardrails.max_turns == 500
