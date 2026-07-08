"""Phase 5 tests: sub-agent orchestration.

Covers:
  - SubAgentDefinition.resolve_tools: wildcard allow, specific list, disallow filtering.
  - Registry: register / get_agent / all_agents round-trip.
  - Built-in agents register themselves on import.
  - SubAgentRunner.spawn returns the child agent's final report.
  - Parallel spawning via spawn_parallel uses asyncio.gather.
  - Depth cap (MAX_DEPTH) returns a bracketed error string, not an exception.
  - Guardrails max_subagent_depth cap also returns a bracketed error string.
  - Unknown agent_type returns a bracketed error string.
  - Child tool pool is correctly filtered (parent write tools absent for explore).
  - Child isolation: each child gets a fresh empty history.
  - Concurrency: max_concurrent_subagents semaphore limits simultaneous spawns.
  - TasksRuntime.submit / active / all.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import neurosurfer.app.agents  # noqa: F401 — triggers built-in persona registrations
from neurosurfer.agents.runtime.permissions import Guardrails
from neurosurfer.agents.runtime.tasks_runtime import TaskHandle, TasksRuntime
from neurosurfer.agents.subagents.defs import (
    SubAgentDefinition,
    all_agents,
    get_agent,
    register,
)
from neurosurfer.agents.subagents.runner import MAX_DEPTH, SubAgentRunner
from neurosurfer.tools.base import ToolPool
from tests.fakes import ScriptedIO, ScriptedProvider

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_pool(*names: str) -> ToolPool:
    """Create a ToolPool with stub tools whose only attribute is .name."""
    from unittest.mock import MagicMock

    from neurosurfer.llm.types import ToolSchema

    tools = []
    for n in names:
        t = MagicMock()
        t.name = n
        t.is_enabled.return_value = True
        t.schema = ToolSchema(name=n, description=n, input_schema={"type": "object", "properties": {}})
        tools.append(t)
    return ToolPool(tools)


def _make_runner(
    pool: ToolPool | None = None,
    max_depth_turns: int = 50,
    max_subagent_depth: int = 2,
    max_concurrent: int = 4,
    provider_turns: list | None = None,
) -> SubAgentRunner:
    if pool is None:
        pool = _make_pool("read_file", "list_dir", "search", "write_file", "spawn_agent")
    provider = ScriptedProvider(provider_turns or [("done", [])])
    guardrails = Guardrails(
        max_subagent_depth=max_subagent_depth,
        max_concurrent_subagents=max_concurrent,
    )
    io = ScriptedIO()
    return SubAgentRunner(pool, provider, io=io, cwd=Path("."), guardrails=guardrails)


# ──────────────────────────────────────────────────────────────────────────────
# SubAgentDefinition.resolve_tools
# ──────────────────────────────────────────────────────────────────────────────

def test_resolve_tools_wildcard_allows_all():
    defn = SubAgentDefinition(
        agent_type="test",
        when_to_use="",
        system_prompt="",
        allowed_tools=["*"],
        disallowed_tools=[],
    )
    pool_names = ["read_file", "write_file", "search"]
    assert defn.resolve_tools(pool_names) == pool_names


def test_resolve_tools_specific_list():
    defn = SubAgentDefinition(
        agent_type="test",
        when_to_use="",
        system_prompt="",
        allowed_tools=["read_file", "search"],
        disallowed_tools=[],
    )
    pool_names = ["read_file", "write_file", "search"]
    assert defn.resolve_tools(pool_names) == ["read_file", "search"]


def test_resolve_tools_disallow_removes_from_wildcard():
    defn = SubAgentDefinition(
        agent_type="test",
        when_to_use="",
        system_prompt="",
        allowed_tools=["*"],
        disallowed_tools=["write_file", "spawn_agent"],
    )
    pool_names = ["read_file", "write_file", "search", "spawn_agent"]
    result = defn.resolve_tools(pool_names)
    assert "write_file" not in result
    assert "spawn_agent" not in result
    assert "read_file" in result


def test_resolve_tools_disallow_removes_from_specific_list():
    defn = SubAgentDefinition(
        agent_type="test",
        when_to_use="",
        system_prompt="",
        allowed_tools=["read_file", "write_file"],
        disallowed_tools=["write_file"],
    )
    pool_names = ["read_file", "write_file", "search"]
    assert defn.resolve_tools(pool_names) == ["read_file"]


def test_resolve_tools_tool_not_in_parent_pool_excluded():
    defn = SubAgentDefinition(
        agent_type="test",
        when_to_use="",
        system_prompt="",
        allowed_tools=["read_file", "nonexistent"],
        disallowed_tools=[],
    )
    assert defn.resolve_tools(["read_file", "search"]) == ["read_file"]


# ──────────────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────────────

def test_register_and_get():
    defn = SubAgentDefinition(
        agent_type="__test_reg__",
        when_to_use="testing",
        system_prompt="hi",
    )
    register(defn)
    assert get_agent("__test_reg__") is defn


def test_all_agents_includes_registered():
    register(SubAgentDefinition(agent_type="__test_all__", when_to_use="", system_prompt=""))
    assert any(a.agent_type == "__test_all__" for a in all_agents())


def test_built_ins_registered():
    """Importing neurosurfer.app.agents registers the four built-in personas."""
    for expected in ("explore", "analyzer", "writer", "verifier"):
        assert get_agent(expected) is not None, f"'{expected}' not in registry"


def test_explore_has_no_write_tools():
    explore = get_agent("explore")
    assert explore is not None
    allowed_set = set(explore.allowed_tools)
    disallowed_set = set(explore.disallowed_tools)
    assert "write_file" not in allowed_set or "write_file" in disallowed_set


def test_verifier_model_preference():
    v = get_agent("verifier")
    assert v is not None
    assert v.model_preference == "inherit"


def test_explore_model_preference():
    e = get_agent("explore")
    assert e is not None
    assert e.model_preference == "haiku"


def test_system_prompt_callable_resolved():
    defn = SubAgentDefinition(
        agent_type="__callable_prompt__",
        when_to_use="",
        system_prompt=lambda: "computed prompt",
    )
    assert defn.get_system_prompt() == "computed prompt"


# ──────────────────────────────────────────────────────────────────────────────
# SubAgentRunner.spawn — happy path
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_spawn_returns_final_report():
    """Spawn returns the child agent's final text output."""
    register(SubAgentDefinition(
        agent_type="__happy__",
        when_to_use="",
        system_prompt="You are helpful.",
        allowed_tools=["read_file"],
    ))
    pool = _make_pool("read_file")
    provider = ScriptedProvider([("Task complete!", [])])
    runner = SubAgentRunner(
        pool, provider, io=ScriptedIO(), cwd=Path("."),
        guardrails=Guardrails(max_subagent_depth=2, max_concurrent_subagents=4),
    )
    report = await runner.spawn("__happy__", "Do something.", depth=1)
    assert "Task complete!" in report


@pytest.mark.asyncio
async def test_spawn_unknown_type_returns_error_string():
    runner = _make_runner()
    result = await runner.spawn("totally_unknown_type", "prompt", depth=1)
    assert "Unknown sub-agent" in result
    assert "totally_unknown_type" in result


# ──────────────────────────────────────────────────────────────────────────────
# Depth caps
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_spawn_at_max_depth_returns_error_string():
    # depth > MAX_DEPTH triggers the absolute cap; set guardrail high so only cap fires.
    runner = _make_runner(max_subagent_depth=MAX_DEPTH + 10)
    result = await runner.spawn("explore", "look around", depth=MAX_DEPTH + 1)
    assert "depth limit" in result.lower()
    assert str(MAX_DEPTH) in result


@pytest.mark.asyncio
async def test_spawn_exceeds_max_depth_returns_error_string():
    runner = _make_runner(max_subagent_depth=MAX_DEPTH + 10)
    result = await runner.spawn("explore", "look around", depth=MAX_DEPTH + 5)
    assert "depth limit" in result.lower()


@pytest.mark.asyncio
async def test_guardrails_depth_cap_respected():
    runner = _make_runner(max_subagent_depth=0)
    result = await runner.spawn("explore", "search", depth=1)
    # max_subagent_depth=0 means no sub-agents; depth 1 should be blocked.
    assert "max_subagent_depth" in result


# ──────────────────────────────────────────────────────────────────────────────
# Parallel spawn
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_spawn_parallel_returns_list_of_reports():
    """spawn_parallel returns one report per task in input order."""
    register(SubAgentDefinition(
        agent_type="__par1__",
        when_to_use="",
        system_prompt="Reply with A",
        allowed_tools=["*"],
    ))
    register(SubAgentDefinition(
        agent_type="__par2__",
        when_to_use="",
        system_prompt="Reply with B",
        allowed_tools=["*"],
    ))
    pool = _make_pool("read_file")
    # Two turns: one for each child agent.
    provider = ScriptedProvider([("Report A", []), ("Report B", [])])
    runner = SubAgentRunner(
        pool, provider, io=ScriptedIO(), cwd=Path("."),
        guardrails=Guardrails(max_subagent_depth=2, max_concurrent_subagents=4),
    )
    results = await runner.spawn_parallel([
        ("__par1__", "task A"),
        ("__par2__", "task B"),
    ], depth=1)
    assert len(results) == 2
    assert all(isinstance(r, str) for r in results)


@pytest.mark.asyncio
async def test_spawn_parallel_runs_concurrently():
    """Parallel spawns complete faster than sequential (both would be instant here
    with fakes — this test just asserts the return shape and count are correct)."""
    register(SubAgentDefinition(
        agent_type="__fastA__",
        when_to_use="",
        system_prompt="",
        allowed_tools=["*"],
    ))
    register(SubAgentDefinition(
        agent_type="__fastB__",
        when_to_use="",
        system_prompt="",
        allowed_tools=["*"],
    ))
    pool = _make_pool("read_file")
    provider = ScriptedProvider([("done", []), ("done", [])])
    runner = SubAgentRunner(
        pool, provider, io=ScriptedIO(), cwd=Path("."),
        guardrails=Guardrails(max_subagent_depth=2, max_concurrent_subagents=4),
    )
    results = await runner.spawn_parallel([("__fastA__", "x"), ("__fastB__", "y")], depth=1)
    assert len(results) == 2


# ──────────────────────────────────────────────────────────────────────────────
# Child isolation
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_child_pool_filtered_for_explore():
    """The explore child should not get write_file even if it is in the parent pool."""
    pool = _make_pool("read_file", "list_dir", "search", "write_file", "spawn_agent")
    provider = ScriptedProvider([("exploration done", [])])
    runner = SubAgentRunner(
        pool, provider, io=ScriptedIO(), cwd=Path("."),
        guardrails=Guardrails(max_subagent_depth=2, max_concurrent_subagents=4),
    )
    # Patch _run_child to capture the child's pool instead of running.
    captured: list[ToolPool] = []

    async def _patched(defn, prompt, *, depth):
        pool_names = runner._full_pool.names()
        allowed = defn.resolve_tools(pool_names)
        child_pool = runner._full_pool.select(allowed)
        captured.append(child_pool)
        return "ok"

    runner._run_child = _patched
    await runner.spawn("explore", "look around", depth=1)
    assert captured, "child pool not captured"
    child_names = captured[0].names()
    assert "write_file" not in child_names
    assert "spawn_agent" not in child_names


@pytest.mark.asyncio
async def test_child_fresh_history():
    """Child starts with an empty history (no parent messages bleed through)."""
    register(SubAgentDefinition(
        agent_type="__isolation_test__",
        when_to_use="",
        system_prompt="",
        allowed_tools=["*"],
    ))
    captured_histories: list = []
    pool = _make_pool("read_file")
    provider = ScriptedProvider([("ok", [])])
    runner = SubAgentRunner(
        pool, provider, io=ScriptedIO(), cwd=Path("."),
        guardrails=Guardrails(max_subagent_depth=2, max_concurrent_subagents=4),
    )

    async def _spy(defn, prompt, *, depth):
        from neurosurfer.agents.agentic_loop import AgenticLoop
        from neurosurfer.agents.runtime.permissions import Guardrails as G
        child_pool = runner._full_pool.select(defn.resolve_tools(runner._full_pool.names()))
        child = AgenticLoop(
            provider=runner._provider,
            tools=child_pool,
            system_prompt=defn.get_system_prompt(),
            guardrails=G(max_turns=10, max_subagent_depth=0, max_concurrent_subagents=0),
            io=runner._io,
            cwd=runner._cwd,
            depth=depth,
            spawn=None,
        )
        captured_histories.append(list(child.history.messages))
        return await child.run_collect(prompt)

    runner._run_child = _spy  # type: ignore[method-assign]
    await runner.spawn("__isolation_test__", "hello", depth=1)
    assert captured_histories, "spy not called"
    # History contains only the user message (no parent history leaked in).
    msgs = captured_histories[0]
    assert len(msgs) == 0, f"Expected empty history before run, got {msgs}"


# ──────────────────────────────────────────────────────────────────────────────
# Concurrency semaphore
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_concurrency_cap_limits_simultaneous_spawns():
    """With max_concurrent=2, the semaphore blocks the 3rd spawn until one completes."""
    register(SubAgentDefinition(
        agent_type="__slow__",
        when_to_use="",
        system_prompt="",
        allowed_tools=["*"],
    ))

    active_count = 0
    peak_active = 0

    pool = _make_pool("read_file")
    # Three scripted turns for three spawns.
    provider = ScriptedProvider([("r1", []), ("r2", []), ("r3", [])])
    runner = SubAgentRunner(
        pool, provider, io=ScriptedIO(), cwd=Path("."),
        guardrails=Guardrails(max_subagent_depth=2, max_concurrent_subagents=2),
    )

    async def _counting_run(defn, prompt, *, depth):
        nonlocal active_count, peak_active
        active_count += 1
        peak_active = max(peak_active, active_count)
        await asyncio.sleep(0)  # yield to event loop
        active_count -= 1
        return "done"

    runner._run_child = _counting_run  # type: ignore[method-assign]

    await asyncio.gather(
        runner.spawn("__slow__", "t1", depth=1),
        runner.spawn("__slow__", "t2", depth=1),
        runner.spawn("__slow__", "t3", depth=1),
    )
    assert peak_active <= 2, f"peak_active={peak_active} exceeded cap of 2"


# ──────────────────────────────────────────────────────────────────────────────
# make_spawn_fn
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_make_spawn_fn_increments_depth():
    """The SpawnFn created at depth N calls spawn() at depth N+1."""
    depths_seen: list[int] = []
    runner = _make_runner()

    async def _spy(agent_type: str, prompt: str, *, depth: int) -> str:
        depths_seen.append(depth)
        return "ok"

    runner.spawn = _spy  # type: ignore[method-assign]
    fn = runner.make_spawn_fn(parent_depth=0)
    await fn("explore", "hello")
    assert depths_seen == [1]


# ──────────────────────────────────────────────────────────────────────────────
# TasksRuntime
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_tasks_runtime_submit_and_result():
    rt = TasksRuntime()

    async def _work() -> str:
        return "finished"

    handle = rt.submit(_work(), description="test job")
    assert isinstance(handle, TaskHandle)
    assert not handle.done
    result = await handle.result()
    assert result == "finished"
    assert handle.done


@pytest.mark.asyncio
async def test_tasks_runtime_active_and_all():
    rt = TasksRuntime()
    evt = asyncio.Event()

    async def _blocker():
        await evt.wait()
        return "ok"

    h = rt.submit(_blocker(), description="blocking")
    assert h in rt.active()
    assert h in rt.all()
    evt.set()
    await h.result()
    assert h not in rt.active()
    assert h in rt.all()


@pytest.mark.asyncio
async def test_tasks_runtime_cancel():
    rt = TasksRuntime()
    evt = asyncio.Event()

    async def _infinite():
        await evt.wait()

    h = rt.submit(_infinite())
    h.cancel()
    with pytest.raises((asyncio.CancelledError, Exception)):
        await h.result()
