from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from neurosurfer.agents import events
from neurosurfer.agents.oneshot import Agent
from neurosurfer.agents.runtime.permissions import Guardrails
from neurosurfer.tools import default_pool

from .fakes import ScriptedIO, ScriptedProvider


class Plan(BaseModel):
    name: str = Field(description="snake_case name")
    steps: list[str] = Field(min_length=1)


def build_oneshot(provider, cwd, *, output_schema=None, max_tool_rounds=1):
    return Agent(
        provider=provider,
        tools=default_pool(),
        system_prompt="Answer.",
        guardrails=Guardrails(write_scope=["**"]),
        io=ScriptedIO(),
        cwd=cwd,
        output_schema=output_schema,
        max_tool_rounds=max_tool_rounds,
    )


@pytest.mark.asyncio
async def test_text_mode_returns_answer(tmp_path):
    agent = build_oneshot(ScriptedProvider([("Paris.", [])]), tmp_path)
    result = await agent.complete("capital of France?")
    assert result == "Paris."
    assert agent.result == "Paris."


@pytest.mark.asyncio
async def test_structured_mode_returns_model(tmp_path):
    turns = [("", [("submit_result", {"name": "docs", "steps": ["a", "b"]})])]
    agent = build_oneshot(ScriptedProvider(turns), tmp_path, output_schema=Plan)
    result = await agent.complete("make a plan")
    assert isinstance(result, Plan)
    assert result.name == "docs" and result.steps == ["a", "b"]


@pytest.mark.asyncio
async def test_bounded_one_tool_round_then_synthesis(tmp_path):
    (tmp_path / "a.txt").write_text("hello\n")
    turns = [
        ("", [("read_file", {"path": "a.txt"})]),   # round 1: model calls a tool
        ("The file says hello.", []),                # synthesis call
    ]
    provider = ScriptedProvider(turns)
    agent = build_oneshot(provider, tmp_path, max_tool_rounds=1)
    evs = [ev async for ev in agent.run("read a.txt")]

    tool_done = [e for e in evs if isinstance(e, events.ToolFinished)]
    assert [e.name for e in tool_done] == ["read_file"]
    assert agent.result == "The file says hello."
    # Exactly two model calls: initial + one synthesis.
    assert provider.calls == 2
    finished = [e for e in evs if isinstance(e, events.RunFinished)]
    assert finished[-1].status == "completed"


@pytest.mark.asyncio
async def test_tool_rounds_are_bounded(tmp_path):
    (tmp_path / "a.txt").write_text("x\n")
    # Model keeps asking for tools; max_tool_rounds=1 must stop it after synthesis.
    call = ("", [("read_file", {"path": "a.txt"})])
    provider = ScriptedProvider([call, call, call])
    agent = build_oneshot(provider, tmp_path, max_tool_rounds=1)
    await agent.complete("go")
    # call 1 (tools) -> execute -> call 2 (still tools, but rounds exhausted -> stop)
    assert provider.calls == 2
