from __future__ import annotations

import pytest

from neurosurfer.agents import events
from neurosurfer.agents.react_agent import (
    ReactAgent,
    _build_react_system,
    _parse_action_input,
    _parse_react_output,
)
from neurosurfer.agents.runtime.permissions import Guardrails
from neurosurfer.tools import default_pool

from .fakes import ScriptedIO, ScriptedProvider


def build_react(provider, cwd, *, max_turns=8):
    return ReactAgent(
        provider=provider,
        tools=default_pool(),
        system_prompt="Answer the user's question.",
        guardrails=Guardrails(write_scope=["**"], max_turns=max_turns),
        io=ScriptedIO(),
        cwd=cwd,
    )


async def drive(agent, user_input):
    return [ev async for ev in agent.run(user_input)]


# ── parser unit tests ───────────────────────────────────────────────────────

def test_parse_final_answer_wins_and_is_clean():
    final, name, raw = _parse_react_output(
        "Thought: done\nFinal Answer: The answer is 42."
    )
    assert final == "The answer is 42."
    assert name is None and raw is None


def test_parse_action_and_input():
    final, name, raw = _parse_react_output(
        'Thought: look it up\nAction: read_file\nAction Input: {"path": "a.txt"}'
    )
    assert final is None
    assert name == "read_file"
    obj, err = _parse_action_input(raw)
    assert err is None and obj == {"path": "a.txt"}


def test_parse_action_name_strips_brackets_and_quotes():
    _, name, _ = _parse_react_output('Action: ["read_file"]\nAction Input: {}')
    assert name == "read_file"


def test_parse_action_input_strips_code_fence_and_recovers_brace():
    obj, err = _parse_action_input('```json\n{"path": "a.txt"}\n```')
    assert err is None and obj == {"path": "a.txt"}
    obj2, err2 = _parse_action_input('here you go: {"x": 1} trailing junk')
    assert err2 is None and obj2 == {"x": 1}


def test_parse_action_input_reports_unparseable():
    obj, err = _parse_action_input("path = a.txt")
    assert obj is None and err is not None


def test_empty_action_input_is_empty_object():
    obj, err = _parse_action_input("")
    assert err is None and obj == {}


def test_build_react_system_lists_tools_and_format():
    sys = _build_react_system("BASE", default_pool())
    assert "BASE" in sys
    assert "Final Answer:" in sys
    assert "Action Input:" in sys
    assert "read_file" in sys  # a known builtin appears in the catalog


# ── loop integration tests ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_action_then_observation_then_final(tmp_path):
    (tmp_path / "a.txt").write_text("hello source\n")
    turns = [
        ('Thought: read it\nAction: read_file\nAction Input: {"path": "a.txt"}', []),
        ("Thought: I now know the final answer\nFinal Answer: The file says hello source.", []),
    ]
    agent = build_react(ScriptedProvider(turns), tmp_path)
    evs = await drive(agent, "what does a.txt say?")

    tool_done = [e for e in evs if isinstance(e, events.ToolFinished)]
    assert [e.name for e in tool_done] == ["read_file"]
    finished = [e for e in evs if isinstance(e, events.RunFinished)]
    assert finished and finished[-1].status == "completed"
    # No sentinel leakage: the emitted text is the clean answer.
    text = "".join(e.text for e in evs if isinstance(e, events.TextDelta))
    assert "Final Answer:" not in text
    assert "The file says hello source." in text


@pytest.mark.asyncio
async def test_bad_action_input_recovers_via_observation(tmp_path):
    turns = [
        ("Thought: act\nAction: read_file\nAction Input: path = a.txt", []),  # invalid JSON
        ("Thought: done\nFinal Answer: recovered", []),
    ]
    agent = build_react(ScriptedProvider(turns), tmp_path)
    evs = await drive(agent, "go")
    # No tool ran (input never parsed), but the run still finishes cleanly.
    assert not [e for e in evs if isinstance(e, events.ToolStarted)]
    finished = [e for e in evs if isinstance(e, events.RunFinished)]
    assert finished[-1].status == "completed"
    assert "recovered" in finished[-1].report


@pytest.mark.asyncio
async def test_plain_prose_is_treated_as_final(tmp_path):
    turns = [("The capital of France is Paris.", [])]
    agent = build_react(ScriptedProvider(turns), tmp_path)
    evs = await drive(agent, "capital of France?")
    finished = [e for e in evs if isinstance(e, events.RunFinished)]
    assert finished[-1].status == "completed"
    assert "Paris" in finished[-1].report


@pytest.mark.asyncio
async def test_terminates_on_max_turns_with_partial(tmp_path):
    (tmp_path / "a.txt").write_text("x\n")
    # Always acts, never gives a Final Answer.
    act = ('Thought: again\nAction: read_file\nAction Input: {"path": "a.txt"}', [])
    agent = build_react(ScriptedProvider([act, act, act]), tmp_path, max_turns=2)
    evs = await drive(agent, "loop forever")
    finished = [e for e in evs if isinstance(e, events.RunFinished)]
    assert finished[-1].status == "max_turns"
    assert finished[-1].report  # never empty
    assert agent.turns == 2
