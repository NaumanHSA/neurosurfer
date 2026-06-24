"""R4 foundation: structured output on the native stack via native tool-use."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from neurosurfer.agents.runtime.structured import (
    StructuredCompletionError,
    structured_completion,
)
from neurosurfer.llm.types import (
    CanonicalResponse,
    ToolUseBlock,
    Usage,
)


class Plan(BaseModel):
    name: str = Field(description="snake_case name")
    steps: list[str] = Field(min_length=1)


def _resp_tool(tool_input: dict, name: str = "submit_result") -> CanonicalResponse:
    return CanonicalResponse(
        content=[ToolUseBlock(id="t1", name=name, input=tool_input)],
        stop_reason="tool_use",
        usage=Usage(),
    )


def _resp_text(text: str) -> CanonicalResponse:
    from neurosurfer.llm.types import TextBlock

    return CanonicalResponse(
        content=[TextBlock(text=text)], stop_reason="end_turn", usage=Usage()
    )


class _FakeProvider:
    """Returns a scripted list of responses in order (last repeats)."""

    def __init__(self, responses):
        self._responses = responses
        self.calls = 0

    async def complete(self, messages, system, tools, config):  # noqa: ANN001
        r = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return r


@pytest.mark.asyncio
async def test_returns_validated_model_on_first_call():
    provider = _FakeProvider([_resp_tool({"name": "docs", "steps": ["a", "b"]})])
    result = await structured_completion(provider, Plan, user="make a plan")
    assert isinstance(result, Plan)
    assert result.name == "docs"
    assert result.steps == ["a", "b"]
    assert provider.calls == 1


@pytest.mark.asyncio
async def test_retries_when_model_skips_the_tool():
    provider = _FakeProvider([
        _resp_text("here is your plan in prose"),  # no tool call
        _resp_tool({"name": "docs", "steps": ["x"]}),
    ])
    result = await structured_completion(provider, Plan, user="go")
    assert result.steps == ["x"]
    assert provider.calls == 2


@pytest.mark.asyncio
async def test_repairs_invalid_then_succeeds():
    provider = _FakeProvider([
        _resp_tool({"name": "docs", "steps": []}),  # min_length=1 violated
        _resp_tool({"name": "docs", "steps": ["ok"]}),
    ])
    result = await structured_completion(provider, Plan, user="go")
    assert result.steps == ["ok"]
    assert provider.calls == 2


@pytest.mark.asyncio
async def test_raises_after_exhausting_attempts():
    provider = _FakeProvider([_resp_tool({"name": "docs"})])  # missing required 'steps'
    with pytest.raises(StructuredCompletionError, match="Plan"):
        await structured_completion(provider, Plan, user="go", max_attempts=2)
    assert provider.calls == 2


@pytest.mark.asyncio
async def test_accepts_misnamed_tool_call():
    provider = _FakeProvider([_resp_tool({"name": "docs", "steps": ["a"]}, name="wrong_name")])
    result = await structured_completion(provider, Plan, user="go")
    assert result.name == "docs"
