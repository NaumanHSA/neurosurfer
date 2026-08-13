"""A `react` node can return a shaped answer, and the field is honoured.

`kinds/react.py` withdrew `output_schema` because it was inert — offered by the
panel, written into the YAML, reviewed, and read by nothing, so "return an
object" silently returned prose. The note said the field would come back when
the loop honoured a schema. These are that.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from neurosurfer.graph.engine.kinds import node_kind_spec
from neurosurfer.graph.engine.node_runner import run_react_node
from neurosurfer.llm.types import CanonicalResponse, TextBlock, ToolUseBlock, Usage
from neurosurfer.tools.base import AutoApproveIOHandler, ToolContext, ToolPool


class Summary(BaseModel):
    headline: str
    confidence: int


class _ScriptedProvider:
    """Replays canonical turns, and records what each call was asked."""

    model = "scripted"

    def __init__(self, turns):
        from neurosurfer.llm.capabilities import ProviderCapabilities

        self._turns = list(turns)
        self.prompts: list[str] = []
        self.capabilities = ProviderCapabilities(
            supports_thinking=False, supports_prompt_cache=False,
            supports_token_count=False, tool_call_style="anthropic",
            context_window=8192, max_output_tokens=2048,
        )

    async def stream(self, messages, system, tools, config):
        from neurosurfer.llm.types import Done, TextDelta, ToolUseArgsDelta, ToolUseStart

        self.prompts.append(
            "\n".join(
                b.text
                for m in (messages or [])
                for b in getattr(m, "content", []) or []
                if getattr(b, "text", None)
            )
        )
        content = self._turns.pop(0) if self._turns else [TextBlock(text="")]
        for i, block in enumerate(content):
            if isinstance(block, TextBlock):
                yield TextDelta(text=block.text)
            elif isinstance(block, ToolUseBlock):
                import json

                yield ToolUseStart(index=i, id=block.id, name=block.name)
                yield ToolUseArgsDelta(index=i, partial_json=json.dumps(block.input))
        yield Done(
            response=CanonicalResponse(
                content=content,
                stop_reason="tool_use" if any(isinstance(b, ToolUseBlock) for b in content) else "end_turn",
                usage=Usage(input_tokens=10, output_tokens=5),
                model=self.model,
            )
        )

    async def complete(self, messages, system, tools, config):
        from neurosurfer.llm.types import Done

        async for ev in self.stream(messages, system, tools, config):
            if isinstance(ev, Done):
                return ev.response
        return None

    async def count_tokens(self, messages, system=None, tools=None):
        return 0


def _finish_pool():
    from neurosurfer.registry.core.agent.finish import FinishTool

    return ToolPool([FinishTool()])


def _run(turns, schema=None):
    ctx = ToolContext(cwd=Path("."), io=AutoApproveIOHandler())
    provider = _ScriptedProvider(turns)
    call = run_react_node(
        provider, _finish_pool(), ctx, "sys", "summarise the findings",
        output_schema=schema,
    )
    return call, provider


class TestTheSpecOffersItAgain:
    def test_react_declares_output_schema(self):
        assert node_kind_spec("react").field("output_schema") is not None

    def test_validation_now_covers_it(self):
        """While the field was inert the rule deliberately skipped `react` —
        validating a field the engine ignores endorses it."""
        from neurosurfer.graph.workflow.validation.registry import rules_for_kind

        assert "output_schema_resolves" in {r.name for r in rules_for_kind("react")}


class TestItIsHonoured:
    def test_the_loop_runs_then_the_answer_is_shaped(self):
        submit = ToolUseBlock(
            id="s1", name="submit_result",
            input={"headline": "Revenue rose", "confidence": 4},
        )
        call, provider = _run(
            [
                [TextBlock(text="Revenue rose 12% year on year.")],  # the loop
                [submit],                                            # the shaping
            ],
            schema=Summary,
        )

        assert isinstance(call.output, Summary)
        assert call.output.headline == "Revenue rose"
        assert call.output.confidence == 4

    def test_the_shaping_call_is_given_the_answer_not_the_task(self):
        """Re-running the original task without the tools would ask the model to
        invent what the loop just spent its turns finding out."""
        submit = ToolUseBlock(
            id="s1", name="submit_result", input={"headline": "x", "confidence": 1}
        )
        _, provider = _run(
            [[TextBlock(text="THE LOOP FOUND THIS")], [submit]], schema=Summary
        )

        assert "THE LOOP FOUND THIS" in provider.prompts[-1]

    def test_shaping_tokens_are_billed_to_the_node(self):
        """Otherwise a structured react node under-reports by a whole call."""
        submit = ToolUseBlock(
            id="s1", name="submit_result", input={"headline": "x", "confidence": 1}
        )
        with_schema, _ = _run(
            [[TextBlock(text="answer")], [submit]], schema=Summary
        )
        without, _ = _run([[TextBlock(text="answer")]])

        assert with_schema.usage.input_tokens > without.usage.input_tokens

    def test_without_a_schema_the_answer_is_still_prose(self):
        call, provider = _run([[TextBlock(text="just prose")]])

        assert call.output == "just prose"
        assert len(provider.prompts) == 1, "no shaping call should have been made"

    def test_an_empty_loop_still_fails_before_shaping(self):
        """Shaping nothing would produce a confidently invented object."""
        from neurosurfer.graph.engine.errors import GraphConfigurationError

        with pytest.raises(GraphConfigurationError, match="without producing an answer"):
            _run([[TextBlock(text="")]], schema=Summary)
