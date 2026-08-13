"""The three defects plan 01 carried across, pinned so they cannot come back.

All three predate the port and survived it — the studio branch had them too. Each
was reproduced on *this* engine before it was fixed, because the original repros
were written against the studio branch's engine and Phase 1 replaced it.

What they have in common is the reason they lasted: **every one of them produced a
plausible-looking success.** A prompt that reads fine, a green run, a blank answer.
Nothing raised, nothing logged, nothing red.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from neurosurfer.graph.engine.errors import GraphConfigurationError
from neurosurfer.graph.engine.manager import ManagerAgent
from neurosurfer.graph.engine.node_runner import run_base_node
from neurosurfer.graph.engine.schema import GraphNode
from neurosurfer.llm.types import CanonicalResponse, TextBlock, ToolUseBlock, Usage
from neurosurfer.tools.base import Tool, ToolPool, ToolResult

# ── the prompt assembler ─────────────────────────────────────────────────────


def _prompt(node, task, dependency_results):
    return ManagerAgent().compose_user_prompt(
        node=node,
        task=task,
        dependency_results=dependency_results,
    )


def test_a_workflow_that_is_not_the_architect_is_not_told_its_request_is_unspecified():
    """`user_intent` was the Architect's own graph's key, and was the only one read.

    So every hand-built workflow opened with "User request: (not specified)" —
    on a run that then succeeded and answered plausibly, which is why it lasted.

    It cannot recur, and not because the header was made conditional: there is
    **no header**, and no ambient block for one to sit above. A node is given
    its task and the steps it declared, so there is nothing left that could
    describe a request the graph never had.
    """
    node = GraphNode(id="a", kind="base", instructions="Do it.", depends_on=["intake"])
    out = _prompt(node, "Your task:\nDo it.", {"intake": "summarise the report"})

    assert "(not specified)" not in out
    assert "User request:" not in out
    assert "summarise the report" in out


def test_an_input_node_value_is_shown_once_not_three_times():
    """`_run_input_node` writes what it collected under its own id, so the same
    string arrived as a graph input *and* as a dependency result — under two
    headings that disagreed about what it was, plus a third time in the header.

    One door now. It is a dependency result, because that is what an upstream
    node's output is, and the ambient inputs block that was the other two doors
    is gone.
    """
    msg = "visit https://example.com and write your findings"
    node = GraphNode(id="a", kind="base", instructions="Do it.", depends_on=["intake"])

    out = _prompt(node, "Your task:\nDo it.", {"intake": msg})

    assert out.count(msg) == 1, f"value repeated:\n{out}"
    assert "Context from previous nodes:" in out


def test_the_architect_is_not_recited_its_own_plumbing():
    """`user_intent` and `available_tools` are the Architect graph's inputs, and
    every one of its nodes interpolates `{user_intent}` itself.

    Neither is recited here — nothing is. The check is kept because these two
    names are the ones that used to be special-cased, and a special case is
    exactly the sort of thing that grows back.
    """
    node = GraphNode(id="plan", kind="base", instructions="Plan it.")
    out = _prompt(node, "Your task:\nPlan {user_intent}.", {})

    assert "<catalog>" not in out
    assert "available_tools" not in out


def test_a_genuine_upstream_result_is_still_shown():
    """The narrowing must not reach the dependency block. What a node declared
    as a dependency is the one thing it is unambiguously entitled to."""
    node = GraphNode(id="w", kind="base", instructions="Write it.", depends_on=["summarise"])
    out = _prompt(node, "Your task:\nWrite it.", {"summarise": "Otters are mustelids."})

    assert "Otters are mustelids." in out


def test_a_node_is_not_shown_a_dependency_it_never_declared():
    """`depends_on` is the whole of what a node inherits. A sibling's output
    reaching it anyway would be the ambient-context problem in a second place."""
    node = GraphNode(id="w", kind="base", instructions="Write it.", depends_on=["summarise"])
    out = _prompt(node, "Your task:\nWrite it.",
                  {"summarise": "Otters are mustelids.", "unrelated": "Badgers are not."})

    assert "Otters are mustelids." in out
    assert "Badgers are not." not in out


# ── the one-round truncation ─────────────────────────────────────────────────


class _Args(BaseModel):
    x: str = "1"


class _Echo(Tool):
    name = "echo"
    description = "echo"
    input_model = _Args

    async def call(self, args, ctx):  # type: ignore[override]
        return ToolResult.ok("echoed")


class _Caps:
    max_output_tokens = 100
    tool_call_style = "openai"
    supports_tools = True


def _provider(turns: list[list]):
    """A provider that replays `turns`, repeating the last one forever."""

    class _P:
        capabilities = _Caps()

        def __init__(self) -> None:
            self.i = 0

        async def complete(self, messages, system, tools, config):
            blocks = turns[min(self.i, len(turns) - 1)]
            self.i += 1
            return CanonicalResponse(
                content=blocks,
                stop_reason="tool_use",
                usage=Usage(input_tokens=10, output_tokens=5),
            )

    return _P()


def _wants_tool(i: int) -> list:
    return [ToolUseBlock(id=f"t{i}", name="echo", input={"x": "1"})]


def test_a_step_cut_off_mid_plan_with_nothing_to_show_is_an_error():
    """The failure that started this: "fetch the page then write it to a file".

    A `base` step gets one round. The model spends it on the fetch, is refused
    the second, and its last turn is tool calls with no prose — so the node
    returned `""` and the run reported success. Three green nodes, blank answer.
    """
    with pytest.raises(GraphConfigurationError) as e:
        run_base_node(
            _provider([_wants_tool(1), _wants_tool(2)]),
            "s",
            "fetch the page then write it to a file",
            tool_pool=ToolPool([_Echo()]),
        )

    message = str(e.value)
    assert "echo" in message, "it must name what it did call"
    assert "react" in message, "and the way out"


def test_a_partial_answer_is_kept_rather_than_thrown_away():
    """Empty is the line, not cut-short.

    A truncated step that still produced text produced a partial answer, and a
    partial answer is sometimes what was wanted. Failing it would break runs that
    are useful today; what is never useful is nothing at all.
    """
    call = run_base_node(
        _provider([_wants_tool(1), [TextBlock(text="Partial but useful."), *_wants_tool(2)]]),
        "s",
        "fetch then write",
        tool_pool=ToolPool([_Echo()]),
    )
    assert call.output == "Partial but useful."


def test_an_ordinary_answer_is_untouched():
    """The guard must not fire on a step that simply finished."""
    call = run_base_node(_provider([[TextBlock(text="All done.")]]), "s", "just answer")
    assert call.output == "All done."


def test_a_step_that_never_wanted_tools_is_not_called_cut_short():
    """`cut_short` keys on pending tool calls, not on an empty string — a model
    that returns nothing for its own reasons is a different problem."""
    call = run_base_node(_provider([[TextBlock(text="")]]), "s", "answer")
    assert call.output == ""


# ── a react node's answer can arrive by either channel ───────────────────────


def _finish_pool():
    from neurosurfer.registry.core.agent.finish import FinishTool

    return ToolPool([FinishTool()])


def _react(turns):
    from pathlib import Path

    from neurosurfer.graph.engine.node_runner import run_react_node
    from neurosurfer.tools.base import AutoApproveIOHandler, ToolContext
    from tests.fakes import ScriptedProvider

    ctx = ToolContext(cwd=Path("."), io=AutoApproveIOHandler())
    return run_react_node(ScriptedProvider(turns=turns), _finish_pool(), ctx, "sys", "go")


def test_an_answer_given_to_finish_is_the_nodes_output():
    """Found in tutorial 03, against a real model.

    `RunResult` has two channels — `final_text` accumulates text deltas, `report`
    carries what `RunFinished` was given. A model that does its work and then
    calls `finish(summary=…)` emits no text deltas at all, so reading
    `final_text` alone returned `""` from a node that had just written the answer
    down.

    Nothing failed: the node was green, its tool calls were in the trace, and the
    *next* node was handed an empty string and improvised — "I don't have the
    scout findings above to summarise." One node's silence became the next node's
    invention.
    """
    call = _react([("", [("finish", {"summary": "Neurosurfer is an agent framework.",
                                     "status": "success"})])])
    assert call.output == "Neurosurfer is an agent framework."


def test_a_prose_answer_still_wins_when_there_is_no_report():
    call = _react([("A plain prose answer.", [])])
    assert call.output == "A plain prose answer."


def test_a_react_node_that_produced_nothing_at_all_is_an_error():
    """Empty is the line here too — same rule as `run_base_node`."""
    with pytest.raises(GraphConfigurationError) as e:
        _react([("", [("finish", {"summary": "", "status": "success"})])])
    assert "finish" in str(e.value)


class _ThinksOutLoud:
    """A provider whose turn is reasoning and nothing else.

    What a local reasoning model does intermittently: no tool call, no text, the
    whole turn in `reasoning_content`.
    """

    model = "thinker"

    def __init__(self, thinking: str) -> None:
        from neurosurfer.llm.capabilities import ProviderCapabilities

        self._thinking = thinking
        self.capabilities = ProviderCapabilities(
            supports_thinking=True, supports_prompt_cache=False,
            supports_token_count=False, tool_call_style="anthropic",
            context_window=8192, max_output_tokens=2048,
        )

    async def stream(self, messages, system, tools, config):  # noqa: ANN001
        from neurosurfer.llm.types import (
            CanonicalResponse,
            Done,
            ThinkingBlock,
            ThinkingDelta,
            Usage,
        )

        yield ThinkingDelta(text=self._thinking)
        yield Done(response=CanonicalResponse(
            content=[ThinkingBlock(thinking=self._thinking)],
            stop_reason="end_turn",
            usage=Usage(input_tokens=1, output_tokens=1),
            model=self.model,
        ))

    async def complete(self, messages, system, tools, config):  # noqa: ANN001
        from neurosurfer.llm.types import Done

        async for ev in self.stream(messages, system, tools, config):
            if isinstance(ev, Done):
                return ev.response
        return None

    async def count_tokens(self, messages, system, tools):  # noqa: ANN001
        return 0


def test_a_thinking_only_turn_is_an_answer_not_a_silence():
    """`final_text` collects `TextDelta` only, so a turn that is all reasoning
    left it empty and the node failed — taking every node downstream with it.

    `CanonicalResponse.text()` already falls back to thinking for the one-shot
    path (*"thinking-only models put everything in reasoning_content"*); the
    streamed path disagreed purely because it accumulates deltas. Measured on
    `qwen/qwen3.5-9b` driving the capstone tutorial's vision node: two runs in six.
    """
    from pathlib import Path

    from neurosurfer.graph.engine.node_runner import run_react_node
    from neurosurfer.tools.base import AutoApproveIOHandler, ToolContext

    ctx = ToolContext(cwd=Path("."), io=AutoApproveIOHandler())
    call = run_react_node(
        _ThinksOutLoud("The chart trends up, then spikes in November."),
        _finish_pool(), ctx, "sys", "go",
    )
    assert "spikes in November" in call.output


def test_a_real_answer_still_beats_the_reasoning_that_preceded_it():
    """The fallback is last in the order — reasoning must never displace prose."""
    call = _react([("The considered answer.", [])])
    assert call.output == "The considered answer."
