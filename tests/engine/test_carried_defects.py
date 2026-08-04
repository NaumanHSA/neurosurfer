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


def _prompt(node, graph_inputs, dependency_results):
    return ManagerAgent().compose_user_prompt(
        node=node,
        graph_inputs=graph_inputs,
        dependency_results=dependency_results,
        previous_result=None,
    )


def test_a_workflow_that_is_not_the_architect_is_not_told_its_request_is_unspecified():
    """`user_intent` is the Architect's own graph's key, and was the only one read.

    So every hand-built workflow opened with "User request: (not specified)" —
    on a run that then succeeded and answered plausibly, which is why it lasted.
    """
    node = GraphNode(id="a", kind="base", instructions="Do it.", depends_on=["intake"])
    out = _prompt(node, {"intake": "summarise the report"}, {"intake": "summarise the report"})

    assert "(not specified)" not in out
    assert "User request:" not in out
    assert "summarise the report" in out


def test_an_input_node_value_is_shown_once_not_three_times():
    """`_run_input_node` writes what it collected under its own id, so the same
    string arrives as a graph input *and* as a dependency result."""
    msg = "visit https://example.com and write your findings"
    node = GraphNode(id="a", kind="base", instructions="Do it.", depends_on=["intake"])

    out = _prompt(node, {"intake": msg}, {"intake": msg})

    assert out.count(msg) == 1, f"value repeated:\n{out}"
    assert "Context from previous nodes:" not in out


def test_the_architect_still_gets_its_header_and_hides_its_plumbing():
    node = GraphNode(id="plan", kind="base", instructions="Plan it.")
    out = _prompt(node, {"user_intent": "build a PR digest", "available_tools": "<catalog>"}, {})

    assert out.startswith("User request: build a PR digest")
    # `available_tools` is interpolated into the system prompt already.
    assert "<catalog>" not in out


def test_a_genuine_upstream_result_is_still_shown():
    """The de-duplication must key on the *value*, not on "it is a dependency" —
    otherwise the fix for the echo would swallow real upstream context."""
    node = GraphNode(id="w", kind="base", instructions="Write it.", depends_on=["summarise"])
    out = _prompt(node, {"topic": "otters"}, {"summarise": "Otters are mustelids."})

    assert "Otters are mustelids." in out
    assert "topic: otters" in out


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
