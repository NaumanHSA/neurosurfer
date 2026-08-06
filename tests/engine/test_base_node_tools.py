"""A base node uses the tools attached to it — one LLM call, one round of tools.

This is the rung between `base` and `react`: the model may call its tools once, sees
the results and answers, rather than looping. Before this, a base node was handed an
empty pool unconditionally, so a tool attached to one was validated, drawn on the
canvas, and silently dropped at run time.
"""

from __future__ import annotations

from pydantic import BaseModel

from neurosurfer.graph.engine.executor import GraphExecutor
from neurosurfer.graph.engine.loader import load_graph_from_dict
from neurosurfer.llm.capabilities import ProviderCapabilities
from neurosurfer.llm.types import CanonicalResponse, TextBlock, ToolUseBlock, Usage
from neurosurfer.tools.base import Tool, ToolPool, ToolResult

USAGE = Usage(input_tokens=4, output_tokens=2)


class _EchoArgs(BaseModel):
    text: str = ""


class _Echo(Tool):
    name = "echo"
    description = "Echo the text back."
    input_model = _EchoArgs

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict] = []

    async def call(self, args, ctx):  # noqa: ANN001
        self.calls.append({"text": args.text})
        return ToolResult.ok(f"echoed:{args.text}")


class _ToolThenText:
    """Asks for one tool call on the first turn, then answers."""

    model = "fake"
    capabilities = ProviderCapabilities(
        context_window=8192, max_output_tokens=2048,
        supports_thinking=False, supports_prompt_cache=False,
        supports_token_count=False, tool_call_style="openai",
    )

    def __init__(self) -> None:
        self.turns = 0
        self.tools_offered: list[int] = []

    async def complete(self, messages, system, tools, config):  # noqa: ANN001
        offered = len(tools or [])
        self.tools_offered.append(offered)
        self.turns += 1
        # Only reach for a tool that was actually offered — a model cannot call
        # what it was never shown, and pretending otherwise would test the fake.
        if self.turns == 1 and offered:
            return CanonicalResponse(
                content=[ToolUseBlock(id="t1", name="echo", input={"text": "hi"})],
                stop_reason="tool_use", usage=USAGE,
            )
        return CanonicalResponse(
            content=[TextBlock(text="done")], stop_reason="end_turn", usage=USAGE,
        )


def _run(nodes, provider, tools=None, validate=True):
    """Run a graph. `validate=False` is for the tests that deliberately build a
    graph the validator refuses, to assert what the *runtime* does with it."""
    graph = load_graph_from_dict({"name": "t", "nodes": nodes, "outputs": [nodes[-1]["id"]]})
    return GraphExecutor(
        graph, provider=provider, log_traces=False, validate=validate,
        **({"native_tools": ToolPool(tools)} if tools else {}),
    ).run({})


def test_a_base_node_calls_the_tool_attached_to_it():
    echo = _Echo()
    provider = _ToolThenText()
    res = _run(
        [{"id": "step", "kind": "base", "goal": "echo hi", "tools": ["echo"]}],
        provider, [echo],
    )
    assert echo.calls == [{"text": "hi"}], "the attached tool was never called"
    assert res.nodes["step"].raw_output == "done"
    # Offered on the first turn — a pool the model is not shown cannot be called.
    assert provider.tools_offered[0] == 1


def test_a_base_node_with_no_tools_is_unchanged():
    """One call, nothing offered. The change must not alter existing graphs."""
    provider = _ToolThenText()
    _run([{"id": "step", "kind": "base", "goal": "just answer"}], provider)
    assert provider.tools_offered == [0]
    assert provider.turns == 1


def test_the_tool_calls_are_recorded_on_the_node_result():
    """Otherwise a trace of a base node says nothing about what it did."""
    echo = _Echo()
    res = _run(
        [{"id": "step", "kind": "base", "goal": "echo", "tools": ["echo"]}],
        _ToolThenText(), [echo],
    )
    assert "echo" in (res.nodes["step"].tool_calls or [])


class TestBoundArgumentsReachAnAgentNode:
    """`tool_args` and `tool_settings` on a node that calls a model.

    These two lines are the only route by which an agent node can be handed a
    credential — a secret reaches a tool as `${NAME}` in `tool_args`, never
    through a prompt — and until the executor became a package they were methods
    on `GraphExecutor`. The split moved them to module-level functions taking the
    executor, and the two call sites in `llm.py` kept calling them as methods.

    Nothing offline caught it: every existing test here attaches `tools` without
    `tool_args`, so the branch never ran. It surfaced as `AttributeError:
    'GraphExecutor' object has no attribute '_render_tool_args'` on a live
    Architect build — a failure path, one directory away from any import-time
    check, which is the same shape as the relative-import breakage that cost the
    first attempt at this split.
    """

    def test_a_bound_argument_is_substituted_and_hidden(self):
        echo = _Echo()
        res = _run(
            [{
                "id": "step", "kind": "base", "goal": "echo", "tools": ["echo"],
                "tool_args": {"text": "bound"},
            }],
            _ToolThenText(), [echo],
        )
        assert res.nodes["step"].error is None, res.nodes["step"].error
        # The model asked for text="hi"; the bound value wins and it never had
        # the chance to choose — that is what makes this safe for a credential.
        assert echo.calls == [{"text": "bound"}]

    def test_a_bound_argument_interpolates_the_graph_scope(self):
        """`{name}` in `tool_args` resolves the same way it does on a tool node."""
        echo = _Echo()
        graph = load_graph_from_dict({
            "name": "t",
            "inputs": [{"name": "who", "type": "string"}],
            "nodes": [{
                "id": "step", "kind": "base", "goal": "echo", "tools": ["echo"],
                "tool_args": {"text": "hello {who}"},
            }],
            "outputs": ["step"],
        })
        res = GraphExecutor(
            graph, provider=_ToolThenText(), log_traces=False,
            native_tools=ToolPool([echo]),
        ).run({"who": "world"})
        assert res.nodes["step"].error is None, res.nodes["step"].error
        assert echo.calls == [{"text": "hello world"}]

    def test_tool_settings_are_applied(self, tmp_path):
        """The other call site, broken the same way and for the same reason."""
        echo = _Echo()
        res = _run(
            [{
                "id": "step", "kind": "base", "goal": "echo", "tools": ["echo"],
                "tool_settings": {"echo": {"root": str(tmp_path)}},
            }],
            _ToolThenText(), [echo],
        )
        assert res.nodes["step"].error is None, res.nodes["step"].error
        assert echo.calls == [{"text": "hi"}]


def test_a_react_node_still_needs_tools():
    """`base` taking tools must not make `react` with none suddenly legal — an
    agent loop with nothing to call invents the actions it reports.

    The executor records a node's failure rather than propagating it, so the refusal
    shows up as the node's `error`, which is where a run would report it.
    """
    res = _run([{"id": "r", "kind": "react", "goal": "act"}], _ToolThenText(),
               validate=False)  # the graph is invalid on purpose; this asserts the runtime
    assert "no tools" in (res.nodes["r"].error or "")
