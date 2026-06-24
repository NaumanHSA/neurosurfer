"""R3+R4: native provider + ToolPool drive GraphExecutor (no adapter layer).

Replaces the old Phase-A tests that verified ProviderChatModel / MasterAgentToolAdapter.
Those adapter classes still exist for backward compat but are no longer the primary path.
"""

from __future__ import annotations

from pathlib import Path

from neurosurfer.llm.base import Provider
from neurosurfer.llm.capabilities import ProviderCapabilities
from neurosurfer.llm.types import CanonicalResponse, Done, TextBlock, Usage
from neurosurfer.tools.base import ToolContext, ToolPool
from neurosurfer.tools.registry import all_tools
from neurosurfer.graph import (
    Graph,
    GraphExecutor,
    GraphNode,
    NodeMode,
)


class _EchoProvider(Provider):
    """Minimal provider that echoes the last user message back as the reply."""

    model = "echo-model"
    capabilities = ProviderCapabilities(
        context_window=8192,
        max_output_tokens=2048,
        supports_thinking=False,
        supports_prompt_cache=False,
        supports_token_count=False,
        tool_call_style="openai",
    )

    async def stream(self, messages, system, tools, config):
        last = messages[-1].text() if messages else ""
        yield Done(
            response=CanonicalResponse(
                content=[TextBlock(text=f"REPLY[{last[:60]}]")],
                stop_reason="stop",
                usage=Usage(),
            )
        )

    async def count_tokens(self, messages, system, tools):
        return 0


class _StubIO:
    async def ask(self, question, options=None):
        return ""

    async def request_plan_approval(self, plan):
        return (True, "")

    async def request_shell_approval(self, command, reason):
        return True

    async def request_write_approval(self, path, summary):
        return "deny"

    def notify(self, message):
        return None


def test_native_provider_drives_graph_executor():
    """Native provider (no ProviderChatModel wrapper) drives a 2-node base graph."""
    provider = _EchoProvider()
    graph = Graph(
        name="native_smoke",
        inputs=[{"name": "topic", "type": "string"}],
        nodes=[
            GraphNode(id="research", purpose="Research {topic}", mode=NodeMode.TEXT),
            GraphNode(
                id="summarize",
                depends_on=["research"],
                purpose="Summarize the research",
                mode=NodeMode.TEXT,
            ),
        ],
        outputs=["summarize"],
    )
    executor = GraphExecutor(graph, provider=provider, log_traces=False)
    result = executor.run({"topic": "graph workflows"})

    assert result.final, "final outputs should be populated"
    assert "summarize" in result.final
    assert all(node.error is None for node in result.nodes.values())
    # Dependency output must reach the downstream node (echo nests the prior reply).
    assert "REPLY[" in str(result.nodes["summarize"].raw_output)


def test_native_tool_pool_drives_tool_node():
    """Native ToolPool (no MasterAgentToolAdapter) executes a tool-kind node."""
    list_dir_tool = {t.name: t for t in all_tools()}["list_dir"]
    repo_root = Path(__file__).resolve().parent.parent
    ctx = ToolContext(cwd=repo_root, io=_StubIO())
    pool = ToolPool([list_dir_tool])

    graph = Graph(
        name="tool_node_smoke",
        nodes=[
            GraphNode(
                id="list_files",
                kind="tool",
                tools=["list_dir"],
                tool_args={"path": "."},
            ),
        ],
        outputs=["list_files"],
    )
    executor = GraphExecutor(
        graph, provider=_EchoProvider(), native_tools=pool, tool_ctx=ctx, log_traces=False
    )
    result = executor.run({})

    assert result.nodes["list_files"].error is None
    assert result.nodes["list_files"].raw_output  # non-empty listing
