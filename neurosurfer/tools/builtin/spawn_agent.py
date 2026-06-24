from __future__ import annotations

from pydantic import BaseModel, Field

from ..base import Tool, ToolContext, ToolResult


class SpawnAgentArgs(BaseModel):
    agent_type: str = Field(
        description="Built-in sub-agent type (e.g. explore, analyzer, writer, verifier)."
    )
    prompt: str = Field(description="The task/instructions for the sub-agent.")


class SpawnAgentTool(Tool):
    name = "spawn_agent"
    description = (
        "Spawn a sub-agent with its own fresh context and tool subset to handle a "
        "focused piece of work. Only the sub-agent's final report returns to you. "
        "Issue several spawn_agent calls in one turn to run them in parallel."
    )
    input_model = SpawnAgentArgs

    async def call(self, args: SpawnAgentArgs, ctx: ToolContext) -> ToolResult:  # type: ignore[override]
        if ctx.spawn is None:
            return ToolResult.error("Sub-agents are not available in this run.")
        try:
            report = await ctx.spawn(args.agent_type, args.prompt)
        except Exception as e:  # noqa: BLE001 - surface as tool error
            return ToolResult.error(f"Sub-agent '{args.agent_type}' failed: {e}")
        return ToolResult.ok(report)
