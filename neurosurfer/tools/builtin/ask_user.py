from __future__ import annotations

from pydantic import BaseModel, Field

from ..base import Tool, ToolContext, ToolResult


class AskUserArgs(BaseModel):
    question: str = Field(description="A clarifying question to ask the user.")
    options: list[str] | None = Field(
        default=None, description="Optional list of suggested answers."
    )


class AskUserTool(Tool):
    name = "ask_user"
    description = (
        "Ask the user a clarifying question and wait for their answer. Use when a "
        "decision genuinely requires user input — not for things you can infer."
    )
    input_model = AskUserArgs

    async def call(self, args: AskUserArgs, ctx: ToolContext) -> ToolResult:  # type: ignore[override]
        answer = await ctx.io.ask(args.question, args.options)
        return ToolResult.ok(f"User answered: {answer}")
