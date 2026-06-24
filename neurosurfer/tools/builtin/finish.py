from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ..base import Tool, ToolContext, ToolResult


class FinishArgs(BaseModel):
    summary: str = Field(description="Final report: what was accomplished.")
    status: Literal["success", "partial", "failed"] = "success"


class FinishTool(Tool):
    name = "finish"
    description = (
        "Signal that the task is complete. Provide a final report of what was done. "
        "Calling this ends the run."
    )
    input_model = FinishArgs

    def progress_message(self, args: dict) -> str:
        return "Finishing up…"

    async def call(self, args: FinishArgs, ctx: ToolContext) -> ToolResult:  # type: ignore[override]
        return ToolResult.ok(
            args.summary, finished=True, status=args.status, report=args.summary
        )
