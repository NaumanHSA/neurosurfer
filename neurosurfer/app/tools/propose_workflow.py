from __future__ import annotations

from pydantic import BaseModel, Field

from neurosurfer.tools.base import Tool, ToolContext, ToolResult
from neurosurfer.tools.registry import register_tool_factory


class ProposeWorkflowArgs(BaseModel):
    intent: str = Field(
        description=(
            "Refined, expanded description of what the workflow should do — "
            "enough for the Architect to design it without further clarification."
        )
    )
    reason: str = Field(
        description="Why a durable, registered workflow beats a one-off execution here."
    )
    recurring: bool = Field(
        default=True,
        description="True if the user is likely to run this more than once.",
    )


class ProposeWorkflowTool(Tool):
    name = "propose_workflow"
    description = (
        "Propose handing the current task off to the Architect to design and register "
        "a reusable workflow pipeline. Use this when the task is recurring, multi-stage, "
        "or complex enough that a durable workflow is a better fit than a one-off run. "
        "The user must confirm before the handoff happens."
    )
    input_model = ProposeWorkflowArgs

    async def call(self, args: ProposeWorkflowArgs, ctx: ToolContext) -> ToolResult:  # type: ignore[override]
        header = (
            f"I'd like to design a reusable workflow for this:\n\n"
            f"**{args.intent}**\n\n"
            f"{args.reason}\n\n"
            f"The Architect will ask a few clarifying questions, then design and register "
            f"a workflow you can run any time with `/workflow run`."
        )
        confirmed = await ctx.io.request_plan_approval(header)
        approved, _ = confirmed if isinstance(confirmed, tuple) else (confirmed, "")
        if approved:
            return ToolResult.ok(
                "User confirmed the workflow handoff.",
                finished=True,
                status="handoff_workflow",
                report=args.intent,
                handoff_workflow=True,
            )
        return ToolResult.ok(
            "User declined the workflow. Continue solving the task as a one-off."
        )


register_tool_factory(ProposeWorkflowTool)
