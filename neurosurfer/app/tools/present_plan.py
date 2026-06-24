from __future__ import annotations

from pydantic import BaseModel, Field

from neurosurfer.tools.base import Tool, ToolContext, ToolResult
from neurosurfer.tools.registry import register_tool_factory


class PresentPlanArgs(BaseModel):
    plan: str = Field(description="The proposed plan, in markdown.")
    title: str = Field(default="Proposed plan", description="Short title for the plan.")


class PresentPlanTool(Tool):
    name = "present_plan"
    description = (
        "Present a plan to the user for approval before executing. Writes the plan "
        "to durable state and blocks until the user approves or requests changes. "
        "Only after approval may writes/shell proceed (when plan mode is active)."
    )
    input_model = PresentPlanArgs

    async def call(self, args: PresentPlanArgs, ctx: ToolContext) -> ToolResult:  # type: ignore[override]
        if ctx.durable is not None:
            ctx.durable.set_plan(args.title, args.plan)
        approved, feedback = await ctx.io.request_plan_approval(args.plan)
        if approved:
            return ToolResult.ok(
                "Plan approved by the user. You may now execute it.",
                plan_presented=True,
                plan_approved=True,
            )
        msg = "User did not approve the plan."
        if feedback:
            msg += f" Their instructions: {feedback}"
        msg += " Revise the plan based on this feedback and present it again."
        return ToolResult.ok(msg, plan_presented=True, plan_approved=False)


# Contribute this product tool to the framework registry when the app is imported.
register_tool_factory(PresentPlanTool)
