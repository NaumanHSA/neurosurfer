"""Tool that writes a single node's agent YAML to the workflow's staging directory.

Called by the architect's `write_nodes` react-node once per workflow node it designs.
Files land in ``~/.neurosurfer/projects/<workflow_name>/agents/<node_id>.yaml`` and
are picked up by the ``assemble`` function-node.
"""

from __future__ import annotations

import yaml
from pydantic import BaseModel, Field

from neurosurfer.tools.base import Tool, ToolContext, ToolResult
from neurosurfer.tools.registry import register_tool_factory


class WriteWorkflowNodeArgs(BaseModel):
    workflow_name: str = Field(description="Destination workflow name (slug, no spaces).")
    node_id: str = Field(description="Node identifier (snake_case). Becomes the filename.")
    content: str = Field(
        description=(
            "YAML content for the agent node config. "
            "Must include 'id', 'purpose', and 'kind' at minimum."
        )
    )


class WriteWorkflowNodeTool(Tool):
    """Write one agent-node YAML file into the workflow's staging area."""

    name = "write_workflow_node"
    description = (
        "Write a single workflow node's YAML configuration to the staging directory "
        "at ~/.neurosurfer/projects/<workflow_name>/agents/<node_id>.yaml. "
        "Call once per node you are designing. The assemble step reads all staged "
        "files to build the final workflow package."
    )
    input_model = WriteWorkflowNodeArgs

    def is_read_only(self, args: BaseModel) -> bool:
        return False

    async def call(  # type: ignore[override]
        self, args: WriteWorkflowNodeArgs, ctx: ToolContext
    ) -> ToolResult:
        from neurosurfer.config.projects import ProjectsConfig

        projects = ProjectsConfig()
        agents_dir = projects.project_dir(args.workflow_name) / "agents"
        agents_dir.mkdir(parents=True, exist_ok=True)

        node_id = args.node_id.strip().lower().replace(" ", "_").replace("-", "_")
        if not node_id:
            return ToolResult.error("node_id must not be empty.")

        # Validate YAML parses cleanly
        try:
            data = yaml.safe_load(args.content) or {}
        except yaml.YAMLError as exc:
            return ToolResult.error(f"Invalid YAML content: {exc}")

        if not isinstance(data, dict):
            return ToolResult.error("content must be a YAML mapping (dict), not a scalar.")

        # Reject tools that aren't registered so the agent self-corrects instead of
        # producing a workflow that fails at run time. The Architect must compose from
        # the real catalog (read_file, run_command, search, write_file, … plus any
        # generated tools).
        from neurosurfer.tools.registry import workflow_node_tool_names  # noqa: PLC0415

        valid_tools = workflow_node_tool_names()
        declared_tools = data.get("tools") or []
        if isinstance(declared_tools, str):
            declared_tools = [declared_tools]
        unknown = [t for t in declared_tools if t not in valid_tools]
        if unknown:
            valid = ", ".join(sorted(valid_tools))
            return ToolResult.error(
                f"Node '{node_id}' references unregistered tool(s): {', '.join(unknown)}. "
                f"Only these tools exist: {valid}. "
                "Re-write this node using ONLY tools from that list — compose general "
                "tools (e.g. use run_command + read_file to analyse code) instead of "
                "inventing specialised tools."
            )

        # Ensure the id field matches
        data["id"] = node_id

        dest = agents_dir / f"{node_id}.yaml"
        try:
            dest.write_text(
                yaml.dump(data, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
        except OSError as exc:
            return ToolResult.error(f"Cannot write {dest}: {exc}")

        return ToolResult.ok(f"Wrote node '{node_id}' → {dest}")


# Contribute this architect tool to the framework registry when workflows is imported.
register_tool_factory(WriteWorkflowNodeTool)
