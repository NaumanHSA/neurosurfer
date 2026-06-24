from neurosurfer.tools.base_tool import BaseTool, ToolResponse
from neurosurfer.tools.tool_spec import ToolSpec, ToolParam, ToolReturn
import subprocess

class LintTool(BaseTool):
    spec = ToolSpec(
        name="lint",
        description="Run linter (ruff) and return a compact report.",
        when_to_use="Before committing changes or after generation.",
        inputs=[ 
            ToolParam(name="path", type="string", description="Target path; '.' for repo", required=True),
            # ToolParam(name="max_lines", type="int", description="Max lines to return", required=False, default=50),
        ],
        returns=ToolReturn(type="string", description="Lint output (last lines)"),
    )

    def __init__(self):
        pass

    def __call__(self, path: str, **kwargs) -> ToolResponse:
        try:
            proc = subprocess.run(["ruff", "check", path], capture_output=True, text=True)
            out = (proc.stdout or proc.stderr).splitlines()[-50:]
            return ToolResponse(final_answer=False, results="\n".join(out))
        except Exception as e:
            return ToolResponse(final_answer=False, results=f"ruff error: {e}")
