from neurosurfer.tools.base_tool import BaseTool, ToolResponse
from neurosurfer.tools.tool_spec import ToolSpec, ToolParam, ToolReturn
import subprocess, json

class PytestRunTool(BaseTool):
    spec = ToolSpec(
        name="pytest_run",
        description="Run pytest for a path or the whole repo and return a short JSON summary.",
        when_to_use="After generating/modifying code or tests.",
        inputs=[
            ToolParam(name="target", type="string", description="Path (file/dir) to test; use '.' for all", required=True),
        ],
        returns=ToolReturn(type="object", description="JSON summary with exit_code and lines"),
    )

    def __call__(self, target: str, **kwargs) -> ToolResponse:
        try:
            proc = subprocess.run(["pytest", "-q", target], capture_output=True, text=True)
            summary = {
                "exit_code": proc.returncode,
                "stdout_tail": proc.stdout.strip().splitlines()[-20:],
                "stderr_tail": proc.stderr.strip().splitlines()[-20:],
            }
            return ToolResponse(final_answer=False, results=json.dumps(summary))
        except Exception as e:
            return ToolResponse(final_answer=False, results=f"pytest error: {e}")
