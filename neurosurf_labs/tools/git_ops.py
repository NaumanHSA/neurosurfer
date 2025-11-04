from neurosurfer.tools.base_tool import BaseTool, ToolResponse
from neurosurfer.tools.tool_spec import ToolSpec, ToolParam, ToolReturn
import subprocess, json

class GitOpsTool(BaseTool):
    spec = ToolSpec(
        name="git_ops",
        description="Create a branch, add/commit changes, and show a short diff summary.",
        when_to_use="When a change batch is ready for review.",
        inputs=[
            ToolParam(name="branch", type="string", description="Branch name to create or reuse", required=True),
            ToolParam(name="message", type="string", description="Commit message", required=True),
        ],
        returns=ToolReturn(type="object", description="JSON with branch, commit, diff summary"),
    )

    def __call__(self, branch: str, message: str, **kwargs) -> ToolResponse:
        def run(cmd):
            return subprocess.run(cmd, capture_output=True, text=True)
        run(["git", "checkout", "-B", branch])
        run(["git", "add", "-A"])
        c = run(["git", "commit", "-m", message])
        d = run(["git", "diff", "--stat", "HEAD~1..HEAD"])
        obs = {"branch": branch, "commit_stdout": c.stdout, "commit_stderr": c.stderr, "diff_stat": d.stdout}
        return ToolResponse(final_answer=False, observation=json.dumps(obs))
