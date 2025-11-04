from neurosurfer.tools.base_tool import BaseTool, ToolResponse
from neurosurfer.tools.tool_spec import ToolSpec, ToolParam, ToolReturn
from pathlib import Path
import os

class WriteFileTool(BaseTool):
    spec = ToolSpec(
        name="write_file",
        description="Write or overwrite a text file in the repository.",
        when_to_use="When you need to create or modify a source/test/docs file.",
        inputs=[
            ToolParam(name="path", type="string", description="Repo-relative file path", required=True),
            ToolParam(name="content", type="string", description="Full file content (UTF-8)", required=True),
        ],
        returns=ToolReturn(type="string", description="OK or error message"),
    )

    def __call__(self, path: str, content: str, **kwargs) -> ToolResponse:
        # Basic guardrail: limit to repo (cwd) and block dangerous paths
        path_obj = Path(path).resolve()
        repo_root = Path(os.getcwd()).resolve()
        if not str(path_obj).startswith(str(repo_root)):
            return ToolResponse(final_answer=False, observation=f"Blocked write outside repo: {path_obj}")
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        path_obj.write_text(content, encoding="utf-8")
        return ToolResponse(final_answer=False, observation=f"Wrote {path}")
