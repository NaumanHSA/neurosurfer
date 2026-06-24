from neurosurfer.tools.base_tool import BaseTool, ToolResponse
from neurosurfer.tools.tool_spec import ToolSpec, ToolParam, ToolReturn
from pathlib import Path

class DocsUpdateTool(BaseTool):
    spec = ToolSpec(
        name="docs_update",
        description="Create or append content to a docs page (Markdown).",
        when_to_use="After adding a new tool/agent/adapter to document usage.",
        inputs=[
            ToolParam(name="path", type="string", description="Docs path (e.g., docs/tools/calculator.md)", required=True),
            ToolParam(name="append_md", type="string", description="Markdown to append (creates file if missing)", required=True),
        ],
        returns=ToolReturn(type="string", description="OK or error"),
    )

    def __call__(self, path: str, append_md: str, **kwargs) -> ToolResponse:
        fp = Path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        if fp.exists():
            prev = fp.read_text(encoding="utf-8")
            fp.write_text(prev.rstrip() + "\n\n" + append_md.strip() + "\n", encoding="utf-8")
            return ToolResponse(final_answer=False, results=f"Appended docs: {path}")
        else:
            fp.write_text(append_md.strip() + "\n", encoding="utf-8")
            return ToolResponse(final_answer=False, results=f"Created docs: {path}")
