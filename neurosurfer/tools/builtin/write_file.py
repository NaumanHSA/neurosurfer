from __future__ import annotations

from pydantic import BaseModel, Field

from ..base import FileState, Tool, ToolContext, ToolResult
from ..utils import resolve_path


class WriteFileArgs(BaseModel):
    path: str = Field(description="File path to write (relative or absolute).")
    content: str = Field(description="Full file contents to write.")


class WriteFileTool(Tool):
    name = "write_file"
    description = (
        "Create or overwrite a whole file with the given contents, creating parent "
        "directories as needed. Prefer apply_edit to modify an existing file — it "
        "changes only the targeted snippet. Use write_file for a new file or a "
        "deliberate full rewrite. Restricted to the Task's write scope."
    )
    input_model = WriteFileArgs

    def is_read_only(self, args: BaseModel) -> bool:
        return False

    async def call(self, args: WriteFileArgs, ctx: ToolContext) -> ToolResult:  # type: ignore[override]
        path = resolve_path(ctx.cwd, args.path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            existed = path.exists()
            path.write_text(args.content, encoding="utf-8")
        except OSError as e:
            return ToolResult.error(f"Could not write {args.path}: {e}")

        stat = path.stat()
        ctx.file_state[str(path)] = FileState(
            mtime=stat.st_mtime, size=stat.st_size, content=args.content
        )
        verb = "Updated" if existed else "Created"
        n = len(args.content.splitlines())
        return ToolResult.ok(f"{verb} {args.path} ({n} lines).")
