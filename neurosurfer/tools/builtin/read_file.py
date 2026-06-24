from __future__ import annotations

from pydantic import BaseModel, Field

from ..base import FileState, Tool, ToolContext, ToolResult
from ..utils import is_probably_binary, resolve_path, with_line_numbers

MAX_LINES = 2000
MAX_LINE_LEN = 2000


class ReadFileArgs(BaseModel):
    path: str = Field(description="File path (absolute, or relative to the working dir).")
    offset: int = Field(default=1, ge=1, description="1-based line to start at.")
    limit: int = Field(default=MAX_LINES, ge=1, description="Max lines to read.")


class ReadFileTool(Tool):
    name = "read_file"
    description = (
        "Read a text file and return its contents with line numbers. "
        "Use offset/limit for large files. Records the file for staleness checks."
    )
    input_model = ReadFileArgs

    def is_read_only(self, args: BaseModel) -> bool:
        return True

    def progress_message(self, args: dict) -> str:
        path = args.get("path") or "file"
        return f"Reading file {path}…"

    async def call(self, args: ReadFileArgs, ctx: ToolContext) -> ToolResult:  # type: ignore[override]
        path = resolve_path(ctx.cwd, args.path)
        if not path.exists():
            return ToolResult.error(f"File not found: {args.path}")
        if path.is_dir():
            return ToolResult.error(f"{args.path} is a directory; use list_dir.")
        if is_probably_binary(path):
            return ToolResult.error(f"{args.path} appears to be a binary file.")
        try:
            full = path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            return ToolResult.error(f"Could not read {args.path}: {e}")

        # Record state for apply_edit staleness checks.
        stat = path.stat()
        ctx.file_state[str(path)] = FileState(
            mtime=stat.st_mtime, size=stat.st_size, content=full
        )

        lines = full.splitlines()
        start = args.offset
        window = lines[start - 1 : start - 1 + args.limit]
        clipped = [
            ln if len(ln) <= MAX_LINE_LEN else ln[:MAX_LINE_LEN] + "… [truncated]"
            for ln in window
        ]
        body = with_line_numbers("\n".join(clipped), start=start)
        suffix = ""
        if start - 1 + args.limit < len(lines):
            suffix = f"\n… [{len(lines) - (start - 1 + args.limit)} more lines]"
        if not body:
            return ToolResult.ok("(empty file)")
        return ToolResult.ok(body + suffix)
