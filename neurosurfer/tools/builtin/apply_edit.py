from __future__ import annotations

from pydantic import BaseModel, Field

from ..base import FileState, Tool, ToolContext, ToolResult
from ..utils import resolve_path


class ApplyEditArgs(BaseModel):
    path: str = Field(description="File to edit.")
    old_string: str = Field(description="Exact text to replace (must be unique unless replace_all).")
    new_string: str = Field(description="Replacement text.")
    replace_all: bool = Field(default=False, description="Replace every occurrence.")


class ApplyEditTool(Tool):
    name = "apply_edit"
    description = (
        "Replace an exact string in a file. Fails if old_string is missing or "
        "ambiguous (unless replace_all). Detects edits to files changed on disk "
        "since they were last read (staleness check)."
    )
    input_model = ApplyEditArgs

    def is_read_only(self, args: BaseModel) -> bool:
        return False

    async def call(self, args: ApplyEditArgs, ctx: ToolContext) -> ToolResult:  # type: ignore[override]
        path = resolve_path(ctx.cwd, args.path)
        if not path.exists():
            return ToolResult.error(f"File not found: {args.path}")
        try:
            current = path.read_text(encoding="utf-8")
        except OSError as e:
            return ToolResult.error(f"Could not read {args.path}: {e}")

        # Staleness check: if we recorded this file and it changed on disk, refuse.
        recorded = ctx.file_state.get(str(path))
        if recorded is not None and recorded.content != current:
            if path.stat().st_mtime > recorded.mtime:
                return ToolResult.error(
                    f"{args.path} changed on disk since it was last read. "
                    "Re-read it before editing."
                )

        if args.old_string == args.new_string:
            return ToolResult.error("old_string and new_string are identical.")

        count = current.count(args.old_string)
        if count == 0:
            return ToolResult.error(
                f"old_string not found in {args.path}. Read the file and match exactly."
            )
        if count > 1 and not args.replace_all:
            return ToolResult.error(
                f"old_string is ambiguous ({count} matches). Add context or set replace_all."
            )

        updated = current.replace(args.old_string, args.new_string)
        try:
            path.write_text(updated, encoding="utf-8")
        except OSError as e:
            return ToolResult.error(f"Could not write {args.path}: {e}")

        stat = path.stat()
        ctx.file_state[str(path)] = FileState(
            mtime=stat.st_mtime, size=stat.st_size, content=updated
        )
        replaced = count if args.replace_all else 1
        return ToolResult.ok(f"Edited {args.path} ({replaced} replacement(s)).")
