from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from ..base import FileState, Tool, ToolContext, ToolResult
from ..utils import resolve_path


class EditItem(BaseModel):
    old_string: str = Field(description="Exact text to replace (must be unique unless replace_all).")
    new_string: str = Field(description="Replacement text.")
    replace_all: bool = Field(default=False, description="Replace every occurrence of this hunk.")


class ApplyEditArgs(BaseModel):
    path: str = Field(description="File to edit.")
    old_string: str | None = Field(
        default=None,
        description="Exact text to replace (single-edit mode). Omit when using `edits`.",
    )
    new_string: str | None = Field(
        default=None, description="Replacement text (single-edit mode)."
    )
    replace_all: bool = Field(
        default=False, description="Replace every occurrence (single-edit mode)."
    )
    edits: list[EditItem] | None = Field(
        default=None,
        description=(
            "Multiple old_string -> new_string hunks applied atomically, in order, "
            "to the same file — use for a multi-spot change instead of several calls. "
            "If any hunk fails to match, none of the edits are written."
        ),
    )

    @model_validator(mode="after")
    def _check_shape(self) -> ApplyEditArgs:
        if self.edits:
            if self.old_string is not None or self.new_string is not None:
                raise ValueError("Provide either old_string/new_string or edits, not both.")
        elif self.old_string is None or self.new_string is None:
            raise ValueError("Provide old_string and new_string, or edits for multiple hunks.")
        return self

    @property
    def hunks(self) -> list[EditItem]:
        if self.edits:
            return self.edits
        assert self.old_string is not None and self.new_string is not None
        return [EditItem(old_string=self.old_string, new_string=self.new_string, replace_all=self.replace_all)]


class ApplyEditTool(Tool):
    name = "apply_edit"
    description = (
        "Replace exact string(s) in a file. Fails if an old_string is missing or "
        "ambiguous (unless replace_all). Pass `edits` (a list of old_string/new_string "
        "hunks) to apply several changes to the same file atomically — if any hunk "
        "fails to match, nothing is written. Detects edits to files changed on disk "
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

        hunks = args.hunks
        working = current
        total_replaced = 0
        for i, edit in enumerate(hunks, start=1):
            prefix = "" if len(hunks) == 1 else f"Edit {i}: "
            if edit.old_string == edit.new_string:
                return ToolResult.error(f"{prefix}old_string and new_string are identical.")
            count = working.count(edit.old_string)
            if count == 0:
                return ToolResult.error(
                    f"{prefix}old_string not found in {args.path}. Read the file and match exactly."
                )
            if count > 1 and not edit.replace_all:
                return ToolResult.error(
                    f"{prefix}old_string is ambiguous ({count} matches). Add context or set replace_all."
                )
            working = working.replace(edit.old_string, edit.new_string)
            total_replaced += count if edit.replace_all else 1

        try:
            path.write_text(working, encoding="utf-8")
        except OSError as e:
            return ToolResult.error(f"Could not write {args.path}: {e}")

        stat = path.stat()
        ctx.file_state[str(path)] = FileState(
            mtime=stat.st_mtime, size=stat.st_size, content=working
        )
        if len(hunks) == 1:
            return ToolResult.ok(f"Edited {args.path} ({total_replaced} replacement(s)).")
        return ToolResult.ok(
            f"Edited {args.path} ({len(hunks)} hunks, {total_replaced} replacement(s) total)."
        )
