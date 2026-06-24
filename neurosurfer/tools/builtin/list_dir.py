from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from ..base import Tool, ToolContext, ToolResult
from ..utils import resolve_path

_IGNORE = {".git", "__pycache__", "node_modules", ".venv", "venv", ".mypy_cache", ".ruff_cache"}
MAX_ENTRIES = 400


class ListDirArgs(BaseModel):
    path: str = Field(default=".", description="Directory to list (relative or absolute).")
    pattern: str | None = Field(
        default=None,
        description="Optional glob (e.g. '**/*.py') evaluated under path.",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        # Accept "paths" (array or string) as an alias for "path".
        if "paths" in data and "path" not in data:
            val = data.pop("paths")
            data["path"] = val[0] if isinstance(val, list) and val else str(val)
        return data


class ListDirTool(Tool):
    name = "list_dir"
    description = (
        "List directory entries, or glob a pattern under a directory "
        "(e.g. pattern='**/*.py'). Skips VCS/build noise."
    )
    input_model = ListDirArgs

    def is_read_only(self, args: BaseModel) -> bool:
        return True

    def progress_message(self, args: dict) -> str:
        path = args.get("path") or args.get("paths") or "."
        if isinstance(path, list):
            path = path[0] if path else "."
        pattern = args.get("pattern")
        if pattern:
            return f"Exploring {path} for {pattern}…"
        return f"Exploring directory {path}…"

    async def call(self, args: ListDirArgs, ctx: ToolContext) -> ToolResult:  # type: ignore[override]
        base = resolve_path(ctx.cwd, args.path)
        if not base.exists():
            return ToolResult.error(f"Path not found: {args.path}")
        if not base.is_dir():
            return ToolResult.error(f"{args.path} is not a directory.")

        if args.pattern:
            matches = sorted(base.glob(args.pattern))
            matches = [m for m in matches if not _ignored(m)]
            if not matches:
                return ToolResult.ok(f"No matches for '{args.pattern}' under {args.path}")
            rendered = "\n".join(self._render(m, ctx.cwd) for m in matches[:MAX_ENTRIES])
            return ToolResult.ok(_cap(rendered, len(matches)))

        entries = sorted(base.iterdir(), key=lambda p: (p.is_file(), p.name))
        entries = [e for e in entries if not _ignored(e)]
        rendered = "\n".join(self._render(e, ctx.cwd) for e in entries[:MAX_ENTRIES])
        return ToolResult.ok(_cap(rendered or "(empty directory)", len(entries)))

    @staticmethod
    def _render(p: Path, cwd: Path) -> str:
        try:
            rel = p.relative_to(cwd)
        except ValueError:
            rel = p
        return f"{rel}/" if p.is_dir() else str(rel)


def _ignored(p: Path) -> bool:
    return any(part in _IGNORE for part in p.parts)


def _cap(rendered: str, total: int) -> str:
    if total > MAX_ENTRIES:
        return rendered + f"\n… [{total - MAX_ENTRIES} more entries]"
    return rendered
