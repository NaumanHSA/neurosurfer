from __future__ import annotations

import fnmatch
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
        "(e.g. pattern='**/*.py'). Skips VCS/build noise and .gitignore'd paths."
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

        patterns = _load_gitignore_patterns(ctx.cwd)

        def skip(p: Path) -> bool:
            return _ignored(p) or _gitignored(p, ctx.cwd, patterns)

        if args.pattern:
            matches = sorted(base.glob(args.pattern))
            matches = [m for m in matches if not skip(m)]
            if not matches:
                return ToolResult.ok(f"No matches for '{args.pattern}' under {args.path}")
            rendered = "\n".join(self._render(m, ctx.cwd) for m in matches[:MAX_ENTRIES])
            return ToolResult.ok(_cap(rendered, len(matches)))

        entries = sorted(base.iterdir(), key=lambda p: (p.is_file(), p.name))
        entries = [e for e in entries if not skip(e)]
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


def _load_gitignore_patterns(root: Path) -> list[str]:
    """Best-effort top-level ``.gitignore`` reader: plain glob patterns, no negation.

    A leading ``**/`` (by far the most common form — "match at any depth") is
    stripped so the pattern compares directly against a candidate's basename.
    """
    gitignore = root / ".gitignore"
    if not gitignore.is_file():
        return []
    try:
        lines = gitignore.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return []
    patterns = []
    for line in lines:
        line = line.strip().rstrip("/")
        if not line or line.startswith("#") or line.startswith("!"):
            continue
        if line.startswith("**/"):
            line = line[3:]
        patterns.append(line)
    return patterns


def _gitignored(p: Path, cwd: Path, patterns: list[str]) -> bool:
    if not patterns:
        return False
    try:
        rel = p.relative_to(cwd).as_posix()
    except ValueError:
        return False
    return any(
        fnmatch.fnmatch(p.name, pat) or fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(rel, f"*/{pat}")
        for pat in patterns
    )


def _cap(rendered: str, total: int) -> str:
    if total > MAX_ENTRIES:
        return rendered + f"\n… [{total - MAX_ENTRIES} more entries]"
    return rendered
