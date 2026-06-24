from __future__ import annotations

import re

from pydantic import BaseModel, Field

from ..base import Tool, ToolContext, ToolResult
from ..utils import is_probably_binary, resolve_path

_IGNORE_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", ".mypy_cache"}
MAX_RESULTS = 200
MAX_FILE_BYTES = 1_500_000


class SearchArgs(BaseModel):
    pattern: str = Field(description="Regular expression to search for.")
    path: str = Field(default=".", description="Directory (or file) to search under.")
    glob: str = Field(default="**/*", description="File glob to include.")
    ignore_case: bool = Field(default=False)
    max_results: int = Field(default=MAX_RESULTS, ge=1, le=2000)


class SearchTool(Tool):
    name = "search"
    description = (
        "Search file contents by regular expression (ripgrep-style). "
        "Returns file:line: matched-line. Filter files with the glob argument."
    )
    input_model = SearchArgs

    def is_read_only(self, args: BaseModel) -> bool:
        return True

    async def call(self, args: SearchArgs, ctx: ToolContext) -> ToolResult:  # type: ignore[override]
        try:
            flags = re.IGNORECASE if args.ignore_case else 0
            regex = re.compile(args.pattern, flags)
        except re.error as e:
            return ToolResult.error(f"Invalid regex: {e}")

        base = resolve_path(ctx.cwd, args.path)
        if not base.exists():
            return ToolResult.error(f"Path not found: {args.path}")

        files = [base] if base.is_file() else sorted(base.glob(args.glob))
        results: list[str] = []
        scanned = 0
        for f in files:
            if not f.is_file() or any(p in _IGNORE_DIRS for p in f.parts):
                continue
            try:
                if f.stat().st_size > MAX_FILE_BYTES or is_probably_binary(f):
                    continue
                text = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            scanned += 1
            try:
                rel = f.relative_to(ctx.cwd)
            except ValueError:
                rel = f
            for lineno, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    results.append(f"{rel}:{lineno}: {line.strip()[:300]}")
                    if len(results) >= args.max_results:
                        body = "\n".join(results)
                        return ToolResult.ok(body + "\n… [result limit reached]")
        if not results:
            return ToolResult.ok(f"No matches for /{args.pattern}/ in {scanned} files.")
        return ToolResult.ok("\n".join(results))
