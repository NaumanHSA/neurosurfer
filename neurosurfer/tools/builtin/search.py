from __future__ import annotations

import asyncio
import re
import shutil
from pathlib import Path

from pydantic import BaseModel, Field

from ..base import Tool, ToolContext, ToolResult
from ..utils import is_probably_binary, resolve_path

_IGNORE_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", ".mypy_cache"}
MAX_RESULTS = 200
MAX_FILE_BYTES = 1_500_000
RIPGREP_TIMEOUT = 30


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

    def progress_message(self, args: dict) -> str:
        pat = args.get("pattern") or ""
        where = args.get("path") or "."
        return f"Searching for {pat!r} in {where}…"

    async def call(self, args: SearchArgs, ctx: ToolContext) -> ToolResult:  # type: ignore[override]
        try:
            flags = re.IGNORECASE if args.ignore_case else 0
            regex = re.compile(args.pattern, flags)
        except re.error as e:
            return ToolResult.error(f"Invalid regex: {e}")

        base = resolve_path(ctx.cwd, args.path)
        if not base.exists():
            return ToolResult.error(f"Path not found: {args.path}")

        if shutil.which("rg") is not None:
            rg_result = await _search_ripgrep(args, base, ctx.cwd)
            if rg_result is not None:
                return rg_result
            # rg errored (bad glob, etc.) — fall through to the pure-Python scan.

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


async def _search_ripgrep(args: SearchArgs, base: Path, cwd: Path) -> ToolResult | None:
    """Shell out to ``rg`` for context-aware, gitignore-respecting search.

    Returns ``None`` (never an error result) on any failure so the caller falls
    back to the pure-Python scan — ripgrep is a speed/quality upgrade, not a
    dependency.
    """
    try:
        target = base.relative_to(cwd)
    except ValueError:
        target = base

    cmd = ["rg", "--line-number", "--no-heading", "--color=never"]
    if args.ignore_case:
        cmd.append("-i")
    if not base.is_file():
        if args.glob and args.glob != "**/*":
            cmd += ["--glob", args.glob]
        for d in _IGNORE_DIRS:
            cmd += ["--glob", f"!{d}/**"]
    cmd += ["--", args.pattern, str(target)]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd),
        )
        out, _err = await asyncio.wait_for(proc.communicate(), timeout=RIPGREP_TIMEOUT)
    except (OSError, TimeoutError):
        return None

    if proc.returncode == 1:  # no matches — a valid, non-error outcome
        return ToolResult.ok(f"No matches for /{args.pattern}/.")
    if proc.returncode != 0:
        return None  # bad glob / regex syntax rg rejected — let the fallback try

    lines = [ln for ln in out.decode("utf-8", errors="replace").splitlines() if ln]
    if not lines:
        return ToolResult.ok(f"No matches for /{args.pattern}/.")

    truncated = len(lines) > args.max_results
    lines = lines[: args.max_results]
    body = "\n".join(f"{ln[:400]}" for ln in lines)
    if truncated:
        body += "\n… [result limit reached]"
    return ToolResult.ok(body)
