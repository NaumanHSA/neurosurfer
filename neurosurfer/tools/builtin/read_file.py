from __future__ import annotations

import base64

from pydantic import BaseModel, Field

from ...llm.types import ImageBlock
from ..base import FileState, Tool, ToolContext, ToolResult
from ..utils import is_probably_binary, resolve_path, with_line_numbers

MAX_LINES = 2000
MAX_LINE_LEN = 2000

# Image files are returned as an ImageBlock (for vision models) rather than text.
_IMAGE_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}
# Guard against blowing up history with a huge base64 payload.
MAX_IMAGE_BYTES = 5 * 1024 * 1024


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
        media_type = _IMAGE_MEDIA_TYPES.get(path.suffix.lower())
        if media_type is not None:
            return self._read_image(path, args.path, media_type)
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

    def _read_image(self, path, display: str, media_type: str) -> ToolResult:
        """Return an image file as an ImageBlock for vision-capable models.

        Non-vision models drop the image at the provider boundary, falling back to the
        text note in ``content``.
        """
        try:
            raw = path.read_bytes()
        except OSError as e:
            return ToolResult.error(f"Could not read {display}: {e}")
        if len(raw) > MAX_IMAGE_BYTES:
            return ToolResult.error(
                f"{display} is {len(raw) // 1024} KB; images over "
                f"{MAX_IMAGE_BYTES // (1024 * 1024)} MB are not supported."
            )
        data = base64.b64encode(raw).decode("ascii")
        note = f"Loaded image {display} ({media_type}, {len(raw) // 1024} KB)."
        return ToolResult.with_images(note, [ImageBlock.from_base64(data, media_type)])
