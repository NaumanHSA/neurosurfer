"""Shared helpers for filesystem-touching tools."""

from __future__ import annotations

from pathlib import Path

_BINARY_SNIFF_BYTES = 2048
_TEXT_CHARS = bytes(range(0x20, 0x7F)) + b"\n\r\t\f\b"


def resolve_path(cwd: Path, raw: str) -> Path:
    """Resolve ``raw`` against ``cwd`` (absolute paths kept as-is)."""
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = cwd / p
    return p


def is_probably_binary(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            chunk = fh.read(_BINARY_SNIFF_BYTES)
    except OSError:
        return False
    if not chunk:
        return False
    if b"\x00" in chunk:
        return True
    # Valid UTF-8 is always text — including non-ASCII source files.
    try:
        chunk.decode("utf-8")
        return False
    except UnicodeDecodeError:
        pass
    # Non-UTF-8: fall back to ASCII byte-ratio heuristic.
    nontext = chunk.translate(None, _TEXT_CHARS)
    return len(nontext) / len(chunk) > 0.30


def with_line_numbers(text: str, start: int = 1) -> str:
    lines = text.splitlines()
    width = len(str(start + len(lines) - 1)) if lines else 1
    return "\n".join(f"{str(i).rjust(width)}\t{line}" for i, line in enumerate(lines, start))
