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


class PathOutsideRoot(ValueError):
    """A path argument resolved outside the directory its tool was given.

    Carries both halves because the message a person needs names them: *what*
    they asked for and *where* the tool is allowed to work. Raised, not returned,
    so it cannot be mistaken for a tool result — the caller turns it into one.
    """

    def __init__(self, raw: str, resolved: Path, root: Path) -> None:
        self.raw = raw
        self.resolved = resolved
        self.root = root
        super().__init__(
            f"'{raw}' is outside this step's directory. "
            f"It is configured to work under {root}, and that path resolves to {resolved}."
        )


def resolve_within(root: Path, raw: str) -> Path:
    """Resolve ``raw`` under ``root``, refusing anything that escapes it.

    The confinement rule, in one place, because it is the kind of check that is
    subtly wrong every time it is written twice. Three ways out of a directory and
    all three are closed here:

    - a **relative** path resolves under *root*, as it would against a cwd;
    - ``..`` is normalised **before** the comparison, so ``out/../../etc`` is
      judged by where it lands rather than by how it is spelled;
    - an **absolute** path is kept absolute and then checked, so naming the root
      exactly still works and naming anything else does not.

    Symlinks are resolved on both sides (``strict=False``, since the destination
    of a write does not exist yet). A link inside the root pointing out of it is
    therefore refused — which is the point of resolving rather than string-matching.

    Raises :class:`PathOutsideRoot`.
    """
    base = Path(root).expanduser().resolve(strict=False)
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = base / p
    resolved = p.resolve(strict=False)
    if resolved != base and base not in resolved.parents:
        raise PathOutsideRoot(raw, resolved, base)
    return resolved


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
