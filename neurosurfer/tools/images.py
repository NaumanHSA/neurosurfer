"""Shared image loading + detection used by read_file and prompt auto-attach.

Two entry points:
  * :func:`load_image_block` — bytes on disk → an :class:`ImageBlock` (base64) plus
    a short human-readable note (or ``None`` + an error string).
  * :func:`find_image_paths` — pull image file paths out of free-form user text so a
    prompt like ``explain /path/to/pic.jpeg`` attaches the image directly, without
    relying on the model to route to ``read_file`` first.

Non-vision models drop the resulting block (with a text note) at the provider
boundary, so attaching is always safe.
"""

from __future__ import annotations

import base64
import re
from pathlib import Path

from ..llm.types import ImageBlock
from .utils import resolve_path

# Image files are returned as an ImageBlock (for vision models) rather than text.
IMAGE_MEDIA_TYPES: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}
# Guard against blowing up history with a huge base64 payload.
MAX_IMAGE_BYTES = 5 * 1024 * 1024

_EXT_RE = re.compile(r"\.(?:png|jpe?g|gif|webp)\b", re.IGNORECASE)


def load_image_block(path: Path, display: str | None = None) -> tuple[ImageBlock | None, str]:
    """Load an image file into an ``ImageBlock``.

    Returns ``(block, note)`` on success or ``(None, error)`` if the type is
    unsupported, the file is unreadable, or it exceeds :data:`MAX_IMAGE_BYTES`.
    """
    display = display or path.name
    media_type = IMAGE_MEDIA_TYPES.get(path.suffix.lower())
    if media_type is None:
        return None, f"{display} is not a supported image type."
    try:
        raw = path.read_bytes()
    except OSError as e:
        return None, f"Could not read {display}: {e}"
    if len(raw) > MAX_IMAGE_BYTES:
        return None, (
            f"{display} is {len(raw) // 1024} KB; images over "
            f"{MAX_IMAGE_BYTES // (1024 * 1024)} MB are not supported."
        )
    data = base64.b64encode(raw).decode("ascii")
    note = f"Loaded image {display} ({media_type}, {len(raw) // 1024} KB)."
    return ImageBlock.from_base64(data, media_type), note


def find_image_paths(text: str, cwd: Path) -> list[tuple[str, Path]]:
    """Find existing image files referenced in free-form user text.

    Tolerates spaces in filenames (e.g. ``WhatsApp Image 2026-06-30 at 1.jpeg``) by
    extending leftward from each image extension to the longest whitespace-delimited
    prefix that resolves to an existing file. Returns de-duplicated
    ``(display, resolved_path)`` pairs in the order they appear.
    """
    found: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for m in _EXT_RE.finditer(text):
        head = text[: m.end()]
        # Candidate start offsets: string start, plus just-after each whitespace run.
        # Trying the smallest offset first means the longest candidate wins, so a path
        # with spaces is preferred over its trailing single-word fragment.
        starts = sorted({0, *(i + 1 for i, ch in enumerate(head) if ch.isspace())})
        for s in starts:
            cand = head[s:].strip().strip("'\"`")
            if not cand:
                continue
            p = resolve_path(cwd, cand)
            key = str(p)
            # A candidate that cannot even be *asked* about is not a file.
            #
            # The longest candidate is tried first, and for a node's turn that
            # means the whole task text up to the extension — hundreds of bytes,
            # which `stat()` rejects with ENAMETOOLONG rather than "no". That
            # error escaped to the caller, so a prompt that merely *mentioned*
            # an image killed the run before the model was asked anything.
            # `is_file()` only swallows ENOENT/ENOTDIR/EBADF/ELOOP; everything
            # else here means "not a usable path", which is the same answer.
            try:
                is_file = p.is_file()
            except OSError:
                continue
            if is_file and key not in seen:
                seen.add(key)
                found.append((cand, p))
                break
    return found
