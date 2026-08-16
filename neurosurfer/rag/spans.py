"""Locating a chunk inside the document it came from.

The chunk strategies return text, not spans, and there are a dozen of them —
threading offsets through every return type would touch far more than it buys.
Scanning for each chunk works for any strategy that *slices* rather than
rewrites, which is all of them, and yields nothing for one that does not rather
than a wrong span.
"""

from __future__ import annotations

__all__ = ["locate_chunks"]


def locate_chunks(text: str, chunks: list[str]) -> list[tuple[int, int] | None]:
    """Where each of *chunks* sits in *text*, or ``None`` if it cannot be found.

    Chunks usually arrive in document order, so the search starts from where the
    last one was found — linear rather than quadratic on the common path.

    **But overlap breaks a forward-only scan.** An overlapping chunker emits
    chunks that start *before* the previous one ended, so advancing the cursor
    past the whole previous chunk makes the next search begin after its own
    match. Advancing half-way is not enough either: with a three-word overlap on
    a four-word chunk the next match still starts behind the cursor. So a failed
    forward search falls back to a search from the beginning — an earlier
    identical passage is a plausible span, and no span at all is not.
    """
    out: list[tuple[int, int] | None] = []
    cursor = 0
    for chunk in chunks:
        if not chunk:
            out.append(None)
            continue
        start = text.find(chunk, cursor)
        if start < 0:
            start = text.find(chunk)
        if start < 0:
            out.append(None)
            continue
        out.append((start, start + len(chunk)))
        cursor = start + max(1, len(chunk) // 2)
    return out
