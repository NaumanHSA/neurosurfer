"""Two ways to cut prose that a fixed size cannot.

The chunker's existing strategies are structural — an AST for Python, headers for
Markdown — and fall back to a fixed character window for everything else. A fixed
window cuts mid-sentence and mid-argument, so the half of a point that survives
in a chunk is often the half that does not answer the question.

Both of these register through the chunker's own `register_custom`, so they are
options rather than a replacement:

    chunker.register_custom("semantic", make_semantic_handler(embedder))
    chunker.use_custom_for_ext([".md", ".txt"], "semantic")
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence

__all__ = [
    "make_semantic_handler",
    "make_sentence_window_handler",
    "semantic_chunks",
    "sentence_window_chunks",
    "split_sentences",
]

#: Sentence end: terminal punctuation, optional quote/bracket, then whitespace.
#: Guarded against the common abbreviations that would otherwise split a sentence
#: in half — a full list is a losing game, these are the ones that actually occur.
#:
#: **The lookbehinds must include the period.** They are evaluated at the
#: position *after* the terminal punctuation, so `(?<!\bDr)` there looks back at
#: `r.` and never matches — the first version of this silently guarded nothing.
_ABBREV = (
    r"(?<!\bMr\.)(?<!\bMrs\.)(?<!\bDr\.)(?<!\bSt\.)"
    r"(?<!\bvs\.)(?<!\be\.g\.)(?<!\bi\.e\.)(?<!\betc\.)"
)
_SENTENCE_END = re.compile(rf"(?<=[.!?]){_ABBREV}[\"')\]]*\s+")


def split_sentences(text: str) -> list[str]:
    """Split *text* into sentences, keeping their trailing punctuation."""
    if not text or not text.strip():
        return []
    parts = [p.strip() for p in _SENTENCE_END.split(text)]
    return [p for p in parts if p]


def sentence_window_chunks(
    text: str, window: int = 3, stride: int = 1
) -> list[str]:
    """Overlapping windows of *window* sentences, advancing by *stride*.

    The point is the overlap. A fact stated in one sentence and qualified in the
    next survives in at least one chunk with both halves intact, where a
    non-overlapping split has a fifty-fifty chance of separating them.

    ``stride=1`` gives maximum redundancy and the largest index; ``stride=window``
    is a plain non-overlapping split.
    """
    sentences = split_sentences(text)
    if not sentences:
        return []
    window = max(1, window)
    stride = max(1, stride)

    out: list[str] = []
    for start in range(0, len(sentences), stride):
        chunk = " ".join(sentences[start : start + window])
        if chunk:
            out.append(chunk)
        if start + window >= len(sentences):
            break
    return out


def semantic_chunks(
    text: str,
    embed: Callable[[list[str]], list[list[float]]],
    *,
    threshold_percentile: float = 0.75,
    min_sentences: int = 2,
    max_sentences: int = 12,
) -> list[str]:
    """Split where the subject changes, measured by embedding distance.

    Each adjacent pair of sentences is compared; the pairs that are least similar
    are the seams. `threshold_percentile` decides how many seams to cut at — 0.75
    breaks at the quarter of positions where the text shifts most.

    `min_sentences` and `max_sentences` bound the result, because a purely
    distance-driven split produces one-sentence chunks in dialogue and
    thousand-word chunks in a uniform passage.
    """
    sentences = split_sentences(text)
    if len(sentences) <= min_sentences:
        return [" ".join(sentences)] if sentences else []

    vectors = embed(sentences)
    if len(vectors) != len(sentences):
        # An embedder that returned the wrong shape is not one to guess with.
        return sentence_window_chunks(text, window=max_sentences, stride=max_sentences)

    distances = [
        1.0 - _cos(vectors[i], vectors[i + 1]) for i in range(len(sentences) - 1)
    ]
    cut_after = _seam_positions(distances, threshold_percentile)

    chunks: list[str] = []
    current: list[str] = []
    for i, sentence in enumerate(sentences):
        current.append(sentence)
        at_seam = i in cut_after and len(current) >= min_sentences
        if at_seam or len(current) >= max_sentences:
            chunks.append(" ".join(current))
            current = []
    if current:
        # A short tail belongs with the chunk before it rather than alone.
        if chunks and len(current) < min_sentences:
            chunks[-1] = chunks[-1] + " " + " ".join(current)
        else:
            chunks.append(" ".join(current))
    return chunks


def _seam_positions(distances: Sequence[float], percentile: float) -> set[int]:
    """Indices after which to cut — the positions with the largest distances."""
    if not distances:
        return set()
    ordered = sorted(distances)
    idx = min(len(ordered) - 1, max(0, int(len(ordered) * percentile)))
    cutoff = ordered[idx]
    return {i for i, d in enumerate(distances) if d >= cutoff and d > 0.0}


def _cos(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


# ── chunker integration ─────────────────────────────────────────────────────


def make_sentence_window_handler(window: int = 3, stride: int = 1):
    """A handler for `Chunker.register_custom`."""

    def handler(text: str, file_path: str | None = None) -> list[str]:
        return sentence_window_chunks(text, window=window, stride=stride)

    return handler


def make_semantic_handler(embedder, **kw):
    """A handler for `Chunker.register_custom`, backed by *embedder*."""

    def handler(text: str, file_path: str | None = None) -> list[str]:
        return semantic_chunks(text, embedder.embed, **kw)

    return handler
