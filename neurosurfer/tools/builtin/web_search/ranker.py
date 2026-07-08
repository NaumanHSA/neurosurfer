"""Paragraph-boundary chunker and BM25 ranking helpers."""

from __future__ import annotations

import re

# Re-exported from agents.lexical (shared with memory retrieval).
# Existing imports like `from ...tools.builtin.web_search import rank_chunks` keep working.
from ....agents.lexical import rank_chunks, select_within_budget  # noqa: F401
from ....llm.tokens import estimate_text_tokens
from . import config as _config

__all__ = ["chunk_text", "rank_chunks", "select_within_budget"]


def chunk_text(text: str, chunk_tokens: int | None = None) -> list[str]:
    """Split ``text`` into ~``chunk_tokens``-sized chunks on paragraph boundaries."""
    if chunk_tokens is None:
        chunk_tokens = _config.CHUNK_TOKENS
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf: list[str] = []
    buf_tokens = 0
    for para in paras:
        ptok = estimate_text_tokens(para)
        if ptok >= chunk_tokens and not buf:
            # A single oversized paragraph becomes its own chunk.
            chunks.append(para)
            continue
        if buf_tokens + ptok > chunk_tokens and buf:
            chunks.append("\n\n".join(buf))
            buf, buf_tokens = [], 0
        buf.append(para)
        buf_tokens += ptok
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks
