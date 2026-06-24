"""Local-first lexical ranking: tokenize, BM25-rank, and budget-select chunks.

Extracted from ``tools/web_search.py`` so both web-search result injection and
long-term memory retrieval share one implementation. Pure-python and dependency
light: BM25 via the optional ``rank-bm25`` package, degrading to a term-overlap
score when it is absent — never raising, so a ranking call can't break a run.
"""

from __future__ import annotations

import re

from ..llm.tokens import estimate_text_tokens

_TOKEN_RE = re.compile(r"[a-z0-9]+")

DEFAULT_BUDGET_TOKENS = 3000


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def rank_chunks(query: str, chunks: list[str]) -> list[int]:
    """Return chunk indices ordered by relevance to ``query`` (best first).

    Uses BM25 via ``rank-bm25`` when available; otherwise a pure-python
    term-overlap (frequency) score. Ties keep original document order (stable).
    """
    if not chunks:
        return []
    q_terms = tokenize(query)
    if not q_terms:
        return list(range(len(chunks)))

    tokenized = [tokenize(c) for c in chunks]
    scores: list[float]
    try:
        from rank_bm25 import BM25Okapi  # type: ignore

        # BM25 needs non-empty docs; substitute a placeholder token for empties.
        bm25 = BM25Okapi([t or ["\x00"] for t in tokenized])
        scores = list(bm25.get_scores(q_terms))
    except Exception:  # noqa: BLE001 - fallback to term-overlap
        q_set = set(q_terms)
        scores = [sum(tok in q_set for tok in toks) / (len(toks) or 1) for toks in tokenized]

    return sorted(range(len(chunks)), key=lambda i: (-scores[i], i))


def select_within_budget(
    chunks: list[str], ranked: list[int], budget_tokens: int = DEFAULT_BUDGET_TOKENS
) -> list[int]:
    """Greedily take top-ranked chunks until the token budget is reached.

    Returns the selected indices in original document order so the injected
    excerpt reads coherently. Always includes at least the single best chunk.
    """
    selected: list[int] = []
    used = 0
    for idx in ranked:
        ctok = estimate_text_tokens(chunks[idx])
        if selected and used + ctok > budget_tokens:
            continue
        selected.append(idx)
        used += ctok
        if used >= budget_tokens:
            break
    return sorted(selected)
