"""One question, several retrievals.

A query is one phrasing of an information need, and retrieval is sensitive to
phrasing in ways the person asking cannot predict. Both strategies here spend one
model call to stop a single unlucky wording from being the whole attempt.

* :func:`multi_query` — rewrite the question several ways and retrieve each.
  Different vocabulary reaches different chunks; fusing the rankings keeps what
  they agree on.
* :func:`hyde_query` — write a *hypothetical answer* and embed that instead. The
  insight is that a question and its answer often share little vocabulary
  ("how do I make it faster?" vs. a passage about caching), while an invented
  answer and the real one share a great deal. The invention does not need to be
  correct — it only needs to look like the thing being searched for.
"""

from __future__ import annotations

__all__ = ["hyde_query", "multi_query"]

MULTI_QUERY_PROMPT = (
    "Rewrite this search query {n} different ways, so that together they cover "
    "different vocabulary someone might have used to write the answer. Keep each "
    "on one line, no numbering, no preamble.\n\nQuery: {query}"
)

HYDE_PROMPT = (
    "Write a short passage — three sentences at most — that would answer this "
    "question, in the style of documentation. Do not hedge, do not say you are "
    "unsure, and do not mention the question. If you do not know the specifics, "
    "invent plausible ones: this text is used to search with, never shown to "
    "anyone.\n\nQuestion: {query}"
)


def _complete(provider, prompt: str, system: str, max_tokens: int) -> str:
    from neurosurfer.llm.types import GenerationConfig, Message
    from neurosurfer.rag.agent import _run_async

    response = _run_async(
        provider.complete(
            messages=[Message.user_text(prompt)],
            system=system,
            tools=[],
            config=GenerationConfig(
                max_tokens=max_tokens, temperature=0.3, stream=False
            ),
        )
    )
    return (response.text() or "").strip()


def multi_query(provider, query: str, n: int = 3) -> list[str]:
    """*query* plus *n* rewrites of it, original first.

    The original is always included and always first: a rewrite is a guess about
    what the asker meant, and a strategy that discards the actual question in
    favour of three guesses can lose an exact term the asker typed deliberately.

    A failed or empty generation returns just the original — degrading to
    ordinary retrieval, which is a working answer.
    """
    out = [query]
    try:
        text = _complete(
            provider,
            MULTI_QUERY_PROMPT.format(n=n, query=query),
            "You rewrite search queries.",
            200,
        )
    except Exception:  # noqa: BLE001
        return out

    for line in text.splitlines():
        cleaned = line.strip().lstrip("0123456789.-) ").strip()
        if cleaned and cleaned.lower() != query.lower() and cleaned not in out:
            out.append(cleaned)
        if len(out) > n:
            break
    return out


def hyde_query(provider, query: str) -> str:
    """A hypothetical answer to *query*, to embed in its place.

    Returns the original query unchanged if generation fails or comes back empty
    — searching with the real question is always available and always sane.
    """
    try:
        text = _complete(
            provider,
            HYDE_PROMPT.format(query=query),
            "You write short documentation passages.",
            300,
        )
    except Exception:  # noqa: BLE001
        return query
    return text or query
