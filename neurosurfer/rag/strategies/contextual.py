"""Contextual retrieval — telling a chunk what document it came from.

The failure this fixes is easy to see and easy to miss. A chunk reading

    It returns None on failure, so callers can fall back to lexical search.

is unfindable by a query about `get_embedder`, because the chunk never says what
"it" is. Splitting a document destroys exactly the context that made each part
searchable, and no amount of better ranking recovers a term that is not there.

The fix is to prepend one line of document-level context *before embedding*:

    From "neurosurfer/embeddings — resolving a backend spec":
    It returns None on failure, so callers can fall back to lexical search.

Cheap, large, and well attested. The summary is produced once per document, not
once per chunk, so the cost is a single call per file rather than per fragment.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

__all__ = ["ContextualEnricher", "llm_summariser"]

DEFAULT_TEMPLATE = 'From "{context}":\n{chunk}'

SUMMARY_PROMPT = (
    "Write one short line — under 20 words — describing what this document is "
    "about, so that a fragment of it can be identified out of context. Name the "
    "subject explicitly. Reply with the line only, no preamble.\n\n"
    "Document:\n{document}"
)


def llm_summariser(provider, *, max_chars: int = 6000) -> Callable[[str, str], str]:
    """A summariser backed by an LLM, for :class:`ContextualEnricher`.

    Only the head of a document is sent: a one-line "what is this" needs the
    opening far more than the tail, and sending a whole book to caption it is how
    ingestion becomes more expensive than it is worth.
    """
    from neurosurfer.llm.types import GenerationConfig, Message

    def summarise(text: str, source_id: str) -> str:
        from neurosurfer.rag.agent import _run_async

        prompt = SUMMARY_PROMPT.format(document=text[:max_chars])
        response = _run_async(
            provider.complete(
                messages=[Message.user_text(prompt)],
                system="You label documents so their fragments stay identifiable.",
                tools=[],
                config=GenerationConfig(max_tokens=100, temperature=0.0, stream=False),
            )
        )
        line = (response.text() or "").strip().splitlines()
        return line[0].strip() if line else source_id

    return summarise


class ContextualEnricher:
    """Prefixes each chunk with a line describing the document it came from.

    ``summarise(text, source_id) -> str`` produces that line. Pass
    :func:`llm_summariser` for a model-written one, or any callable — the
    filename alone is already better than nothing, and free.
    """

    def __init__(
        self,
        summarise: Callable[[str, str], str] | None = None,
        *,
        template: str = DEFAULT_TEMPLATE,
    ) -> None:
        self.summarise = summarise or (lambda _text, source_id: source_id)
        self.template = template
        self._cache: dict[str, str] = {}

    def context_for(self, document: str, source_id: str) -> str:
        """The context line for a document, computed once per source.

        Cached because a document produces many chunks and the summary is a
        property of the document — recomputing it per chunk would multiply the
        cost of ingestion by the chunk count, which is the whole objection to
        this technique when it is implemented carelessly.
        """
        if source_id not in self._cache:
            try:
                self._cache[source_id] = self.summarise(document, source_id) or source_id
            except Exception:  # noqa: BLE001
                # A summariser that fails must not fail the ingest; the filename
                # is a worse context line than a model's, and far better than
                # losing the document.
                self._cache[source_id] = source_id
        return self._cache[source_id]

    def enrich(self, chunks: Sequence[str], document: str, source_id: str) -> list[str]:
        """Every chunk, prefixed with the document's context line."""
        context = self.context_for(document, source_id)
        return [self.template.format(context=context, chunk=c) for c in chunks]

    @staticmethod
    def strip(chunk: str, template: str = DEFAULT_TEMPLATE) -> str:
        """The original chunk text, without the prefix.

        The prefix exists to be *embedded*, not necessarily to be read: a caller
        rendering a citation usually wants what the document actually said.
        """
        marker = template.split("{chunk}")[0].split("{context}")[-1]
        if marker and marker in chunk:
            return chunk.split(marker, 1)[1]
        return chunk
