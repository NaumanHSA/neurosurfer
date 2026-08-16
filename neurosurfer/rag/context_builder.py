from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from neurosurfer.vectorstores.base import Doc


@dataclass(frozen=True)
class Citation:
    """Where one piece of the context came from, precisely enough to highlight.

    `build` emitted a `Source: …` header and nothing machine-readable, so an
    answer could name its sources but nothing could point *into* them. The spans
    come from the ingestor, which locates each chunk in its source text; they are
    `None` for a chunk whose strategy rewrote rather than sliced.
    """

    doc_id: str
    source: str
    text: str
    char_start: int | None = None
    char_end: int | None = None
    score: float | None = None

    def locator(self) -> str:
        """A human-readable pointer — `guide.md:1200-1480`, or just the source."""
        if self.char_start is None or self.char_end is None:
            return self.source
        return f"{self.source}:{self.char_start}-{self.char_end}"


class ContextBuilder:
    def __init__(
        self,
        *,
        include_metadata_in_context: bool = True,
        context_separator: str = "\n\n---\n\n",
        context_item_header_fmt: str = "Source: {source}",
        make_source: Callable[[Doc], str] | None = None,
        clean_chunks: bool = True,
    ):
        self.include_metadata_in_context = include_metadata_in_context
        self.context_separator = context_separator
        self.context_item_header_fmt = context_item_header_fmt
        self.make_source = make_source or self._default_source
        self.clean_chunks = clean_chunks

    def build(self, docs: list[Doc]) -> str:
        return self.context_separator.join(p for p, _ in self._pieces(docs))

    def build_with_citations(
        self, docs: list[Doc], scores: list[float] | None = None
    ) -> tuple[str, list[Citation]]:
        """The context, plus where each part of it came from.

        Same text as :meth:`build` — the citations are alongside it, not instead
        of it, so an answer can be rendered with source spans without the prompt
        changing at all.
        """
        pieces = self._pieces(docs)
        by_score = scores or []
        citations = [
            Citation(
                doc_id=d.id,
                source=self.make_source(d),
                text=d.text or "",
                char_start=(d.metadata or {}).get("char_start"),
                char_end=(d.metadata or {}).get("char_end"),
                score=by_score[i] if i < len(by_score) else None,
            )
            for i, (_, d) in enumerate(pieces)
        ]
        return self.context_separator.join(p for p, _ in pieces), citations

    def _pieces(self, docs: list[Doc]) -> list[tuple[str, Doc]]:
        """Rendered text per doc, dropping the ones that render to nothing."""
        out: list[tuple[str, Doc]] = []
        for d in docs:
            piece = d.text or ""
            if self.clean_chunks:
                piece = self._clean_chunk(piece)
            # **Emptiness is decided before the header, not after.** A chunk that
            # cleans away to nothing used to still contribute `Source: …`, which
            # is a citation pointing at no content — and it made the citation
            # list disagree with the context it was supposed to describe.
            if not piece.strip():
                continue
            if self.include_metadata_in_context:
                source = self.make_source(d)
                if source:
                    piece = f"{self.context_item_header_fmt.format(source=source)}\n{piece}"
            out.append((piece.strip(), d))
        return out

    @staticmethod
    def _clean_chunk(text: str) -> str:
        """Drop letter-free lines (bare numbers, dashes — PDF table residue) and
        normalise runs of blank lines to a single paragraph break.
        """
        lines = text.splitlines()
        out: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped and not re.search(r'[A-Za-z]', stripped):
                continue  # pure numbers / punctuation / dashes — skip
            out.append(line)
        # Collapse 3+ consecutive blank lines to 2
        result = re.sub(r'\n{3,}', '\n\n', '\n'.join(out))
        return result.strip()

    @staticmethod
    def _default_source(d: Doc) -> str:
        md = d.metadata or {}
        return md.get("filename") or md.get("source") or md.get("doc_id") or d.id or ""
