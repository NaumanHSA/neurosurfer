from __future__ import annotations

import re
from collections.abc import Callable

from neurosurfer.vectorstores.base import Doc


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
        parts: list[str] = []
        for d in docs:
            piece = d.text or ""
            if self.clean_chunks:
                piece = self._clean_chunk(piece)
            if self.include_metadata_in_context:
                source = self.make_source(d)
                if source:
                    piece = f"{self.context_item_header_fmt.format(source=source)}\n{piece}"
            piece = piece.strip()
            if piece:
                parts.append(piece)
        return self.context_separator.join(parts)

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
