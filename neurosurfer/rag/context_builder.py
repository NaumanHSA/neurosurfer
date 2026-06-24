from __future__ import annotations
from typing import Callable, List
from neurosurfer.vectorstores.base import Doc


class ContextBuilder:
    def __init__(
        self,
        *,
        include_metadata_in_context: bool = True,
        context_separator: str = "\n\n---\n\n",
        context_item_header_fmt: str = "Source: {source}",
        make_source: Callable[[Doc], str] | None = None,
    ):
        self.include_metadata_in_context = include_metadata_in_context
        self.context_separator = context_separator
        self.context_item_header_fmt = context_item_header_fmt
        self.make_source = make_source or self._default_source

    def build(self, docs: List[Doc]) -> str:
        parts: List[str] = []
        for d in docs:
            piece = d.text or ""
            if self.include_metadata_in_context:
                source = self.make_source(d)
                if source:
                    piece = f"{self.context_item_header_fmt.format(source=source)}\n{piece}"
            piece = piece.strip()
            if piece:
                parts.append(piece)
        return self.context_separator.join(parts)

    @staticmethod
    def _default_source(d: Doc) -> str:
        md = d.metadata or {}
        return md.get("filename") or md.get("source") or md.get("doc_id") or d.id or ""
