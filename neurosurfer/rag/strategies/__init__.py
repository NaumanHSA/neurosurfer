"""Retrieval shapes beyond "embed the chunk, take the top-k".

Each is a strategy behind one seam rather than a fork of the pipeline, and each
answers a different failure of the classic shape:

* :mod:`~neurosurfer.rag.strategies.chunking` — **sentence-window** and
  **semantic** chunking. Fixed-size chunks cut mid-argument; these cut where the
  text changes subject.
* :mod:`~neurosurfer.rag.strategies.contextual` — **contextual retrieval**. A
  chunk saying *"it returns None on failure"* is unfindable because it never
  says what "it" is; prepending a line of document context before embedding
  fixes exactly that.
* :mod:`~neurosurfer.rag.strategies.parent` — **parent-document retrieval**.
  Embed small for precision, return the enclosing section for context.
* :mod:`~neurosurfer.rag.strategies.query` — **multi-query** and **HyDE**. One
  question becomes several retrievals, so a single unlucky phrasing is not the
  whole attempt.
"""

from __future__ import annotations

from .chunking import semantic_chunks, sentence_window_chunks
from .contextual import ContextualEnricher
from .parent import ParentDocumentRetriever
from .query import hyde_query, multi_query

__all__ = [
    "ContextualEnricher",
    "ParentDocumentRetriever",
    "hyde_query",
    "multi_query",
    "semantic_chunks",
    "sentence_window_chunks",
]
