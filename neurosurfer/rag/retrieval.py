"""Dense, lexical, fused, diversified — the stages between a query and a context.

Retrieval here was one shape: embed the query, take the top-k by cosine, stuff
them into the prompt. That fails on exactly the queries people ask a codebase or
a docs corpus — an error code, an identifier, a proper noun — because a dense
model that has never seen `ENAMETOOLONG` embeds it as noise while BM25 matches it
exactly.

Four pieces, each usable alone:

* :class:`LexicalIndex` — BM25 over the stored chunks, reusing
  ``agents/lexical.py`` rather than adding a dependency. That module was written
  so *"web-search result injection and long-term memory retrieval share one
  implementation"*; `rag/` never imported it.
* :func:`fuse` — reciprocal-rank fusion. Combines rankings without needing their
  scores to be comparable, which cosine similarity and a BM25 score are not.
* :func:`mmr` — maximal marginal relevance, so five chunks of one paragraph
  cannot crowd out the answer.
* :class:`Reranker` — an optional stage over the top-N. Usually the largest
  single quality gain, and the one place a per-query model call earns its cost.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..agents.lexical import rank_chunks, tokenize
from ..vectorstores.base import BaseVectorDB, Doc

__all__ = [
    "LexicalIndex",
    "Reranker",
    "Retrieved",
    "fuse",
    "hybrid_search",
    "mmr",
]


@dataclass
class Retrieved:
    """One result, and where its score came from.

    The provenance fields are not decoration: when a hybrid run returns something
    unexpected, the first question is always *"which retriever put that there"*,
    and without this it cannot be answered without re-running both.
    """

    doc: Doc
    score: float
    dense_rank: int | None = None
    lexical_rank: int | None = None
    rerank_score: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.doc.id


# ── lexical ─────────────────────────────────────────────────────────────────


class LexicalIndex:
    """BM25 over a set of documents, held in memory.

    **Honest about its limits.** This is an in-process index, not an inverted
    index on disk: it is built from documents you hand it and it costs memory in
    proportion to the corpus. That is the right trade to tens of thousands of
    chunks and the wrong one at ten million, where the answer is a store with
    real full-text search behind it.

    Built from a store with :meth:`from_store`, which is a full scan — do it once
    and keep it, not per query.
    """

    def __init__(self, docs: Sequence[Doc] | None = None) -> None:
        self._docs: list[Doc] = list(docs or [])

    @classmethod
    def from_store(cls, store: BaseVectorDB) -> LexicalIndex:
        return cls(store.list_all_documents())

    def add(self, docs: Sequence[Doc]) -> None:
        known = {d.id for d in self._docs}
        self._docs.extend(d for d in docs if d.id not in known)

    def __len__(self) -> int:
        return len(self._docs)

    def search(self, query: str, top_k: int = 20) -> list[Retrieved]:
        """The *top_k* documents by BM25, best first.

        Zero-scoring documents are dropped rather than padded in: a lexical
        search that matched nothing should contribute nothing to a fusion, not a
        tail of arbitrary documents ranked by document order.
        """
        if not self._docs or not tokenize(query):
            return []

        texts = [d.text or "" for d in self._docs]
        order = rank_chunks(query, texts)

        out: list[Retrieved] = []
        for rank, idx in enumerate(order[: top_k * 2]):
            if not _shares_a_term(query, texts[idx]):
                continue
            out.append(
                Retrieved(doc=self._docs[idx], score=1.0 / (rank + 1), lexical_rank=rank)
            )
            if len(out) >= top_k:
                break
        return out


def _shares_a_term(query: str, text: str) -> bool:
    """Does *text* contain any query term at all?

    `rank_chunks` returns a total ordering, so without this every document in the
    corpus is a "result" — and in a fusion, a document nothing matched would
    still earn rank points.
    """
    return bool(set(tokenize(query)) & set(tokenize(text)))


# ── fusion ──────────────────────────────────────────────────────────────────


def fuse(*rankings: Sequence[Retrieved], k: int = 60, top_k: int = 10) -> list[Retrieved]:
    """Reciprocal-rank fusion over any number of rankings.

    ``score = Σ 1 / (k + rank)``. RRF rather than a weighted sum of scores
    because a cosine similarity and a BM25 score are not on the same scale and
    normalising them is a per-corpus guess. Rank is the thing both agree on.

    *k* dampens the top ranks — the usual 60 means the difference between rank 1
    and rank 2 is small, so a document both retrievers liked outranks one that
    either loved alone. That is the behaviour worth having.
    """
    scores: dict[str, float] = {}
    merged: dict[str, Retrieved] = {}

    for ranking in rankings:
        for rank, item in enumerate(ranking):
            scores[item.id] = scores.get(item.id, 0.0) + 1.0 / (k + rank + 1)
            if item.id in merged:
                # Keep whichever provenance each retriever supplied.
                kept = merged[item.id]
                kept.dense_rank = kept.dense_rank if kept.dense_rank is not None else item.dense_rank
                kept.lexical_rank = (
                    kept.lexical_rank if kept.lexical_rank is not None else item.lexical_rank
                )
            else:
                merged[item.id] = Retrieved(
                    doc=item.doc,
                    score=0.0,
                    dense_rank=item.dense_rank,
                    lexical_rank=item.lexical_rank,
                )

    for doc_id, score in scores.items():
        merged[doc_id].score = score

    ranked = sorted(merged.values(), key=lambda r: -r.score)
    return ranked[:top_k]


# ── diversity ───────────────────────────────────────────────────────────────


def mmr(
    query_embedding: Sequence[float],
    candidates: Sequence[Retrieved],
    top_k: int = 5,
    lambda_: float = 0.5,
) -> list[Retrieved]:
    """Maximal marginal relevance: relevance traded against redundancy.

    Plain top-k happily returns five chunks of one paragraph, which fills the
    context window with one fact. Each pick here maximises
    ``λ·sim(query, doc) − (1−λ)·max sim(doc, already picked)``.

    ``λ=1`` is plain relevance; ``λ=0`` is pure diversity. Candidates without an
    embedding are ranked on relevance alone rather than dropped — a store that
    does not return vectors should degrade, not disappear.
    """
    pool = list(candidates)
    if top_k <= 0 or not pool:
        return []
    if lambda_ >= 1.0:
        return pool[:top_k]

    selected: list[Retrieved] = []
    while pool and len(selected) < top_k:
        best_i, best_score = 0, -math.inf
        for i, cand in enumerate(pool):
            relevance = _cos(query_embedding, cand.doc.embedding)
            redundancy = max(
                (_cos(cand.doc.embedding, s.doc.embedding) for s in selected),
                default=0.0,
            )
            score = lambda_ * relevance - (1.0 - lambda_) * redundancy
            if score > best_score:
                best_i, best_score = i, score
        selected.append(pool.pop(best_i))
    return selected


def _cos(a: Sequence[float] | None, b: Sequence[float] | None) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


# ── reranking ───────────────────────────────────────────────────────────────


@runtime_checkable
class Reranker(Protocol):
    """Rescores `(query, document)` pairs directly.

    A bi-encoder embeds the query and the document apart and compares vectors; a
    cross-encoder reads both together and can weigh how they relate. That is why
    reranking a top-50 usually beats improving the retrieval that produced it.
    """

    def rerank(
        self, query: str, results: Sequence[Retrieved], top_k: int = 5
    ) -> list[Retrieved]: ...


class CrossEncoderReranker:
    """A local `sentence-transformers` cross-encoder.

    Optional in every sense: the dependency, and the stage. `available()` is
    checked before use so a missing model degrades to the retrieval order rather
    than failing the query.
    """

    id = "cross-encoder"
    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.model = model
        self._encoder = None

    def available(self) -> bool:
        try:
            self._load()
        except Exception:  # noqa: BLE001
            return False
        return True

    def _load(self):
        if self._encoder is None:
            from sentence_transformers import CrossEncoder

            self._encoder = CrossEncoder(self.model)
        return self._encoder

    def rerank(
        self, query: str, results: Sequence[Retrieved], top_k: int = 5
    ) -> list[Retrieved]:
        items = list(results)
        if not items:
            return []
        encoder = self._load()
        scores = encoder.predict([(query, r.doc.text or "") for r in items])
        for item, score in zip(items, scores, strict=False):
            item.rerank_score = float(score)
        items.sort(key=lambda r: -(r.rerank_score or 0.0))
        return items[:top_k]


# ── the composed search ─────────────────────────────────────────────────────


def hybrid_search(
    query: str,
    query_embedding: Sequence[float],
    store: BaseVectorDB,
    *,
    lexical: LexicalIndex | None = None,
    top_k: int = 5,
    fetch_k: int = 20,
    metadata_filter: dict[str, Any] | None = None,
    reranker: Reranker | None = None,
    mmr_lambda: float | None = None,
) -> list[Retrieved]:
    """Dense + lexical, fused, then optionally reranked and diversified.

    The order is deliberate and is the one that pays: **fuse, then rerank, then
    diversify.** Reranking before fusion wastes the expensive stage on documents
    fusion would have dropped; diversifying before reranking lets the reranker
    reintroduce the redundancy that was just removed.

    ``fetch_k`` is how wide each retriever reaches before fusion — it should be
    several times ``top_k``, because the point of two retrievers is that they
    disagree about what belongs in the top few.
    """
    dense_hits = [
        Retrieved(doc=doc, score=score, dense_rank=rank)
        for rank, (doc, score) in enumerate(
            store.similarity_search(
                list(query_embedding), top_k=fetch_k, metadata_filter=metadata_filter
            )
        )
    ]

    # **A later stage needs a pool to choose from.** Reranking and MMR both
    # *select*, so trimming to `top_k` before them leaves them nothing to do —
    # MMR handed exactly `top_k` candidates returns all of them, redundancy
    # intact, and looks like a diversity setting that does nothing.
    selects_later = reranker is not None or mmr_lambda is not None
    pool_k = max(top_k, fetch_k) if selects_later else top_k

    if lexical is None or len(lexical) == 0:
        fused = dense_hits[:pool_k]
    else:
        lex_hits = lexical.search(query, top_k=fetch_k)
        if metadata_filter:
            # The lexical index has no query language of its own; applying the
            # same canonical filter keeps the two halves answering one question.
            keep = {d.id for d in store.list_all_documents(metadata_filter)}
            lex_hits = [h for h in lex_hits if h.id in keep]
        fused = fuse(dense_hits, lex_hits, top_k=pool_k)

    if reranker is not None:
        # Keep the pool wide if MMR still has to choose from it.
        fused = reranker.rerank(
            query, fused, top_k=pool_k if mmr_lambda is not None else top_k
        )

    if mmr_lambda is not None:
        return mmr(query_embedding, fused, top_k=top_k, lambda_=mmr_lambda)
    return fused[:top_k]
