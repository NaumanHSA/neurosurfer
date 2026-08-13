"""Retrieval quality, measured — the phase's own rule applied to itself.

The headline test is `test_hybrid_beats_dense_on_the_corpus`. Every other claim
in this module is a unit check on a piece; that one is the reason the pieces
exist, and it is stated as a measured delta over a fixture corpus rather than as
a conviction in a changelog.
"""

from __future__ import annotations

import pytest

from neurosurfer.rag.evaluation import compare, evaluate
from neurosurfer.rag.retrieval import (
    LexicalIndex,
    Retrieved,
    fuse,
    hybrid_search,
    mmr,
)
from neurosurfer.vectorstores import Doc, InMemoryVectorStore

from .corpus import CASES, HashingEmbedder, build_docs


@pytest.fixture
def embedder():
    return HashingEmbedder()


@pytest.fixture
def store(embedder):
    s = InMemoryVectorStore()
    s.add_documents(build_docs(embedder))
    return s


@pytest.fixture
def lexical(store):
    return LexicalIndex.from_store(store)


def _dense_only(store, embedder):
    def retrieve(query: str, k: int):
        vec = embedder.embed([query])[0]
        return [d.id for d, _ in store.similarity_search(vec, top_k=k)]

    return retrieve


def _hybrid(store, embedder, lexical, **kw):
    def retrieve(query: str, k: int):
        vec = embedder.embed([query])[0]
        return [
            r.id
            for r in hybrid_search(query, vec, store, lexical=lexical, top_k=k, **kw)
        ]

    return retrieve


# ── the measurement this phase exists for ───────────────────────────────────


class TestHybridIsAnImprovement:
    """Measured on the fixture corpus, at the k values that show different things.

        k=1   dense  recall 0.619  mrr 0.714      hybrid  recall 0.905  mrr 1.000
        k=3   dense  recall 0.952  mrr 0.833      hybrid  recall 0.952  mrr 1.000

    **The gain is in ranking, and at small k that is also recall.** Dense search
    on this corpus usually has the right chunk *somewhere* in the top few; what
    it does badly is put it first. Since a context window is a small k, "third
    out of three" and "not found" cost nearly the same.

    The first draft of this test asserted dense could not find `ENAMETOOLONG` at
    all. It can — at rank 3. The harness caught the overclaim, which is the
    argument for writing it before the thing it measures.
    """

    def test_hybrid_beats_dense_at_the_k_a_context_window_uses(
        self, store, embedder, lexical
    ):
        reports = compare(
            CASES,
            {
                "dense": _dense_only(store, embedder),
                "hybrid": _hybrid(store, embedder, lexical),
            },
            k=1,
        )
        print("\n  dense :", reports["dense"])
        print("  hybrid:", reports["hybrid"])

        assert reports["hybrid"].recall > reports["dense"].recall
        assert reports["hybrid"].better_than(reports["dense"])

    def test_hybrid_orders_better_even_where_recall_ties(self, store, embedder, lexical):
        """At k=3 both find it; only one puts it first."""
        reports = compare(
            CASES,
            {
                "dense": _dense_only(store, embedder),
                "hybrid": _hybrid(store, embedder, lexical),
            },
            k=3,
        )

        assert reports["hybrid"].recall == reports["dense"].recall
        assert reports["hybrid"].mrr > reports["dense"].mrr
        assert reports["hybrid"].ndcg > reports["dense"].ndcg

    def test_dense_buries_the_rare_literal(self, store, embedder):
        """The specific weakness, isolated — so the delta above has a cause."""
        got = _dense_only(store, embedder)("ENAMETOOLONG", 5)

        assert got.index("errors:0") > 0, "dense should not rank it first"

    def test_lexical_finds_it_immediately(self, lexical):
        hits = lexical.search("ENAMETOOLONG", top_k=3)

        assert hits[0].id == "errors:0"

    def test_hybrid_promotes_it_to_first(self, store, embedder, lexical):
        got = _hybrid(store, embedder, lexical)("ENAMETOOLONG", 5)

        assert got[0] == "errors:0"


# ── the harness itself ──────────────────────────────────────────────────────


class TestEvaluation:
    def test_a_perfect_retriever_scores_one(self):
        cases = [type(CASES[0])(query="q", relevant_ids={"a"})]
        report = evaluate(cases, lambda q, k: ["a", "b"], k=2)

        assert report.recall == 1.0
        assert report.mrr == 1.0
        assert report.ndcg == 1.0

    def test_a_retriever_that_returns_nothing_scores_zero(self):
        cases = [type(CASES[0])(query="q", relevant_ids={"a"})]
        report = evaluate(cases, lambda q, k: [], k=2)

        assert (report.recall, report.mrr, report.ndcg) == (0.0, 0.0, 0.0)

    def test_mrr_reflects_the_rank_of_the_first_hit(self):
        cases = [type(CASES[0])(query="q", relevant_ids={"a"})]

        second = evaluate(cases, lambda q, k: ["x", "a"], k=3)
        assert second.mrr == pytest.approx(0.5)
        assert second.recall == 1.0, "recall does not care about position; MRR does"

    def test_no_cases_is_an_empty_report_not_a_crash(self):
        assert evaluate([], lambda q, k: ["a"]).cases == 0

    def test_better_than_prefers_recall_over_ordering(self):
        from neurosurfer.rag.evaluation import EvalReport

        recalls_more = EvalReport(k=3, cases=1, recall=0.9, mrr=0.1, ndcg=0.1)
        orders_better = EvalReport(k=3, cases=1, recall=0.5, mrr=1.0, ndcg=1.0)

        assert recalls_more.better_than(orders_better), (
            "a chunk that never arrives cannot be reordered into place"
        )


# ── fusion ──────────────────────────────────────────────────────────────────


class TestFuse:
    def _r(self, doc_id):
        return Retrieved(doc=Doc(id=doc_id, text=doc_id), score=1.0)

    def test_a_document_both_rankings_like_wins(self):
        """The behaviour RRF is chosen for: agreement beats a single strong vote."""
        dense = [self._r("a"), self._r("shared")]
        lex = [self._r("b"), self._r("shared")]

        assert fuse(dense, lex, top_k=1)[0].id == "shared"

    def test_it_merges_rather_than_duplicating(self):
        dense = [self._r("a")]
        lex = [self._r("a")]

        assert len(fuse(dense, lex, top_k=10)) == 1

    def test_provenance_survives_the_merge(self):
        dense = [Retrieved(doc=Doc(id="a", text=""), score=1.0, dense_rank=0)]
        lex = [Retrieved(doc=Doc(id="a", text=""), score=1.0, lexical_rank=3)]

        [merged] = fuse(dense, lex, top_k=1)
        assert merged.dense_rank == 0 and merged.lexical_rank == 3

    def test_one_ranking_is_passed_through_in_order(self):
        got = fuse([self._r("a"), self._r("b"), self._r("c")], top_k=3)
        assert [r.id for r in got] == ["a", "b", "c"]

    def test_no_rankings_is_empty(self):
        assert fuse(top_k=5) == []


# ── diversity ───────────────────────────────────────────────────────────────


class TestMMR:
    def test_it_breaks_up_near_duplicates(self, store, embedder):
        """Plain top-k returns three chunks of one paragraph; the corpus has
        exactly that shape seeded into it."""
        vec = embedder.embed(["prompt caching reuses an identical system prompt"])[0]
        plain = [d.id for d, _ in store.similarity_search(vec, top_k=3)]
        assert sum(i.startswith("dup:") for i in plain) == 3, "the fixture must be redundant"

        candidates = [
            Retrieved(doc=d, score=s) for d, s in store.similarity_search(vec, top_k=8)
        ]
        diverse = [r.id for r in mmr(vec, candidates, top_k=3, lambda_=0.4)]

        assert sum(i.startswith("dup:") for i in diverse) < 3

    def test_lambda_one_is_plain_relevance(self, store, embedder):
        vec = embedder.embed(["prompt caching"])[0]
        candidates = [
            Retrieved(doc=d, score=s) for d, s in store.similarity_search(vec, top_k=5)
        ]

        assert [r.id for r in mmr(vec, candidates, top_k=3, lambda_=1.0)] == [
            r.id for r in candidates[:3]
        ]

    def test_candidates_without_vectors_degrade_rather_than_vanish(self):
        candidates = [Retrieved(doc=Doc(id="a", text="x", embedding=None), score=1.0)]

        assert [r.id for r in mmr([1.0, 0.0], candidates, top_k=1, lambda_=0.5)] == ["a"]

    def test_empty_in_empty_out(self):
        assert mmr([1.0], [], top_k=3) == []


# ── the lexical index ───────────────────────────────────────────────────────


class TestLexicalIndex:
    def test_a_query_matching_nothing_returns_nothing(self, lexical):
        """`rank_chunks` returns a total ordering, so without a term check every
        document in the corpus would be a 'result' and earn fusion points."""
        assert lexical.search("zzzznonexistentquux", top_k=5) == []

    def test_an_empty_index_is_safe(self):
        assert LexicalIndex().search("anything", top_k=5) == []

    def test_add_is_idempotent_on_id(self, store):
        idx = LexicalIndex.from_store(store)
        before = len(idx)
        idx.add(store.list_all_documents())

        assert len(idx) == before


# ── the composed search ─────────────────────────────────────────────────────


class TestHybridSearch:
    def test_it_works_without_a_lexical_index(self, store, embedder):
        """Dense-only must stay a supported configuration, not a broken one."""
        vec = embedder.embed(["how do I install"])[0]
        hits = hybrid_search("how do I install", vec, store, top_k=3)

        assert len(hits) == 3

    def test_metadata_filters_apply_to_both_halves(self, store, embedder, lexical):
        """A filtered hybrid search must not let the lexical half smuggle in rows
        the filter excluded."""
        vec = embedder.embed(["ENAMETOOLONG install"])[0]
        hits = hybrid_search(
            "ENAMETOOLONG install",
            vec,
            store,
            lexical=lexical,
            top_k=5,
            metadata_filter={"topic": "install"},
        )

        assert {r.doc.metadata["topic"] for r in hits} == {"install"}

    def test_mmr_can_be_applied_at_the_end(self, store, embedder, lexical):
        vec = embedder.embed(["prompt caching"])[0]
        hits = hybrid_search(
            "prompt caching", vec, store, lexical=lexical, top_k=3, mmr_lambda=0.3
        )

        assert len(hits) == 3

    def test_mmr_is_given_a_pool_wider_than_top_k(self, store, embedder):
        """A selecting stage handed exactly `top_k` candidates returns all of
        them — a diversity setting that silently does nothing."""
        vec = embedder.embed(["prompt caching identical system prompt"])[0]

        plain = hybrid_search("prompt caching identical system prompt", vec, store, top_k=3)
        diverse = hybrid_search(
            "prompt caching identical system prompt",
            vec,
            store,
            top_k=3,
            mmr_lambda=0.3,
        )

        assert sum(r.id.startswith("dup:") for r in plain) == 3
        assert sum(r.id.startswith("dup:") for r in diverse) < 3

    def test_a_reranker_reorders_the_result(self, store, embedder, lexical):
        class Reverser:
            def rerank(self, query, results, top_k=5):
                out = list(reversed(list(results)))
                for i, r in enumerate(out):
                    r.rerank_score = float(len(out) - i)
                return out[:top_k]

        vec = embedder.embed(["install"])[0]
        plain = hybrid_search("install", vec, store, lexical=lexical, top_k=3)
        reranked = hybrid_search(
            "install", vec, store, lexical=lexical, top_k=3, reranker=Reverser()
        )

        assert [r.id for r in reranked] != [r.id for r in plain]
        assert all(r.rerank_score is not None for r in reranked)
