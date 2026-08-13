"""The retrieval shapes beyond classic — each tested on the failure it fixes.

A strategy is worth having when it solves a case the default cannot. So each
group here shows the default failing first, and then the strategy handling it.
"""

from __future__ import annotations

import pytest

from neurosurfer.rag.strategies import (
    ContextualEnricher,
    ParentDocumentRetriever,
    hyde_query,
    multi_query,
    semantic_chunks,
    sentence_window_chunks,
)
from neurosurfer.rag.strategies.chunking import split_sentences
from neurosurfer.rag.strategies.parent import PARENT_ID_KEY, split_parent_child
from neurosurfer.vectorstores import Doc

from .corpus import HashingEmbedder

# ── sentence splitting and windows ──────────────────────────────────────────


class TestSentenceSplitting:
    def test_it_splits_on_terminal_punctuation(self):
        got = split_sentences("One. Two! Three? Four.")
        assert got == ["One.", "Two!", "Three?", "Four."]

    def test_common_abbreviations_do_not_end_a_sentence(self):
        """A full abbreviation list is a losing game; these are the ones that
        actually occur in prose and each would otherwise split a sentence."""
        got = split_sentences("See Dr. Smith about it. Then e.g. this one.")
        assert len(got) == 2

    def test_empty_text_is_no_sentences(self):
        assert split_sentences("   ") == []


class TestSentenceWindow:
    def test_windows_overlap(self):
        """The overlap is the point: a fact and its qualifier survive together."""
        text = "A one. B two. C three. D four."
        got = sentence_window_chunks(text, window=2, stride=1)

        assert got[0] == "A one. B two."
        assert got[1] == "B two. C three."

    def test_stride_equal_to_window_is_a_plain_split(self):
        text = "A one. B two. C three. D four."
        got = sentence_window_chunks(text, window=2, stride=2)

        assert got == ["A one. B two.", "C three. D four."]

    def test_it_terminates_on_the_last_window(self):
        got = sentence_window_chunks("A. B. C.", window=5, stride=1)
        assert got == ["A. B. C."]

    def test_empty_in_empty_out(self):
        assert sentence_window_chunks("") == []


# ── semantic chunking ───────────────────────────────────────────────────────


class TestSemanticChunking:
    """Tested with vectors under the test's control.

    The first version of this used the fixture's hashing embedder and asserted a
    seam between two topics. It failed, and correctly: a hash bag-of-words
    measures *vocabulary overlap*, and in short sentences two "Compilers…" lines
    sharing one word can look less alike than a cats→compilers pair that happens
    to collide. That is a fact about the fixture embedder, not about this
    algorithm — which is deterministic given vectors, and is what these check.
    """

    @staticmethod
    def _topic_embed(sentences):
        """One dimension per topic, from a word each sentence declares."""
        return [
            [1.0, 0.0] if s.startswith("A") else [0.0, 1.0] for s in sentences
        ]

    def test_it_cuts_where_the_subject_changes(self):
        text = "A one. A two. A three. B four. B five. B six."

        chunks = semantic_chunks(
            text, self._topic_embed, min_sentences=2, max_sentences=6
        )

        assert len(chunks) == 2
        for chunk in chunks:
            assert not ("A " in chunk and "B " in chunk), chunk

    def test_it_cuts_at_the_largest_distance_not_a_fixed_size(self):
        """The seam is chosen by distance; a fixed split would land mid-topic."""
        text = "A one. A two. A three. A four. B five."

        chunks = semantic_chunks(
            text, self._topic_embed, min_sentences=1, max_sentences=10
        )

        assert chunks[-1].strip() == "B five."

    def test_short_text_is_one_chunk(self):
        embedder = HashingEmbedder()
        assert semantic_chunks("Only this.", embedder.embed) == ["Only this."]

    def test_max_sentences_bounds_a_uniform_passage(self):
        """Distance alone gives thousand-word chunks where nothing changes."""
        embedder = HashingEmbedder()
        text = " ".join(f"The same idea number {i}." for i in range(20))

        chunks = semantic_chunks(text, embedder.embed, max_sentences=4)

        assert all(len(split_sentences(c)) <= 8 for c in chunks)

    def test_a_misbehaving_embedder_falls_back_rather_than_guessing(self):
        chunks = semantic_chunks("One. Two. Three. Four.", lambda texts: [])
        assert chunks  # a result, not a crash


# ── contextual retrieval ────────────────────────────────────────────────────


class TestContextualEnricher:
    DOC = "The get_embedder function resolves a spec string into a backend."
    CHUNK = "It returns None on failure, so callers can fall back to lexical search."

    def test_a_chunk_gains_the_subject_it_never_names(self):
        """The failure, exactly: the chunk says 'it' and is therefore unfindable
        by a query naming the thing."""
        assert "get_embedder" not in self.CHUNK

        enricher = ContextualEnricher(lambda text, sid: "get_embedder — resolving a spec")
        [enriched] = enricher.enrich([self.CHUNK], self.DOC, "embeddings.py")

        assert "get_embedder" in enriched
        assert self.CHUNK in enriched

    def test_it_makes_the_chunk_retrievable(self, ):
        """Not just present — actually found. Measured, not asserted by eye."""
        from neurosurfer.vectorstores import InMemoryVectorStore

        embedder = HashingEmbedder()
        enricher = ContextualEnricher(lambda text, sid: "get_embedder resolves a spec string")

        plain = InMemoryVectorStore()
        plain.add_documents(
            [Doc(id="c", text=self.CHUNK, embedding=embedder.embed([self.CHUNK])[0])]
        )
        [enriched_text] = enricher.enrich([self.CHUNK], self.DOC, "embeddings.py")
        enriched = InMemoryVectorStore()
        enriched.add_documents(
            [Doc(id="c", text=enriched_text, embedding=embedder.embed([enriched_text])[0])]
        )

        query = embedder.embed(["get_embedder"])[0]
        assert plain.similarity_search(query, top_k=1)[0][1] == pytest.approx(0.0)
        assert enriched.similarity_search(query, top_k=1)[0][1] > 0.0

    def test_the_summary_is_computed_once_per_document(self):
        """A document makes many chunks; summarising per chunk is the objection
        to this technique when it is implemented carelessly."""
        calls = []

        def summarise(text, source_id):
            calls.append(source_id)
            return "ctx"

        enricher = ContextualEnricher(summarise)
        enricher.enrich(["a", "b", "c"], self.DOC, "one.py")
        enricher.enrich(["d"], self.DOC, "one.py")

        assert calls == ["one.py"]

    def test_a_failing_summariser_does_not_fail_the_ingest(self):
        def boom(text, source_id):
            raise RuntimeError("model down")

        [got] = ContextualEnricher(boom).enrich(["chunk"], self.DOC, "src.py")

        assert "src.py" in got and "chunk" in got

    def test_the_prefix_can_be_stripped_for_display(self):
        enricher = ContextualEnricher(lambda t, s: "some context")
        [enriched] = enricher.enrich([self.CHUNK], self.DOC, "x")

        assert ContextualEnricher.strip(enriched) == self.CHUNK


# ── parent-document retrieval ───────────────────────────────────────────────


class TestParentDocument:
    def test_children_expand_to_their_parent(self):
        parents = {"sec1": "The whole section, with all its context intact."}
        child = Doc(id="sec1#0", text="a fragment", metadata={PARENT_ID_KEY: "sec1"})

        [got] = ParentDocumentRetriever(parents).expand([child])

        assert got.id == "sec1"
        assert got.text == parents["sec1"]
        assert got.metadata["retrieved_via"] == "sec1#0"

    def test_several_children_of_one_parent_return_it_once(self):
        """Well-chunked sections retrieve together; returning the parent three
        times would fill the window with one passage repeated."""
        parents = {"sec1": "section text"}
        children = [
            Doc(id=f"sec1#{i}", text="frag", metadata={PARENT_ID_KEY: "sec1"})
            for i in range(3)
        ]

        assert len(ParentDocumentRetriever(parents).expand(children)) == 1

    def test_order_follows_the_child_ranking(self):
        parents = {"a": "A text", "b": "B text"}
        children = [
            Doc(id="b#0", text="", metadata={PARENT_ID_KEY: "b"}),
            Doc(id="a#0", text="", metadata={PARENT_ID_KEY: "a"}),
        ]

        got = ParentDocumentRetriever(parents).expand(children)
        assert [d.id for d in got] == ["b", "a"]

    def test_a_missing_parent_falls_back_to_the_child(self):
        """A worse answer than the parent, a much better one than dropping it."""
        child = Doc(id="gone#0", text="the fragment", metadata={PARENT_ID_KEY: "gone"})

        [got] = ParentDocumentRetriever({}).expand([child])

        assert got.text == "the fragment"

    def test_a_child_with_no_parent_id_is_skipped(self):
        assert ParentDocumentRetriever({}).expand([Doc(id="x", text="t")]) == []

    def test_split_parent_child_numbers_children_stably(self):
        parents, children = split_parent_child(
            [("doc1", "a b c")], lambda t: t.split()
        )

        assert parents == {"doc1": "a b c"}
        assert [c[0] for c in children] == ["doc1#0", "doc1#1", "doc1#2"]
        assert all(c[1] == "doc1" for c in children)


# ── query strategies ────────────────────────────────────────────────────────


class _Says:
    """A provider returning one fixed completion."""

    model = "fake"

    def __init__(self, text: str = "", raises: bool = False):
        self._text = text
        self._raises = raises

    async def complete(self, messages, system, tools, config):
        if self._raises:
            raise RuntimeError("model down")
        from neurosurfer.llm.types import CanonicalResponse, TextBlock, Usage

        return CanonicalResponse(
            content=[TextBlock(text=self._text)], stop_reason="end_turn", usage=Usage()
        )


class TestMultiQuery:
    def test_the_original_query_comes_first_and_always(self):
        """A rewrite is a guess about what the asker meant; discarding the actual
        question can lose an exact term they typed deliberately."""
        got = multi_query(_Says("install the library\nsetup instructions"), "how to install")

        assert got[0] == "how to install"
        assert "install the library" in got

    def test_numbering_and_bullets_are_stripped(self):
        got = multi_query(_Says("1. first way\n2) second way\n- third way"), "q")

        assert "first way" in got and "second way" in got and "third way" in got

    def test_a_rewrite_identical_to_the_query_is_not_duplicated(self):
        got = multi_query(_Says("how to install"), "how to install")
        assert got == ["how to install"]

    def test_it_is_capped_at_n(self):
        got = multi_query(_Says("\n".join(f"variant {i}" for i in range(10))), "q", n=3)
        assert len(got) <= 4  # the original plus n

    def test_a_failing_model_degrades_to_the_plain_query(self):
        assert multi_query(_Says(raises=True), "q") == ["q"]


class TestHyDE:
    def test_it_returns_the_hypothetical_answer(self):
        answer = "Caching stores the result so the second call is free."
        assert hyde_query(_Says(answer), "how do I make it faster?") == answer

    def test_a_failing_model_degrades_to_the_query(self):
        assert hyde_query(_Says(raises=True), "q") == "q"

    def test_an_empty_generation_degrades_to_the_query(self):
        assert hyde_query(_Says(""), "q") == "q"
