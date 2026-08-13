"""Spans survive from the chunker to the citation.

`ContextBuilder` emitted a `Source: …` header and nothing machine-readable, so an
answer could name where it came from but nothing could point *into* the document.
"""

from __future__ import annotations

from neurosurfer.rag.context_builder import ContextBuilder
from neurosurfer.rag.spans import locate_chunks
from neurosurfer.vectorstores import Doc


def _doc(id_, text, **meta):
    return Doc(id=id_, text=text, embedding=[1.0], metadata=meta)


class TestCitations:
    def test_the_context_text_is_unchanged_by_asking_for_citations(self):
        """Citations sit alongside the prompt, they do not alter it."""
        docs = [_doc("a", "first"), _doc("b", "second")]
        cb = ContextBuilder()

        text, _ = cb.build_with_citations(docs)
        assert text == cb.build(docs)

    def test_spans_come_through_from_metadata(self):
        docs = [_doc("a", "chunk", source="guide.md", char_start=100, char_end=140)]

        _, [cite] = ContextBuilder().build_with_citations(docs)

        assert (cite.char_start, cite.char_end) == (100, 140)
        assert cite.locator() == "guide.md:100-140"

    def test_a_chunk_with_no_span_still_cites_its_source(self):
        """A strategy that rewrites rather than slices yields no offsets, and a
        source with no span is better than a wrong one."""
        docs = [_doc("a", "chunk", source="guide.md")]

        _, [cite] = ContextBuilder().build_with_citations(docs)

        assert cite.char_start is None
        assert cite.locator() == "guide.md"

    def test_scores_ride_along_when_given(self):
        docs = [_doc("a", "one"), _doc("b", "two")]

        _, cites = ContextBuilder().build_with_citations(docs, scores=[0.9, 0.4])

        assert [c.score for c in cites] == [0.9, 0.4]

    def test_citations_line_up_with_the_pieces_actually_rendered(self):
        """A doc that renders to nothing is dropped from the context, so it must
        be dropped from the citations too or every span after it is wrong."""
        docs = [_doc("a", "kept"), _doc("blank", "   \n  "), _doc("c", "also kept")]

        text, cites = ContextBuilder().build_with_citations(docs)

        assert [c.doc_id for c in cites] == ["a", "c"]
        assert text.count("---") == 1


class TestLocateChunks:
    """The real function the ingestor uses, not a re-implementation of it."""

    def test_chunks_are_located_in_their_source(self):
        text = "alpha beta gamma delta epsilon zeta"
        spans = locate_chunks(text, ["alpha beta", "gamma delta", "epsilon zeta"])

        assert spans == [(0, 10), (11, 22), (23, 35)]
        for span in spans:
            assert text[span[0] : span[1]]

    def test_overlapping_chunks_do_not_skip_their_own_match(self):
        """A forward-only scan starts the next search *after* a match that
        overlaps the previous chunk. The first attempt here half-stepped the
        cursor, and this case still failed — hence the search-from-zero fallback.
        """
        text = "one two three four"
        spans = locate_chunks(text, ["one two three", "two three four"])

        assert spans == [(0, 13), (4, 18)]

    def test_a_chunk_that_was_rewritten_yields_no_span(self):
        """Better than a wrong one — a citation must point where it says."""
        spans = locate_chunks("the original text", ["SUMMARISED VERSION"])

        assert spans == [None]

    def test_repeated_text_takes_the_first_position_it_can_find(self):
        text = "repeat. filler. repeat."
        assert locate_chunks(text, ["repeat."]) == [(0, 7)]

    def test_an_empty_chunk_is_not_located(self):
        assert locate_chunks("anything", [""]) == [None]

    def test_no_chunks_is_empty(self):
        assert locate_chunks("anything", []) == []
