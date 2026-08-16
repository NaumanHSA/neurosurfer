"""A collection remembers which model wrote its vectors.

Nothing recorded this, so pointing a differently-embedded query at an existing
store returned confident nonsense: the widths match, the arithmetic succeeds, the
neighbours are simply meaningless. Three defaults were chosen independently in
three files, which is how you end up with two models over one collection without
doing anything unusual.
"""

from __future__ import annotations

import pytest

from neurosurfer.vectorstores import Doc, InMemoryVectorStore

from .conformance import EAST


def test_an_empty_collection_adopts_the_first_model_it_sees():
    """Never block a first ingest."""
    store = InMemoryVectorStore()
    store.check_embedding_identity("e5-small", 384)

    assert store.embedding_identity() == ("e5-small", 384)


def test_the_same_model_passes():
    store = InMemoryVectorStore()
    store.set_embedding_identity("e5-small", 2)
    store.add_documents([Doc(id="a", text="x", embedding=EAST)])

    store.check_embedding_identity("e5-small", 2)  # no raise


def test_a_different_model_is_refused_with_both_names():
    store = InMemoryVectorStore()
    store.set_embedding_identity("e5-small", 2)
    store.add_documents([Doc(id="a", text="x", embedding=EAST)])

    with pytest.raises(ValueError, match="'e5-small'.*'nomic-embed'"):
        store.check_embedding_identity("nomic-embed", 2)


def test_a_different_width_of_the_same_name_is_refused():
    store = InMemoryVectorStore()
    store.set_embedding_identity("m", 2)
    store.add_documents([Doc(id="a", text="x", embedding=EAST)])

    with pytest.raises(ValueError, match="768-wide"):
        store.check_embedding_identity("m", 768)


def test_an_unlabelled_collection_adopts_rather_than_refusing():
    """A store written before this existed must keep working — an upgrade that
    refuses every existing collection is worse than the bug it fixes."""
    store = InMemoryVectorStore()
    store.add_documents([Doc(id="a", text="x", embedding=EAST)])
    assert store.embedding_identity() == (None, None)

    store.check_embedding_identity("whatever", 2)
    assert store.embedding_identity() == ("whatever", 2)


def test_chroma_persists_it_across_reopening(tmp_path):
    """The case that matters: querying an existing store months later."""
    pytest.importorskip("chromadb")
    from neurosurfer.vectorstores import ChromaVectorStore

    store = ChromaVectorStore("identity-probe", persist_directory=str(tmp_path), clear_collection=True)
    store.set_embedding_identity("e5-small", 384)
    store.add_documents([Doc(id="a", text="x", embedding=EAST)])

    reopened = ChromaVectorStore("identity-probe", persist_directory=str(tmp_path))
    assert reopened.embedding_identity() == ("e5-small", 384)

    with pytest.raises(ValueError, match="not comparable"):
        reopened.check_embedding_identity("nomic-embed", 768)
