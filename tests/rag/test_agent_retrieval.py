"""Hybrid retrieval is reachable from `RAGAgent`, not merely importable.

A strategy nothing routes to is a strategy nobody uses. These check the wiring —
that the config flag reaches `hybrid_search`, that the lexical index is built
once rather than per query, and that the defaults are unchanged for a caller who
upgrades without asking for anything.
"""

from __future__ import annotations

import pytest

from neurosurfer.rag.config import RAGAgentConfig
from neurosurfer.rag.retrieval import LexicalIndex
from neurosurfer.vectorstores import InMemoryVectorStore

from .corpus import HashingEmbedder, build_docs


@pytest.fixture
def store():
    s = InMemoryVectorStore()
    s.add_documents(build_docs(HashingEmbedder()))
    return s


def _agent(store, **cfg):
    from neurosurfer.rag.agent import RAGAgent

    return RAGAgent(
        vectorstore=store,
        embedder=HashingEmbedder(),
        config=RAGAgentConfig(top_k=3, **cfg),
    )


class TestWiring:
    def test_hybrid_is_off_by_default(self, store):
        """Building the BM25 index is a full scan and MMR changes which chunks
        come back — neither should start happening because someone upgraded."""
        agent = _agent(store)

        assert agent._hybrid_enabled() is False
        assert agent._lexical_index() is None

    def test_the_config_flag_turns_it_on(self, store):
        agent = _agent(store, hybrid_search=True)

        assert agent._hybrid_enabled() is True
        assert isinstance(agent._lexical_index(), LexicalIndex)

    def test_the_lexical_index_is_built_once_and_kept(self, store):
        agent = _agent(store, hybrid_search=True)

        assert agent._lexical_index() is agent._lexical_index()

    def test_invalidating_rebuilds_it(self, store):
        agent = _agent(store, hybrid_search=True)
        first = agent._lexical_index()

        agent.invalidate_lexical_index()
        assert agent._lexical_index() is not first

    def test_mmr_alone_enables_the_path_without_a_lexical_scan(self, store):
        """Diversity without hybrid should not pay for a BM25 index."""
        agent = _agent(store, mmr_lambda=0.5)

        assert agent._hybrid_enabled() is True
        assert agent._lexical_index() is None

    def test_a_reranker_alone_enables_the_path(self, store):
        from neurosurfer.rag.agent import RAGAgent

        class Noop:
            def rerank(self, query, results, top_k=5):
                return list(results)[:top_k]

        agent = RAGAgent(
            vectorstore=store,
            embedder=HashingEmbedder(),
            config=RAGAgentConfig(top_k=3),
            reranker=Noop(),
        )

        assert agent._hybrid_enabled() is True


class TestRetrievalThroughTheAgent:
    def _retrieve_ids(self, agent, query):
        result = agent.retrieve(query, retrieval_mode="classic")
        return [d.id for d in result.docs]

    def test_dense_retrieval_still_works(self, store):
        got = self._retrieve_ids(_agent(store), "how do I install the package")

        assert any(i.startswith("install:") for i in got)

    def test_hybrid_promotes_the_rare_literal(self, store):
        """The end-to-end version of the phase's headline measurement."""
        dense = self._retrieve_ids(_agent(store), "ENAMETOOLONG")
        hybrid = self._retrieve_ids(_agent(store, hybrid_search=True), "ENAMETOOLONG")

        assert hybrid[0] == "errors:0"
        assert dense[0] != "errors:0"

    def test_mmr_reduces_redundancy_end_to_end(self, store):
        plain = self._retrieve_ids(_agent(store), "prompt caching identical system prompt")
        diverse = self._retrieve_ids(
            _agent(store, mmr_lambda=0.3), "prompt caching identical system prompt"
        )

        assert sum(i.startswith("dup:") for i in plain) > sum(
            i.startswith("dup:") for i in diverse
        )
