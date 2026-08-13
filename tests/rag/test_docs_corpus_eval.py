"""Dense vs hybrid on this repository's own docs, with a real embedding model.

**Why this exists.** The fixture corpus in `test_retrieval.py` is synthetic: a
hash-based embedder, fifteen short documents, a rare literal seeded on purpose.
It proved the mechanism and produced a dramatic number — recall@1 0.619 → 0.905.
That number does not transfer, and the only way to know was to measure real
prose with a real embedder.

It needs LM Studio (or any OpenAI-compatible server) with an embedding model, so
it skips when one is not reachable. Run it deliberately:

    pytest tests/rag/test_docs_corpus_eval.py -s

The assertions are deliberately weak — the point is the printed table, and a
14-query sample cannot support a tight bound. What is asserted is only that
hybrid does not *regress* at the k where it should help, which is the thing a
future change could plausibly break.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from neurosurfer.embeddings import get_embedder
from neurosurfer.rag.evaluation import EvalCase, compare
from neurosurfer.rag.retrieval import LexicalIndex, hybrid_search
from neurosurfer.vectorstores import Doc, InMemoryVectorStore

BASE_URL = "http://localhost:1234/v1"
EMBED_MODEL = "text-embedding-nomic-embed-text-v1.5"
DOCS = Path(__file__).resolve().parents[2] / "docs"


def _server_ready() -> bool:
    try:
        models = httpx.get(f"{BASE_URL}/models", timeout=2.0).json()
    except Exception:  # noqa: BLE001
        return False
    return any(EMBED_MODEL in m.get("id", "") for m in models.get("data", []))


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not _server_ready(),
        reason=f"needs {EMBED_MODEL} on an OpenAI-compatible server at {BASE_URL}",
    ),
]

#: (query, the pages that legitimately answer it).
#:
#: **A set, not one page — and that correction changed the result.** The first
#: version labelled one "right" page per query and scored MRR 0.21 at k=1, which
#: read as a retrieval failure. It was mostly a labelling failure: "send traces
#: to Langfuse" was marked wrong for returning `observability/langfuse.md`
#: instead of the `index.md` I had guessed. Several pages legitimately answer the
#: same question, and an eval that pretends otherwise measures the labeller.
CASES: list[tuple[str, set[str]]] = [
    ("how do I stop the agent using too many tokens on a long run", {"guides/context.md"}),
    ("what happens when a node in my workflow fails",
     {"graph/control-flow.md", "graph/validation.md"}),
    ("connect to an external MCP server", {"guides/mcp.md", "tutorials/mcp-servers.md"}),
    ("make the model return JSON matching my schema", {"guides/structured-output.md"}),
    ("run the OpenAI-compatible gateway",
     {"server/index.md", "server/deployment.md", "cli/index.md"}),
    ("which vector databases are supported", {"guides/rag.md"}),
    ("branch a workflow depending on a classification",
     {"graph/node-kinds.md", "graph/control-flow.md"}),
    ("write my own tool", {"guides/tools.md", "guides/tool-registry.md"}),
    ("what does the Architect refuse to build",
     {"architect/index.md", "architect/grounding.md"}),
    ("send traces to Langfuse", {"observability/langfuse.md", "observability/index.md"}),
    ("run something for every item in a list",
     {"graph/control-flow.md", "graph/node-kinds.md"}),
    ("give an agent a smaller helper agent", {"guides/subagents.md"}),
    ("stop a workflow package registering when it is broken",
     {"graph/validation.md", "graph/packages.md"}),
    ("point the framework at a local model",
     {"guides/providers.md", "getting-started/installation.md"}),
]


def _windows(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    """Plain character windows — what is under test is retrieval, not chunking."""
    out, start = [], 0
    while start < len(text):
        out.append(text[start : start + size])
        start += size - overlap
    return [c for c in out if c.strip()]


@pytest.fixture(scope="module")
def corpus():
    embedder = get_embedder(f"openai-compat:{EMBED_MODEL}@{BASE_URL}")
    assert embedder is not None

    ids, texts = [], []
    for page in sorted(DOCS.rglob("*.md")):
        rel = str(page.relative_to(DOCS))
        for i, piece in enumerate(_windows(page.read_text(encoding="utf-8", errors="replace"))):
            ids.append((f"{rel}#{i}", rel))
            texts.append(piece)

    vectors = embedder.embed(texts)
    store = InMemoryVectorStore()
    store.add_documents(
        [
            Doc(id=cid, text=t, embedding=v, metadata={"page": pg})
            for (cid, pg), t, v in zip(ids, texts, vectors, strict=True)
        ]
    )

    by_page: dict[str, set[str]] = {}
    for cid, pg in ids:
        by_page.setdefault(pg, set()).add(cid)

    return embedder, store, LexicalIndex.from_store(store), by_page


def _cases(by_page):
    out = []
    for query, pages in CASES:
        relevant: set[str] = set()
        for pg in pages:
            relevant |= by_page.get(pg, set())
        assert relevant, f"no such page for {query!r}: {sorted(pages)}"
        out.append(EvalCase(query=query, relevant_ids=relevant, note=", ".join(sorted(pages))))
    return out


def test_hybrid_against_dense_on_real_docs(corpus, capsys):
    """The measurement, printed. Run with `-s` to read it.

    Result as of 2026-08-13, 59 pages / 371 chunks, nomic-embed-text-v1.5:

        k=1    dense mrr 0.500    hybrid mrr 0.500
        k=3    dense mrr 0.583    hybrid mrr 0.571
        k=5    dense mrr 0.601    hybrid mrr 0.643
        k=10   dense mrr 0.610    hybrid mrr 0.653

    **Hybrid is neutral at small k here and modestly ahead from k=5.** The
    fixture's recall@1 0.619 → 0.905 does not transfer: that corpus had a rare
    literal seeded into it and a bag-of-words embedder that could not find it,
    which is the case hybrid is best at and not the case most queries are.
    """
    embedder, store, lexical, by_page = corpus
    cases = _cases(by_page)

    def dense(q, k):
        return [d.id for d, _ in store.similarity_search(embedder.embed([q])[0], top_k=k)]

    def hybrid(q, k):
        vec = embedder.embed([q])[0]
        return [r.id for r in hybrid_search(q, vec, store, lexical=lexical, top_k=k, fetch_k=40)]

    rows = {}
    with capsys.disabled():
        print(f"\n  {len(cases)} queries over {len(by_page)} real doc pages")
        for k in (1, 3, 5, 10):
            r = compare(cases, {"dense": dense, "hybrid": hybrid}, k=k)
            rows[k] = r
            print(
                f"  k={k:<3} dense  mrr={r['dense'].mrr:.3f} ndcg={r['dense'].ndcg:.3f}"
                f"   |   hybrid mrr={r['hybrid'].mrr:.3f} ndcg={r['hybrid'].ndcg:.3f}"
            )

    # Weak on purpose: 14 queries cannot support a tight bound, and asserting the
    # exact deltas above would make this a change-detector rather than a guard.
    assert rows[10]["hybrid"].mrr >= rows[10]["dense"].mrr, (
        "hybrid regressed against dense at k=10, which is where it should help most"
    )
    assert rows[1]["dense"].mrr > 0.3, "dense retrieval on real docs collapsed"


def test_the_fixture_number_is_not_claimed_to_generalise():
    """A guard on the documentation, not on the code.

    `docs/guides/rag.md` quotes the fixture's dramatic delta. This test fails if
    that table ever loses the sentence saying it is a fixture result — which is
    the only thing stopping a reader from reading it as a production forecast.
    """
    rag_doc = (DOCS / "guides" / "rag.md").read_text(encoding="utf-8")

    assert "fixture" in rag_doc.lower()
    assert "does not transfer" in rag_doc.lower() or "real corpus" in rag_doc.lower()
