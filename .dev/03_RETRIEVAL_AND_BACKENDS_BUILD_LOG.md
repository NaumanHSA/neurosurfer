# 03 — Retrieval and the backends behind it: build log

**Shipped 2026-08-13**, eight phases in one pass. **1506 passed, 7 skipped**,
ruff clean, 59 doc pages and 91 doc imports checked. The suite grew from 1235 to
1506 — 271 tests, most of them on surfaces that previously had none.

Every phase was committed on its own and the full suite run between them, so the
history bisects.

---

## What the phases actually cost

| Phase | Landed | Note |
|---|---|---|
| 1 — the floor | `acddd7f` | The prerequisite. Two defects, a filter grammar, capability flags, and the conformance suite. |
| 2 — embeddings | `e0dc833` | The one users feel: a local stack no longer needs `torch` to embed. |
| 3 — retrieval quality | `308cc0d` | Harness first, and it earned that ordering immediately — see below. |
| 4 — Qdrant | `3f3e21d` | **Passed the conformance suite unmodified, first run.** |
| 5 — RAG shapes | `cfae465` | Four strategies plus incremental re-index. |
| 6 — cost | `3441f7c` | Small, and it makes every existing trace more useful. |
| 7 — providers | `45ad7fe` | Gemini natively; Bedrock as a subclass. |
| 8 — loose ends | this commit | `react` structured output, the worker guard, the README. |

---

## The three things worth remembering

### The conformance suite paid for itself in one commit

Phase 1's deliverable was not the filter grammar — it was
`tests/vectorstores/conformance.py`. Phase 4 added Qdrant: a different query
language, a different id type (integers or UUIDs, where ours are strings like
`install:0`), a different distance implementation. **It passed all 27 behaviours
on the first run with no change to the suite.**

That is the evidence the interface is a contract rather than a description of
Chroma, and it is the strongest result in this plan. The one place Qdrant and
the reference disagreed — `must_not` matches a document that lacks the field
entirely, where `filters.matches` says a field the metadata does not carry never
matches — was found by a test that cross-checks nine filters against the
reference implementation rather than against my reading of Qdrant's docs. That
test exists precisely because "I read the documentation" is not verification.

### Writing the harness first caught my own overclaim, immediately

The plan said the eval harness comes first within Phase 3, on the grounds that
plan 01 had already recorded once what it costs to build a mechanism with no way
to measure it. The first draft of the hybrid test asserted that dense retrieval
**could not find** `ENAMETOOLONG` at all. It can — at rank 3.

The real result is more useful than the overclaim would have been:

```
k=1   dense  recall 0.619  mrr 0.714      hybrid  recall 0.905  mrr 1.000
k=3   dense  recall 0.952  mrr 0.833      hybrid  recall 0.952  mrr 1.000
```

The gain is in **ranking**, and at the small k a context window uses, ranking is
recall. Had the harness come second, that sentence would have shipped in a
changelog as a claim nobody checked.

The same discipline caught two more of my own errors in Phase 5. The
abbreviation guard in the sentence splitter was positioned so it guarded
nothing — the lookbehinds are evaluated *after* the terminal punctuation, so
`(?<!\bDr)` was looking at `r.` and never matched. And the first semantic-chunking
test used the fixture's hashing embedder and failed correctly: a bag-of-words
measures vocabulary overlap, so two short sentences about compilers can look
less alike than a cats→compilers pair that happens to collide. That is a fact
about the fixture, not the algorithm, so the algorithm is now tested with
vectors the test controls.

### The defects were worse than §0 measured

`§0.1` recorded two. There were five, and three only surfaced once the code was
under test:

1. **`InMemoryVectorStore` could not be instantiated** — known, and the reason
   it survived is that nothing ever built one.
2. **The `delete_documents` signature disagreed** between base and Chroma — known.
3. **Chroma's scores were wrong on every collection it had ever created.** Its
   default space is squared L2 and the code returned `1.0 - distance` as though
   it were cosine: orthogonal vectors scored **-1.0** instead of `0.0`, and any
   `similarity_threshold` was applied to that wrong scale. Not in §0 — found by
   probing `chromadb` directly before building on it.
4. **`list_all_documents` minted a fresh `uuid4()` per row**, so the ids it
   returned matched nothing and deleting by them was a silent no-op.
5. **`RAGAgent` could only ever use sentence-transformers.** It accepted an
   `Embedder` object, but a *name* went straight to `_LocalEmbedder`, so the
   whole point of Phase 2 was unreachable from the agent until that one line
   changed.

Chroma's `modify()` also refuses any payload carrying `hnsw:space` — even
unchanged — which is why persisting the embedding identity strips the `hnsw:`
keys. The space lives in `configuration_json` and survives.

---

## Decisions that went differently from the plan

**§2.2 said the embeddings change would break callers, and it did not.**
`get_embedder` had **zero call sites** in the package — it was exported and
documented and never used, because `RAGAgent` bypassed it. So making it raise on
a configured-and-broken backend broke nothing, and the compatibility work went
where it was actually needed: a bare string with no recognised prefix still
means a sentence-transformers model, because that is what every existing config
holds.

**Bedrock is a subclass, which the plan did not anticipate.** Phase 7 was written
as "add two providers". Gemini is genuinely a third shape and got a full
implementation with 23 tests against a fake server. Bedrock serves the *same*
Messages API — so the translation, the streaming map and the thinking handling
are already correct, and copying them would have created two things to keep in
step. Only the client and the model id differ. A test asserts the shared methods
are literally the same function objects, so a future copy-paste fails rather
than diverges silently.

**MMR needed a wider pool than `top_k`, and the bug was subtle.** A selecting
stage handed exactly `top_k` candidates returns all of them — so MMR looked like
a setting that did nothing. Caught by an end-to-end test through `RAGAgent`, not
by the unit test above it, which only asserted the result length.

---

## Deliberately not done

- **GraphRAG.** A different product with an entity-extraction pipeline behind
  it; nothing measured here calls for it.
- **Cohere and Voyage embeddings, and a reranker beyond the local
  cross-encoder.** One class each now that the Protocol exists; they teach the
  interface nothing the first three did not.
- **pgvector.** The conformance suite makes it cheap, and two backends already
  proved the contract. It is the obvious third when someone wants it.
- **A live Bedrock call.** `boto3` is not installed and there are no AWS
  credentials here, so nothing exercises a real request. Everything
  Bedrock-*specific* is tested; the shared path is covered by the Anthropic
  provider's own tests. Stated in the commit rather than implied.
- **Re-running the tutorial notebooks after phases 2–8.** LM Studio was down for
  the second half of this work. The changes are additive — new optional
  parameters, new fields, new modules — and the full suite covers the paths the
  notebooks use, but **they have not been re-executed since Phase 1** and should
  be before the merge.

---

## What this leaves open

- **Promoting `declared_inputs_are_read_by_something` to blocking.** Plan 01 §10
  removed the false positive that made it unsafe; nothing here changed it.
- **Retrieval quality on a real corpus.** The numbers above are from a
  15-document fixture with a deterministic embedder — enough to prove the
  mechanism and to catch a regression, not enough to predict a production delta.
  The harness is the thing to point at a real corpus.
- **The OpenAI and Gemini rows of the price table** are the published rates from
  memory and are marked in the file as worth verifying. The Anthropic rows came
  from the API reference.
