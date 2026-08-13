# 03 — Retrieval, and the backends behind it

**Status: all eight phases shipped, 2026-08-13** (`acddd7f`..`d6dcad2`, eight
commits — one per phase, so the history bisects). **1506 pass / 7 skip**, up from
1235; ruff clean; 59 doc pages and 94 doc imports checked. What each phase cost
and what it turned out to be is in the
[build log](03_RETRIEVAL_AND_BACKENDS_BUILD_LOG.md).

**This document is left as it was written**, apart from the ticked boxes and the
three corrections marked *(wrong — …)* below. A plan that is quietly edited to
match its outcome stops being evidence of what was predicted, which is most of
what makes it worth reading later.

**Goal:** retrieval that can be trusted on the queries people actually ask, and
a set of backend seams that a second implementation has proven — rather than an
abstract base class with one subclass and an `if/else` wearing the word
*pluggable*.

**Why now.** Plans 01 and 02 made the graph engine and the Architect honest: a
workflow that cannot run is refused before a model call, and every subsystem has
a page. Retrieval was not in scope for either, and it shows. The framework
describes itself as *"LLM reasoning, tools, and retrieval"*; two of those three
have had a plan.

---

## §0 — The diagnosis

Measured 2026-08-13 against `3fb20f8`, by reading the package. Every claim
carries the line that proves it.

**Sizes, for proportion.** `neurosurfer/rag/` is 3,574 lines across 13 files.
`neurosurfer/vectorstores/` is 455 across 4. `neurosurfer/embeddings/` is **69
lines in one file** — and it is the layer every retrieval path depends on.

### §0.1 — Two defects, one of them a documented API that cannot run

> **(wrong — there were five.)** Reading the code found two. Probing `chromadb`
> before building on it found three more, all of them worse:
>
> * **Chroma's scores were wrong on every collection it had ever created.** Its
>   default space is squared L2 and the code returned `1.0 - distance` as though
>   it were cosine, so orthogonal vectors scored **-1.0** instead of `0.0` and
>   every `similarity_threshold` was applied to that wrong scale.
> * **`list_all_documents` minted a fresh `uuid4()` per row**, so the ids it
>   returned matched nothing and deleting by them was a silent no-op.
> * **`RAGAgent` could only ever use sentence-transformers** — it accepted an
>   `Embedder` object, but a *name* went straight to `_LocalEmbedder`, which made
>   Phase 2 unreachable from the agent until one line changed.
>
> The lesson is in *how* they were found: none is visible by reading, and all
> three surfaced within minutes of running the thing against a real backend. A
> §0 assembled by reading undercounts by roughly the number of dependencies it
> did not exercise.

**`InMemoryVectorStore` cannot be instantiated.**

```
>>> from neurosurfer.vectorstores import InMemoryVectorStore
>>> InMemoryVectorStore(dim=3)
TypeError: Can't instantiate abstract class InMemoryVectorStore
           with abstract method delete_documents
```

`BaseVectorDB.delete_documents` is abstract
([`base.py:198`](../neurosurfer/vectorstores/base.py#L198));
[`in_memory_store.py`](../neurosurfer/vectorstores/in_memory_store.py)
implements six of the seven abstract methods and not that one. It is exported by
name from [`vectorstores/__init__.py`](../neurosurfer/vectorstores/__init__.py),
and [`docs/guides/rag.md:79`](../docs/guides/rag.md#L79) recommends it —
*"ephemeral, dependency-free; handy for tests and demos"*.

Nothing in the package or the tests instantiates it, which is exactly why it
survived. `check_docs_imports.py` verifies that a documented symbol **imports**;
this one imports perfectly and raises on use.

**The base class and its only real backend disagree about the same method.**

| | Signature |
|---|---|
| [`base.py:198`](../neurosurfer/vectorstores/base.py#L198) | `delete_documents(self, docs: list[Doc])` |
| [`chroma.py:86`](../neurosurfer/vectorstores/chroma.py#L86) | `delete_documents(self, ids: list[str])` |

Nothing calls it, so neither is exercised. One of them is what a third backend
would be written against, and the ABC currently does not say which.

### §0.2 — Embeddings is one backend behind no seam

[`embeddings/__init__.py`](../neurosurfer/embeddings/__init__.py) is the whole
layer. `get_embedder` is a three-branch `if`
([`:44`](../neurosurfer/embeddings/__init__.py#L44)) whose fallthrough assumes
any unrecognised string is a **sentence-transformers model name**
([`:47`](../neurosurfer/embeddings/__init__.py#L47)):

```python
if name in ("local", "sentence-transformers", "st"):
    return _LocalEmbedder()
# Accept a model name directly (e.g. "intfloat/e5-small-v2")
return _LocalEmbedder(model=backend)
```

So there is exactly one way to produce a vector, and it requires `torch` +
`sentence-transformers` — a multi-gigabyte install — even for a user whose entire
stack is a local OpenAI-compatible server that **already exposes
`/v1/embeddings`**. The framework speaks that protocol fluently for chat
([`llm/providers/openai.py`](../neurosurfer/llm/providers/openai.py)) and not at
all for embeddings.

**And it never raises.** Every failure returns `None`
([`:48`](../neurosurfer/embeddings/__init__.py#L48)), documented as the
"cardinal rule" so callers can degrade to lexical search. That is right for *"the
optional dependency is not installed"* and wrong for *"the API key is expired"* —
the second becomes "retrieval quality quietly got worse", which is the
silent-plausible-success class plan 01 spent seven phases removing from the graph
engine. `mcp/sources/` already draws this distinction correctly: `SourceUnavailable`
means *not configured*, `McpRegistryError` means *tried and broke*.

### §0.3 — The vector-store ABC has never been proven

`BaseVectorDB` has one working implementation. An interface with one
implementation is a description of that implementation, and this one shows it:

- **`similarity_search` has no filter grammar.** The `metadata_filter` parameter
  is `dict[str, Any]` and Chroma turns it into `{"$in": v}` for lists and
  equality otherwise ([`chroma.py:45-48`](../neurosurfer/vectorstores/chroma.py#L45)).
  Ranges, negation and boolean composition — the things Qdrant and pgvector are
  chosen *for* — have nowhere to go, so the second backend will invent its own
  and callers will branch on which one they have.
- **No `upsert` in the contract**, though Chroma opportunistically uses it
  ([`chroma.py:32`](../neurosurfer/vectorstores/chroma.py#L32)).
- **Sync only.** Every method is `def`, and the graph executor runs nodes in
  worker threads. A hosted vector DB wants `await`.
- **No namespace or tenancy**, so one process serving two users shares a
  collection or manages the separation itself.

### §0.4 — RAG is one pipeline, and the good parts stop at the chunker

The chunker is genuinely strong — a strategy registry, a router, AST-aware
Python, header-aware Markdown, 636 lines of it. Everything *after* retrieval is
fixed:

- **`RetrievalMode = Literal["classic", "smart"]`**
  ([`config.py:63`](../neurosurfer/rag/config.py#L63)) — and "smart" is scope and
  `top_k` planning, not a different retrieval shape.
- **No reranking anywhere in the package.** `grep -rn rerank neurosurfer/`
  returns nothing.
- **No hybrid search, no MMR, no diversity.** `similarity_search` returns top-k
  by raw cosine, so five chunks from one paragraph can crowd out the answer.
- **No span-level citation.** `ContextBuilder` emits a `Source: {source}` header
  ([`context_builder.py`](../neurosurfer/rag/context_builder.py)); there is no
  offset back into the original document, so an answer cannot be highlighted.
- **No incremental re-index.** `RAGIngestor` dedupes by content hash, but there
  is no "this one file changed" path.

**The BM25 already exists and RAG does not use it.**
[`agents/lexical.py`](../neurosurfer/agents/lexical.py) is a general,
dependency-light BM25 ranker with a term-overlap fallback that never raises. It
was extracted so *"web-search result injection and long-term memory retrieval
share one implementation"*. `rag/` imports nothing from it. Hybrid search here is
mostly a wiring job, not a new dependency.

### §0.5 — Nothing records which model wrote a vector

Three defaults are chosen independently, in three files:

| Where | Default |
|---|---|
| [`embeddings/__init__.py:56`](../neurosurfer/embeddings/__init__.py#L56) | `all-MiniLM-L6-v2` |
| [`rag/config.py:107`](../neurosurfer/rag/config.py#L107) | `intfloat/e5-small-v2` |
| [`cache/embedder.py:23`](../neurosurfer/cache/embedder.py#L23) | `all-MiniLM-L6-v2` (in a docstring example) |

No collection records the model that embedded it, so pointing a differently
embedded query at an existing store returns confident nonsense rather than an
error. `CachedEmbedder`'s key is
`sha256(texts)` ([`cache/embedder.py:11`](../neurosurfer/cache/embedder.py#L11))
with no model component — harmless today because the cache is per-instance and
in-memory, and a trap the moment it is backed by disk.

### §0.6 — Adjacent gaps this survey turned up

- **No cost accounting.** `Usage` threads token counts through every layer and
  nothing converts them to money — there is no price table in the package.
  `pyproject.toml` describes Langfuse as *"traces, cost, evals"*; the cost half is
  Langfuse's, not ours.
- **Provider coverage stops at two APIs.** Anthropic, OpenAI, and
  OpenAI-compatible. **Gemini** and **Bedrock** are the two commonly-asked-for
  APIs that are neither.
- **The run store is JSON files loaded into memory**
  ([`workflow_runs/store.py:166`](../neurosurfer/app/server/workflow_runs/store.py#L166)),
  while the gateway offers `--workers N`. Two workers, two divergent stores.
- **`react` cannot return a shaped answer.**
  [`kinds/react.py`](../neurosurfer/graph/engine/kinds/react.py) says it plainly:
  the field was offered, written into YAML, reviewed, and read by nothing, so it
  was removed. *"Making the loop honour a schema is a real feature and a fine
  one; until it exists the honest thing is not to offer the field."*
- **`README.md:49` claims a memory tool.** There is no memory tool and no
  `MemoryStore` — the same drift already fixed in tutorials 00 and 02.

### §0.7 — The shape to copy already exists in this repository

[`mcp/sources/`](../neurosurfer/mcp/sources/) is the best-designed plugin point
in the package, and it was written for a problem structurally identical to this
one — *"one interface, more than one index"*. It has:

- a `RegistrySource` **Protocol** with `id` / `label` / `blurb`
  ([`base.py:108`](../neurosurfer/mcp/sources/base.py#L108));
- **`Capability` flags** ([`base.py:87`](../neurosurfer/mcp/sources/base.py#L87))
  so a caller *asks* what a backend can do — *"sorting by popularity against an
  index with no use counts is a control that does nothing, which is worse than no
  control"*;
- the rule that **every extra is optional**: a source that cannot report a field
  leaves it empty and never invents it;
- an **ambient active source** via `ContextVar`, with the documented trap that a
  raw thread does not inherit it.

Embeddings and vector stores should be built to that shape. This plan does not
invent a plugin pattern; it applies the one that is already working.

---

## §1 — The phases

Bottom-up, for the reason plan 01 was: a seam is not real until a second
implementation has pushed on it, and quality work built on an unproven interface
gets rewritten when the second backend arrives.

### Phase 1 — The floor: make the contract true ✅

The two defects, and the interface they expose.

- [x] Implement `delete_documents` on `InMemoryVectorStore`, with the test that
      instantiates it — the missing test is why the defect survived.
- [x] Settle the `delete_documents` signature. **Recommendation: `list[str]` of
      ids**, with `delete_docs(docs)` as a thin convenience. Deleting by id is
      what every backend's API takes, and requiring a full `Doc` to delete one
      means reading it back first.
- [x] Give `BaseVectorDB` a **declared filter grammar** — a small typed vocabulary
      (`eq`, `in`, `gt/gte/lt/lte`, `and`/`or`/`not`) that each backend translates.
      Small now; a migration once two backends have shipped their own dialect.
- [x] Put `upsert` semantics in the contract rather than leaving it to
      `hasattr`.
- [x] A **backend conformance suite** — one parametrised test module every
      implementation must pass. This is the deliverable that makes Phase 4 cheap.
- [x] `Capability` flags for stores, following `mcp/sources`: `filtering`,
      `hybrid`, `async_api`, `namespaces`.

**Exit:** `InMemoryVectorStore` passes the conformance suite, and the suite is
what "implements `BaseVectorDB`" means.

### Phase 2 — Embeddings as a real plugin point ✅

The highest value per line in this plan. Everything downstream needs a vector,
and today there is one way to make one.

- [x] An `EmbeddingBackend` **Protocol** mirroring `RegistrySource`: `id`,
      `label`, `dimensions`, `max_batch`, `available()`, `embed(texts)`.
- [x] **`openai-compat`** — `/v1/embeddings` against any base URL. The one that
      unblocks a fully local stack with no torch, on a server the user is already
      running.
- [x] **`openai`** hosted (`text-embedding-3-small` / `-large`).
- [x] Keep **`sentence-transformers`** as-is, behind the same Protocol.
- [x] **Batching, retry and rate limits** via the existing
      [`llm/retry.py`](../neurosurfer/llm/retry.py) — a network backend needs the
      429 handling chat already has.
- [x] Separate **"not configured"** from **"configured and broken."**
      `get_embedder` keeps returning `None` for the first and raises for the
      second. This is a behaviour change and belongs in the CHANGELOG.
- [x] **One default, in one place.** Fold the three of §0.5 into a single
      resolved default.
- [x] Record **model identity and dimension on the collection**, and refuse a
      query embedded by a different model with a message that says which two.

**Deferred to a later phase, deliberately:** Cohere and Voyage. Both are one
class each once the Protocol exists, and neither teaches the interface anything
the first three do not.

### Phase 3 — Retrieval quality ✅

The changes that most improve answers without touching a public interface.

- [x] **Hybrid retrieval** — dense + BM25 with reciprocal-rank fusion, reusing
      [`agents/lexical.py`](../neurosurfer/agents/lexical.py) rather than adding a
      dependency. Fixes the case dense retrieval is worst at: exact identifiers,
      error codes, proper nouns.
- [x] **Reranking** as an optional stage over the top-N: a local cross-encoder,
      and a hosted option behind the same Protocol. Usually the single largest
      quality jump; also the one place a network hop per query is justified.
- [x] **MMR / diversity** on the final selection, so one paragraph cannot fill
      the context window.
- [x] **Span-level citations** — carry `(start, end)` offsets from chunker to
      `ContextBuilder` so an answer can point at its source.
- [x] A **retrieval evaluation harness** — recall@k and MRR over a small fixture
      corpus, so each item above is a measured delta rather than a belief. Shares
      its shape with the Architect's
      [`agent/harness.py`](../neurosurfer/architect/agent/harness.py).

**The harness comes first within this phase.** Every other item is a claim about
quality, and this plan should not repeat plan 01's §7 position of having the
mechanism and no measurement.

### Phase 4 — A second vector backend ✅

- [x] **Qdrant** (recommended first: the strongest filtering story, so it pushes
      hardest on Phase 1's grammar) **or LanceDB** (embedded, no server, the
      natural successor to Chroma for people who chose Chroma to avoid running
      one).
- [x] Make it pass the conformance suite unmodified. Any change the suite needs
      is a finding about Phase 1, and belongs back there.
- [x] **pgvector** if the suite makes it cheap — many teams already run Postgres
      and would rather not add a service.

**Exit:** two backends, one suite, and a documented answer to "which should I
use".

### Phase 5 — RAG shapes beyond classic ✅

Each is a strategy behind one seam, not a fork of the pipeline.

- [x] **Contextual retrieval** — a document-level summary prepended to each chunk
      before embedding. Cheap, large measured gains, and it fits the existing
      ingestor.
- [x] **Query rewriting / multi-query / HyDE** — one question becomes several
      retrievals.
- [x] **Parent-document retrieval** — embed small, return the enclosing section.
- [x] **Sentence-window** and **semantic (embedding-distance) chunking** — both
      land in the chunker's existing strategy registry.
- [x] **Incremental re-index** — re-chunk only what changed.

**Not doing: GraphRAG.** It is a different product with an entity-extraction
pipeline and a graph store behind it, and nothing measured here calls for it.
Revisit when a real corpus demands multi-hop.

### Phase 6 — Cost accounting ✅ built, then ❌ **removed**

> **(wrong — the framework should not know anyone's prices.)** This shipped and
> was then taken out at the owner's direction, and the reasoning is better than
> the plan's was:
>
> *"We do not care about the cost in terms of actual money, what we care about is
> only the number of tokens used, input and output. Nowhere should we convert
> them to dollars. We do tokens because we have traces, we have Langfuse, OTel,
> where it is important."*
>
> That is the right boundary. Vendor rates change per contract and per region, a
> table maintained here goes stale silently, and being confidently wrong about
> money is worse than being silent about it. Langfuse and OTel already receive
> the model name alongside `Usage` and own their own rate tables — so the
> conversion belongs there, not in the framework that counts the tokens.
>
> Removed in full: `llm/pricing.py`, `RunResult.cost()`,
> `GraphExecutionResult.total_cost()`, and the `model` fields added only to feed
> them. `Usage` is unchanged and still threads everywhere it did.
>
> The items below are what was built before the removal, kept as the record.

Small, orthogonal, and it makes every trace already emitted more useful.

- [x] A **price table** — per-model input / output / cache-read rates, as data,
      versioned like the capability manifest.
- [x] Cost on `Usage` → `RunResult` → the trace → the gateway response.
- [x] Surface it where it is felt: an Architect build's repair loop is the most
      expensive thing in the framework and currently reports only turns.

### Phase 7 — Provider coverage ✅

- [x] **Gemini** and **Bedrock** — the two commonly-wanted APIs that are neither
      Anthropic-shaped nor OpenAI-shaped.
- [x] Confirm both against the existing
      [`tests/test_provider_parity.py`](../tests/test_provider_parity.py), which
      is the conformance suite for this seam and already exists.

### Phase 8 — The loose ends §0.6 found ✅

Independent of everything above; listed so they are not lost.

- [x] **Structured output on `react` nodes** — make the loop honour a schema, and
      restore the spec field the kind honestly withdrew.
- [x] **Run store for more than one worker** — SQLite, or an explicit
      single-worker constraint enforced at startup rather than implied.
- [x] **`README.md:49`** — drop the memory tool it claims and does not have.

---

## §2 — Decisions, with a recommendation

### §2.1 — Does this plan own cost and providers?

They are in §0.6 because the survey found them, not because they are retrieval.
Phases 6 and 7 are separable and could be a plan 04.

**Recommendation: keep them here, run them last.** Splitting a two-file price
table into its own plan costs more in ceremony than it saves, and both are
small enough to be the thing you do while waiting on a decision elsewhere. If
either grows a §0 of its own, promote it then — which is the test plan 01's §9
used and is a good one.

### §2.2 — Does the embeddings change break callers?

> **(wrong — it broke nobody.)** `get_embedder` had **zero call sites** in the
> package: exported, documented, and never used, because `RAGAgent` bypassed it
> and built `_LocalEmbedder` directly. So the decision below cost nothing to
> make, and the compatibility work that actually mattered was somewhere else
> entirely — a bare string with no recognised prefix still resolving to a
> sentence-transformers model, because that is what every existing config holds.
>
> Worth noting for the next §2: *"does this break callers"* is answerable with
> `grep` before it is argued about.

Yes, in one narrow way: `get_embedder` currently swallows every failure and
Phase 2 makes a *configured and broken* backend raise. A caller relying on the
silent `None` to degrade to lexical search will now see an exception.

**Recommendation: make it raise anyway, and say so prominently.** The current
behaviour turns an expired API key into slightly worse answers with nothing
logged at the level that matters — the exact failure mode the graph engine was
rebuilt to eliminate. Callers who genuinely want "never fail" get an explicit
flag.

### §2.3 — Chroma's future

Chroma stays the default. It is what the `rag` extra installs, what the tutorials
use, and what every existing store on disk was written by. Phase 4 adds a
backend; it does not migrate anyone.

### §2.4 — How far to take the filter grammar

There is a real temptation to design something SQL-shaped. **Recommendation:
stop at what two backends can both express** — equality, membership, ranges,
and boolean composition. Anything richer is a promise the in-memory store cannot
keep, and a `Capability` flag is the honest way to say a backend does more.

---

## §3 — What "done" looks like

Each line below is what was written before the work; the verdict after it is
what shipped.

| Criterion | Outcome |
|---|---|
| `InMemoryVectorStore` is real, and the docs page recommending it is true | ✅ It is the reference implementation and supports every capability |
| Two vector backends pass one conformance suite; a third is one class | ✅ Three (Chroma, Qdrant, in-memory). **Qdrant passed unmodified, first run** |
| A user with LM Studio and no `torch` can ingest, embed, and query | ✅ `openai-compat:<model>@<base_url>`, verified live against LM Studio |
| Hybrid + rerank land as **measured** deltas, not changelog assertions | ✅ recall@1 0.619 → 0.905, MRR 0.714 → 1.000, from `rag/evaluation.py` |
| A run reports what it cost | ❌ **Withdrawn.** A run reports the *tokens* it used; converting them to money is the observability backend's job, not this framework's — see Phase 6 |
| Every claim in `README.md` and `docs/guides/rag.md` is honourable | ✅ Both corrected; docs gates green |

**Ordering, if only some of it happens:** Phase 1 then Phase 2. Phase 1 is a day
and makes the contract true; Phase 2 is the one users feel. Phase 3 without the
harness at its head is the one way this plan repeats a mistake it has already
written down.

> **All eight ran, and the ordering held.** Phase 1's conformance suite is what
> made Phase 4 a day rather than a week, and Phase 3's harness caught an
> overclaim in its own first test — see the build log. The one criterion above
> that is weaker than it sounds is the measured delta: a 15-document fixture
> with a deterministic embedder proves the mechanism and catches a regression;
> it does not predict a production number. Pointing the harness at a real corpus
> is the first thing to do next.

---

## §4 — What this plan did not close

- **Retrieval quality on a real corpus.** The harness exists and the fixture is
  synthetic on purpose. The numbers are a mechanism proof, not a forecast.
- **A live Bedrock call.** No boto3 and no AWS credentials on this machine.
  Everything Bedrock-*specific* is tested; the shared path rides the Anthropic
  provider's own suite.
- **The tutorial notebooks since Phase 1.** LM Studio went down partway through
  and did not come back. Every change since is additive and the suite covers the
  paths they use, but they have not been re-executed and should be before merge.
- **The OpenAI and Gemini price rows.** Published rates written from memory and
  marked in the file as worth verifying. The Anthropic rows came from the API
  reference.
- **Async store methods, namespaces/tenancy, pgvector.** §0.3 named all three;
  none was needed to prove the contract, and the conformance suite makes each
  cheap when someone wants it.
