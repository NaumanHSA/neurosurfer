# RAG

Retrieval-Augmented Generation lets an agent answer from **your** documents. Neurosurfer's RAG
pipeline is: **ingest → chunk → embed → retrieve → token-aware context → generate**.

Install the extra:

```bash
pip install "neurosurfer[rag]"
```

## RAGAgent — the single entry point

`RAGAgent` wires an embedder, a vector store, and a provider together. Ingest sources once, then ask
questions:

```python
from neurosurfer.rag import RAGAgent
from neurosurfer.vectorstores import ChromaVectorStore

vectorstore = ChromaVectorStore(
    collection_name="handbook",
    persist_directory="./rag-storage",
)

rag = RAGAgent(
    llm=provider,                 # any Provider (see the Providers guide)
    vectorstore=vectorstore,
    embedder="all-MiniLM-L6-v2",  # a sentence-transformers backend name
)

# 1) Ingest — paths, directories, .zip archives, git folders, URLs, or raw text
rag.ingest(["./docs", "https://example.com/spec.html"])

# 2) Ask — retrieves relevant chunks, then generates an answer
answer = rag.run("What does the handbook say about refunds?")
print(answer)
```

`ingest()` accepts a single source or an iterable and routes each to the right reader (PDF, DOCX,
PPTX, HTML, code, plain text). Pass `reset_state=False` to add to an existing index incrementally.

## Retrieve without generating

Use `retrieve()` when you only want the matching chunks (e.g. to build your own prompt):

```python
result = rag.retrieve("refund policy", top_k=5)
for doc in result.documents:
    print(doc)
```

Retrieval can be tuned per call: `top_k`, `metadata_filter`, `similarity_threshold`,
`retrieval_scope` (`"small"` … `"full"`), and `answer_breadth` (`"single_fact"` … `"summary"`).
`retrieval_mode="smart"` lets an LLM plan the retrieval (query rewrite, scope, breadth) before
searching.

## The building blocks

`RAGAgent` composes lower-level pieces you can also use directly (all in `neurosurfer.rag`):

- **`FileReader`** — reads a file/URL into text (PDF, DOCX, PPTX, HTML, code, …).
- **`Chunker`** — splits text into retrieval units (code-aware line chunking or character chunking,
  configured via `RAGIngestorConfig`).
- **`RAGIngestor`** — the ingestion pipeline (`add_files`, `add_directory`, `add_urls`,
  `add_git_folder`, `add_zipfile`, then `ingest()`); takes an `embedder` and a `vectorstore`.
- **`ContextBuilder`** — packs retrieved chunks into a token-aware context block for the prompt.

Configuration lives in `RAGIngestorConfig` (batch size, workers, dedup, chunking) and
`RAGAgentConfig` (retrieval defaults). See the [Vector stores](#vector-stores) below for storage
backends.

## Retrieval quality

Dense-only retrieval fails on the queries people actually type at a codebase or a docs corpus —
an error code, an identifier, a proper noun — because a model that has never seen `ENAMETOOLONG`
embeds it as noise while BM25 matches it exactly.

```python
RAGAgentConfig(
    hybrid_search=True,   # dense + BM25, fused by reciprocal rank
    mmr_lambda=0.5,       # diversity: stop one paragraph filling the window
)
```

Both are **off by default** — building the BM25 index is a full scan of the collection, and MMR
changes which chunks come back. Neither should start happening because you upgraded.

Measured on the fixture corpus in `tests/rag/`:

| | recall@1 | MRR | recall@3 | nDCG@3 |
|---|---|---|---|---|
| dense | 0.619 | 0.714 | 0.952 | 0.842 |
| hybrid | **0.905** | **1.000** | 0.952 | **0.966** |

The gain is in *ranking*, and at the small k a context window uses, ranking is recall. Dense search
usually has the right chunk somewhere in the top few; what it does badly is put it first.

### Reranking

An optional stage over the fused pool, usually the largest single quality gain:

```python
from neurosurfer.rag.retrieval import CrossEncoderReranker
RAGAgent(..., reranker=CrossEncoderReranker())
```

A cross-encoder reads query and document *together*, which is why reranking a top-50 usually beats
improving the retrieval that produced it. The order is fixed and deliberate — **fuse, then rerank,
then diversify**: reranking first wastes the expensive stage on documents fusion would have
dropped, and diversifying first lets the reranker reintroduce the redundancy just removed.

### Measuring your own corpus

```python
from neurosurfer.rag.evaluation import EvalCase, compare

cases = [EvalCase(query="how do I install", relevant_ids={"readme:0"})]
print(compare(cases, {"dense": dense_fn, "hybrid": hybrid_fn}, k=3))
```

`recall@k` asks whether the right chunk arrived at all; `MRR` and `nDCG@k` ask how well it was
ordered. Recall first — a chunk that never arrives cannot be reordered into place.

### Citations

`ContextBuilder.build_with_citations(docs)` returns the same context text plus a `Citation` per
rendered chunk, carrying `char_start` / `char_end` into the original document:

```python
text, citations = agent.ctx.build_with_citations(result.docs, result.distances)
citations[0].locator()   # 'guide.md:1200-1480'
```

Spans are recorded at ingestion by locating each chunk in its source. A chunker that *rewrites*
rather than slices yields no span, and a citation with no span still names its source — better than
a span that points at the wrong place.

## Retrieval shapes beyond classic

`neurosurfer.rag.strategies` holds four techniques, each answering a different failure of
"embed the chunk, take the top-k".

### Contextual retrieval

A chunk reading *"It returns None on failure"* is unfindable by a query naming the function,
because the chunk never says what "it" is. Splitting a document destroys the context that made
each part searchable, and no amount of better ranking recovers a term that is not there.

```python
from neurosurfer.rag.strategies import ContextualEnricher
from neurosurfer.rag.strategies.contextual import llm_summariser

enricher = ContextualEnricher(llm_summariser(provider))
chunks = enricher.enrich(chunks, document_text, source_id)
```

The summary is computed **once per document**, not per chunk — summarising per chunk is the
objection to this technique when it is implemented carelessly. A failing summariser falls back to
the source id rather than failing the ingest.

### Parent-document retrieval

Small chunks embed precisely and read poorly; large chunks the reverse. Index the children, return
the parents:

```python
from neurosurfer.rag.strategies import ParentDocumentRetriever
docs = ParentDocumentRetriever(parent_map).expand(child_hits)
```

Parents are de-duplicated — several children of one section routinely retrieve together, and
returning the parent three times fills the window with one passage repeated.

### Multi-query and HyDE

```python
from neurosurfer.rag.strategies import hyde_query, multi_query

queries = multi_query(provider, "how do I make it faster?", n=3)
text    = hyde_query(provider, "how do I make it faster?")
```

`multi_query` always keeps the original question first — a rewrite is a guess about what the asker
meant, and discarding the real question can lose an exact term they typed deliberately. `hyde_query`
embeds a *hypothetical answer* instead, because a question and its answer often share little
vocabulary while an invented answer and the real one share a lot. Both degrade to the plain query
if the model call fails.

### Sentence-window and semantic chunking

```python
from neurosurfer.rag.strategies.chunking import (
    make_semantic_handler, make_sentence_window_handler,
)
chunker.register_custom("semantic", make_semantic_handler(embedder))
chunker.use_custom_for_ext([".md", ".txt"], "semantic")
```

Sentence-window emits overlapping runs of sentences, so a fact and its qualifier survive together.
Semantic chunking cuts where the embedding distance between adjacent sentences is largest — where
the text changes subject — bounded by `min_sentences`/`max_sentences`, because distance alone gives
one-sentence chunks in dialogue and enormous ones in uniform prose.

## Re-ingesting only what changed

`RAGIngestor` deduplicates within a run and has no memory of the last one, so a directory of a
thousand files with one edit costs a thousand embeddings.

```python
from neurosurfer.rag.incremental import IngestManifest

manifest = IngestManifest.load(".neurosurfer/ingest.json")
delta = manifest.diff({source_id: text for ...})
print(delta)                       # '1 new, 1 changed, 998 unchanged, 0 removed'

store.delete_documents(manifest.stale_chunk_ids(delta))
for source_id in delta.to_ingest:
    ...                            # chunk, embed, add
    manifest.record(source_id, text, chunk_ids)
manifest.save()
```

The chunk ids matter as much as the hashes: when a source changes, its *old* chunks must go, and
without a record of which they were the only options are stale text in the index or clearing the
whole collection. A corrupt or missing manifest costs a full re-ingest, never a failed run.

## Vector stores

`neurosurfer.vectorstores` provides three backends behind the `BaseVectorDB` interface:

- **`ChromaVectorStore(collection_name, persist_directory=...)`** — persistent, disk-backed
  (requires the `rag` extra's `chromadb`).
- **`QdrantVectorStore(collection_name, dim, location=":memory:")`** — the strongest filtering,
  and the only backend that expresses the whole grammar. Runs in-process (`":memory:"`), embedded
  (a path), or against a server (a URL). Requires the `qdrant` extra.
- **`InMemoryVectorStore(dim=None)`** — ephemeral, dependency-free; handy for tests and demos.
  It is the reference implementation: `dim` is optional (the first document sets it, and later
  ones are checked against it).

All three are held to one **conformance suite** — `tests/vectorstores/conformance.py`. "Implements
`BaseVectorDB`" means "passes that suite", so adding a backend is one class and a three-line test
module. Qdrant passed it unmodified on the first run, which is the evidence that the interface is
a contract rather than a description of Chroma.

**Which to use.** Chroma if you want disk persistence with no service and no decisions; Qdrant if
you filter on ranges or negation, or want to move to a server later without changing your code;
InMemory for tests.

### What a store guarantees

- **`add_documents` upserts** on `Doc.id`, so re-ingesting a corpus is idempotent. A document
  with no id gets a stable one derived from its content.
- **`delete_documents(ids)`** takes ids; `delete_docs(docs)` is the convenience. Ids round-trip —
  what `list_all_documents()` returns can be deleted.
- **Scores are cosine similarity, higher is better**, whatever the backend's native metric.
  `similarity_threshold` is applied on that scale.

### Metadata filters

One grammar, `neurosurfer.vectorstores.filters`, which every backend translates:

```python
{"lang": "py"}                        # equals (shorthand)
{"lang": ["py", "rs"]}                 # one of  (shorthand)
{"score": {"$gte": 20, "$lt": 100}}    # ranges, ANDed
{"lang": "py", "kind": "src"}          # two fields, ANDed
{"$or": [{"lang": "py"}, {"kind": "test"}]}
{"$not": {"lang": "py"}}
```

Operators: `$eq`, `$ne`, `$in`, `$nin`, `$gt`, `$gte`, `$lt`, `$lte`, `$and`, `$or`, `$not`.
A field the metadata does not carry never matches — including under `$ne` and `$nin`.

### Capabilities

A backend declares what it can do, so you ask rather than infer:

```python
from neurosurfer.vectorstores import StoreCapability
StoreCapability.RANGE_FILTERS in store.capabilities
```

| Flag | Chroma | Qdrant | InMemory |
|---|:--:|:--:|:--:|
| `RANGE_FILTERS` | ✅ | ✅ | ✅ |
| `BOOLEAN_FILTERS` | ✅ | ✅ | ✅ |
| `NEGATION` (`$not`) | ❌ | ✅ | ✅ |
| `NATIVE_UPSERT` | ✅ | ✅ | ✅ |
| `PERSISTENT` | ✅ | ✅ | ❌ |

A filter needing a capability the store lacks raises `UnsupportedFilter` **before the query is
formed**, rather than returning rows it did not filter — which looks exactly like a working query.

!!! warning "Chroma collections created before this release scored wrongly"
    Chroma's default space is squared L2, and the old code returned `1.0 - distance` as though it
    were cosine — so orthogonal vectors scored **-1.0** instead of `0.0`, and any
    `similarity_threshold` was applied to the wrong scale. New collections are created as cosine;
    existing ones are read for the space they actually have and converted, so an old store now
    reports correct scores without being rebuilt.

## Embeddings

`neurosurfer.embeddings` resolves a **spec string** into a backend:

| Spec | Backend |
|---|---|
| `none` · `bm25` · `off` | `None` — use lexical search |
| `local` | sentence-transformers, default model |
| `intfloat/e5-small-v2` | sentence-transformers, that model |
| `openai` · `openai:text-embedding-3-large` | hosted OpenAI (needs `OPENAI_API_KEY`) |
| `openai-compat:<model>@<base_url>` | any server exposing `/v1/embeddings` |

```python
from neurosurfer.embeddings import get_embedder

# No torch, no sentence-transformers — the server you already run for chat.
emb = get_embedder("openai-compat:nomic-embed-text-v1.5@http://localhost:1234/v1")
vectors = emb.embed(["hello", "world"])
```

A bare string with no recognised prefix is a sentence-transformers model name, which is what it
meant before — existing configs keep working.

### `None` versus an exception

These are different failures and the difference is the point:

- **Not configured** — the optional dependency is missing, or no API key is set. `get_embedder`
  returns `None` and you fall back to lexical search.
- **Configured and broken** — a wrong model name, an expired key, an unreachable server. This
  **raises** (`EmbeddingError`), because returning `None` here turns *"your credentials lapsed"*
  into *"search quietly got worse"* with nothing said anywhere.

Pass `get_embedder(spec, degrade=True)` for the never-raises behaviour when you genuinely want it —
a background re-index that should limp rather than stop.

Requests are batched (`max_batch`) and retried with backoff on 429/5xx, using the same
retryable-error rules as chat completions.

### Which model wrote a collection

A store records the model and dimension it was embedded with, and refuses a query embedded by a
different one:

```
This collection was embedded with 'e5-small' and the query was embedded with
'nomic-embed'. Their vectors are not comparable — re-ingest the collection with
one model, or point at another.
```

An empty or unlabelled collection adopts the identity instead of refusing, so this never blocks a
first ingest or an upgrade from a store written before it existed.
