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

## Vector stores

`neurosurfer.vectorstores` provides two backends behind the `BaseVectorDB` interface:

- **`ChromaVectorStore(collection_name, persist_directory=...)`** — persistent, disk-backed
  (requires the `rag` extra's `chromadb`).
- **`InMemoryVectorStore(dim=None)`** — ephemeral, dependency-free; handy for tests and demos.
  It is the reference implementation: it supports every capability below, and `dim` is optional
  (the first document sets it, and later ones are checked against it).

Both are held to one **conformance suite** — `tests/vectorstores/conformance.py`. "Implements
`BaseVectorDB`" means "passes that suite", so adding a backend is one class and a three-line test
module.

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

| Flag | Chroma | InMemory |
|---|:--:|:--:|
| `RANGE_FILTERS` | ✅ | ✅ |
| `BOOLEAN_FILTERS` | ✅ | ✅ |
| `NEGATION` (`$not`) | ❌ | ✅ |
| `NATIVE_UPSERT` | ✅ | ✅ |
| `PERSISTENT` | ✅ | ❌ |

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
