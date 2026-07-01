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
- **`InMemoryVectorStore(dim)`** — ephemeral, dependency-free; handy for tests and demos.

## Embeddings

`neurosurfer.embeddings` exposes the `Embedder` protocol and `get_embedder(name)`, which loads a
named backend (e.g. a sentence-transformers model) and **degrades to lexical/BM25 search** (returns
`None`) if embeddings are unavailable — retrieval keeps working without a hard dependency.
