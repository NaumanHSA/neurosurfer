# In-Memory Vector Store

> Module: `neurosurfer.vectorstores.in_memory.InMemoryVectorStore`  
> Contract: [`BaseVectorDB`](./base-vectordb.md) · Data: [`Doc`](./base-vectordb.md#doc-dataclass)

A **lightweight**, dependency-free vector store that keeps everything in process memory. Best suited for **tests**, **demos**, and **small prototypes**.

---

## Features

- No external services; pure Python lists.
- **Cosine similarity** implementation (deterministic).
- Simple CRUD-style operations: `add_documents`, `similarity_search`, `count`, `list_all_documents`, `clear_collection`, `delete_collection`.

---

## Usage

```python
from neurosurfer.vectorstores.in_memory import InMemoryVectorStore
from neurosurfer.vectorstores.base import Doc
from neurosurfer.models.embedders.sentence_transformer import SentenceTransformerEmbedder

embedder = SentenceTransformerEmbedder(model_name="intfloat/e5-large-v2")
dim = len(embedder.embed("probe"))

vs = InMemoryVectorStore(dim=dim)

# Add docs (must provide embeddings of correct dimension)
docs = [
    Doc(id="d1", text="Vector search 101", embedding=embedder.embed("Vector search 101")),
    Doc(id="d2", text="Intro to embeddings", embedding=embedder.embed("Intro to embeddings")),
]
vs.add_documents(docs)

# Query
q = embedder.embed("learn about embeddings")
for doc, score in vs.similarity_search(q, top_k=2):
    print(f"{score:.3f} :: {doc.text}")
```

---

## API Notes

### `__init__(dim: int)`
Initializes storage for vectors of fixed `dim`ension.

### `add_documents(docs)`
- Validates each `Doc.embedding` is present and length == `dim`.

### `similarity_search(query_embedding, top_k=5, metadata_filter=None, similarity_threshold=None)`
- Computes cosine similarity with a simple dot/norm routine.
- Returns the top `k` `(Doc, score)` pairs.

### `list_all_documents(...)`, `count()`, `clear_collection()`, `delete_collection()`
- Straightforward in-memory implementations (no persistence).

---

## Tips

- Use this backend for **fast unit tests** and **local dev** where persistence isn’t required.
- For production or larger corpora, switch to a persistent store (e.g., **Chroma**) without changing caller code thanks to the shared `BaseVectorDB` contract.