# Chroma Vector Store

> Module: `neurosurfer.vectorstores.chroma.ChromaVectorStore`  
> Contract: [`BaseVectorDB`](./base-vectordb.md) · Data: [`Doc`](./base-vectordb.md#doc-dataclass)

Chroma-backed vector store using `chromadb.PersistentClient`. Provides **persistent storage**, **metadata filtering**, and **similarity search** (returns cosine similarity).

---

## Features

- Persistent collections on disk (`persist_directory`).
- `add_documents` with **upsert** (or `delete`+`add` depending on Chroma version).
- `similarity_search` with optional `metadata_filter` and `similarity_threshold`.
- `list_all_documents` with embeddings, metadatas, and texts.
- `count`, `clear_collection`, `delete_documents`, `delete_collection`.

---

## Usage

```python
from neurosurfer.vectorstores import ChromaVectorStore
from neurosurfer.vectorstores.base import Doc
from neurosurfer.models.embedders.sentence_transformer import SentenceTransformerEmbedder

vs = ChromaVectorStore(collection_name="my_docs", persist_directory="./chroma_db")
embedder = SentenceTransformerEmbedder(model_name="intfloat/e5-large-v2")

# Add docs (ensure embeddings are same dimension across all docs)
docs = [
    Doc(id="a1", text="Neural search is fun", 
        embedding=embedder.embed("Neural search is fun"),
        metadata={"topic": "search", "source": "notes.md"}),
    Doc(id="b2", text="Transformers are powerful", 
        embedding=embedder.embed("Transformers are powerful"),
        metadata={"topic": "nlp"}),
]
vs.add_documents(docs)

# Query
q = embedder.embed("power of transformers")
hits = vs.similarity_search(q, top_k=3, metadata_filter={"topic": "nlp"}, similarity_threshold=0.55)
for doc, score in hits:
    print(f"{score:.3f} :: {doc.metadata.get('topic')} :: {doc.text[:60]}")
```

---

## API Notes

### `__init__(collection_name: str, persist_directory: str = "chroma_storage")`
Creates/opens a persistent collection. Internally uses `chromadb.PersistentClient(path=...)` and `get_or_create_collection(name=...)`.

### `add_documents(docs: list[Doc]) -> None`
- Builds lists of `ids`, `documents`, `embeddings`, `metadatas`.
- Uses `collection.upsert(...)` if available, else falls back to `delete(...)` + `add(...)` (supports older Chroma versions).
- IDs default to `Doc.id` or fall back to `BaseVectorDB._stable_id(doc)` if missing.

### `similarity_search(query_embedding, top_k=20, metadata_filter=None, similarity_threshold=None) -> list[tuple[Doc, float]]`
- Applies **metadata filters** via Chroma’s `where` clause (values or `$in` for lists).
- Fetches up to `top_k * 2` raw hits, then deduplicates and trims to `top_k`.
- Converts **Chroma distance** to **cosine similarity** as `1.0 - distance`.
- If `similarity_threshold` is provided, drops hits below it.

### `list_all_documents(metadata_filter=None) -> list[Doc]`
- Returns `Doc` objects with `text`, `embedding`, `metadata`.
- Uses `collection.get(include=["documents", "metadatas", "embeddings"])`.

### `clear_collection() / delete_collection() / delete_documents(ids)`
- `clear_collection()` recreates the collection (handy for tests).
- `delete_collection()` deletes and nulls the handle (recreate as needed).
- `delete_documents(ids)` removes specific records by ID.

---

## Tips & Troubleshooting

- **Embeddings**: You are responsible for generating embeddings **with a consistent dimension** per collection.
- **Thresholding**: Tune `similarity_threshold` (as cosine similarity) to filter noisy hits.
- **Filters**: For multi-value filters, pass lists (the store converts to `$in`). Example: `{"topic": ["nlp", "search"]}`.
- **Persistence**: Ensure `persist_directory` is writable and not on a volatile mount if you expect durability.