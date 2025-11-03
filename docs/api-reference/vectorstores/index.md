# Vector Stores

Neurosurfer’s VectorDB layer provides a **unified interface** for storing and retrieving embeddings across different backends (e.g., Chroma, in-memory). Implementations share the same contract so you can **swap backends without changing app code**.

<div class="grid cards" markdown>

-   :material-database-outline:{ .lg .middle } **Base Concepts**

    ---

    Core contracts and data structures. Start here to understand the abstraction used by all backends.

    [:octicons-arrow-right-24: `BaseVectorDB`](./base-vectordb.md) [:octicons-arrow-right-24: `Doc`](./base-vectordb.md#doc-dataclass)

-   :material-database:{ .lg .middle } **Chroma**

    ---

    Persistent, production-ready store using `chromadb` with `PersistentClient`, metadata filtering, and similarity search.

    [:octicons-arrow-right-24: Documentation](./chroma.md)

-   :material-memory:{ .lg .middle } **In-Memory**

    ---

    Lightweight baseline store ideal for tests, demos, and small prototypes. Implements cosine similarity over Python lists.

    [:octicons-arrow-right-24: Documentation](./in_memory.md)

</div>

---

## ⚡ Quick Example

```python
from neurosurfer.vectorstores import ChromaVectorStore
from neurosurfer.vectorstores.base import Doc
from neurosurfer.models.embedders.sentence_transformer import SentenceTransformerEmbedder

# Create vector store and embedder
vectorstore = ChromaVectorStore(
    collection_name="my_docs",
    persist_directory="./chroma_db"
)
embedder = SentenceTransformerEmbedder(model_name="intfloat/e5-large-v2")

# Create and add documents
docs = [
    Doc(id="1", text="Document 1", 
        embedding=embedder.embed("Document 1"),
        metadata={"source": "doc1.txt"}),
    Doc(id="2", text="Document 2", 
        embedding=embedder.embed("Document 2"),
        metadata={"source": "doc2.txt"})
]
vectorstore.add_documents(docs)

# Search (requires embedding the query)
query_embedding = embedder.embed("query")
results = vectorstore.similarity_search(query_embedding, top_k=5)

for doc, score in results:
    print(f"[{score:.3f}] {doc.text}")
```

> Tip: When using the ingestion pipeline, see **RAG Ingestor** for batching, deduplication, and automatic ID strategy. The vector store API is intentionally minimal to keep backends interchangeable.