# RAG System API

Neurosurfer’s Retrieval-Augmented Generation (RAG) system turns heterogeneous sources (files, directories, raw text, URLs, ZIPs) into **chunk-level embeddings** that your LLM can ground on during inference. It is modular, safe, and extensible:

- **FileReader** – normalizes inputs by detecting type/extension and returning UTF-8 text for supported formats; skips binaries and common build/VCS artifacts.
- **Chunker** – converts long text into **context-preserving** chunks using structure-aware strategies (e.g., Python AST, Markdown headers, JSON objects) with configurable line/character fallbacks and overlap.
- **RAGIngestor** – orchestrates: **queue → chunk (parallel) → dedupe (content hash) → batch-embed → upsert to vector store**, preserving metadata (filename, url, source_type, content_hash) and supporting progress callbacks & cancellation.

---

## Components

<div class="grid cards" markdown>

-   :material-content-cut:{ .lg .middle } **Chunker**

    ---

    Split text into context‑preserving chunks (AST/structure‑aware where possible)

    [:octicons-arrow-right-24: Documentation](chunker.md)

-   :material-file-document:{ .lg .middle } **FileReader**

    ---

    Read various file formats into normalized UTF‑8 text

    [:octicons-arrow-right-24: Documentation](filereader.md)

-   :material-database-import:{ .lg .middle } **RAGIngestor**

    ---

    Queue sources → chunk → batch‑embed → dedupe → persist to vector store

    [:octicons-arrow-right-24: Documentation](ingestor.md)

</div>

---

## Minimal Usage Examples

### Chunker

```python
from neurosurfer.rag.chunker import Chunker

text = """# Title

Paragraph one.

## Section
More text here."""

chunks = Chunker().chunk(text, file_path="notes.md")
print(len(chunks), "chunks")  # e.g., 2–5 depending on config
```

### FileReader

```python
from neurosurfer.rag.filereader import FileReader

reader = FileReader()
txt = reader.read("./document.pdf")   # returns extracted text (or "")
print(txt[:200])
```

### RAGIngestor

```python
from neurosurfer.rag.ingestor import RAGIngestor
from neurosurfer.rag.chunker import Chunker
from neurosurfer.rag.filereader import FileReader
from neurosurfer.vectorstores.chroma import ChromaVectorStore
from neurosurfer.models.embedders.sentence_transformer import SentenceTransformerEmbedder

embedder = SentenceTransformerEmbedder("intfloat/e5-small-v2")
vectorstore = ChromaVectorStore(collection_name="docs")

ingestor = RAGIngestor(embedder=embedder, vector_store=vectorstore, chunker=Chunker(), file_reader=FileReader())
ingestor.add_files(["./document.pdf", "./notes.md"]).build()

# Quick retrieval smoke test
for doc, score in ingestor.search("ingestion pipeline", top_k=3):
    print(f"{score:.3f} | {doc.metadata.get('filename', doc.id)}")
```

---
