# RAG Examples

This page gives a practical, end‑to‑end tour of Neurosurfer’s **RAG building blocks**: the `Chunker` (structure‑aware document splitting) and the `RAGIngestor` (read → chunk → embed → store). You’ll find copy‑pasteable snippets for common workflows: chunking files, registering custom strategies, ingesting files/directories/raw text/ZIPs/URLs, retrieving top‑K matches, and wiring progress/cancellation.

---

## 🧩 Chunker — Overview

The `Chunker` intelligently splits content into semantically meaningful chunks while preserving structure. It supports multiple strategies out of the box and chooses an approach based on file type and heuristics:

- Python: **AST‑aware** chunking
- JS/TS/React: **structure‑aware** chunking
- Markdown/Text: **header/section aware** for docs, line/char for prose
- JSON: **object/array aware** chunking
- Comment‑aware filtering, configurable overlaps
- Extensible via **custom strategies** and **handlers**

> Internally, it uses configuration from `config.chunker` (overlap, sizes, etc.).

### Quick start

```python
from neurosurfer.rag.chunker import Chunker

chunker = Chunker()

code = '''
def area(r):
    # Compute circle area
    return 3.14159 * r * r
'''

chunks = chunker.chunk(code, file_path="utils.py")
for i, ch in enumerate(chunks, 1):
    print(f"[{i}] {ch[:80]}")
```

### Register a custom strategy (by extension)

Use a simple function `(text, file_path) -> List[str]` and map it to one or more extensions.

```python
from neurosurfer.rag.chunker import Chunker

def my_double_newline(text: str, file_path: str | None = None):
    # Split on blank lines; trim tiny fragments
    parts = [p.strip() for p in text.split("\n\n")]
    return [p for p in parts if len(p) >= 20]

chunker = Chunker()
chunker.register({".custom", ".note"}, my_double_newline)

sample = "A block...\n\nAnother block...\n\nShort\n"
print(chunker.chunk(sample, file_path="notes.custom"))
```

### Custom handler (full control)

Handlers can accept `config` and other meta; useful for advanced routing or parameterized chunking.

```python
from typing import Optional, List
from neurosurfer.rag.chunker import Chunker, ChunkerConfig

def my_handler(text: str, *, file_path: Optional[str] = None, config: Optional[ChunkerConfig] = None) -> List[str]:
    # Example: fixed-size char windows with slight overlap from cfg
    size = (config.char_chunk_size if config else 800)
    overlap = (config.char_overlap if config else 60)
    out = []
    i = 0
    while i < len(text):
        out.append(text[i:i+size])
        i += max(1, size - overlap)
    return out

chunker = Chunker()
# Register a named handler and bind it to an extension (optional)
chunker.register_handler("wide_chars", my_handler)
chunker.map_extension_to_handler(".log", "wide_chars")

log_text = "..."  # long logs
print(len(chunker.chunk(log_text, file_path="app.log")))
```

> If you only need simple extension→function mapping, `register({'.ext'}, fn)` is enough. Use handlers when you need access to the `ChunkerConfig`.

---

## 📥 RAG Ingestor — Overview

`RAGIngestor` is the production‑grade ingestion pipeline: read files/directories/raw text/URLs/ZIPs → **chunk** → **embed** (batch) → **dedupe** → **persist** to a vector DB.

**Key features**: multi‑source input, parallel processing, progress callbacks, cancellation, content‑hash dedupe, and metadata preservation.

Typical setup:

```python
from neurosurfer.models.embedders.sentence_transformer import SentenceTransformerEmbedder
from neurosurfer.vectorstores import ChromaVectorStore   # or any BaseVectorDB implementation
from neurosurfer.rag.ingestor import RAGIngestor

embedder = SentenceTransformerEmbedder("intfloat/e5-small-v2")
vs = ChromaVectorStore(collection_name="docs")  # must implement BaseVectorDB API

ingestor = RAGIngestor(
    embedder=embedder,
    vector_store=vs,
    batch_size=64,
    max_workers=4,
    deduplicate=True,
    normalize_embeddings=True,
)
```

### Add sources

```python
# 1) Add individual files
ingestor.add_files(["README.md", "guide.md", "src/app.py"])

# 2) Recursively add a directory (skips common junk like node_modules, .git, etc.)
ingestor.add_directory("./docs")

# 3) Raw texts (with optional per‑item metadata)
ingestor.add_texts(
    ["Custom paragraph 1", "Custom paragraph 2"],
    base_id="manual",
    metadatas=[{"section": "intro"}, {"section": "notes"}],
)

# 4) ZIP archive (safe extraction to a temp folder, then indexed like a dir)
ingestor.add_zipfile("./handbook.zip")

# 5) URLs (requires a fetcher; here’s a tiny example)
def fetch(url: str) -> str | None:
    # return cleaned text from URL (left as an exercise: requests + readability/bs4)
    return None

ingestor.add_urls(["https://example.com/page1"], fetcher=fetch)
```

### Build / ingest

```python
stats = ingestor.build()
print(stats)  # {'status': 'ok', 'sources': ..., 'chunks': ..., 'unique_chunks': ..., 'added': ...}
```

#### With progress callback and cancellation

```python
import threading, time

progress = []
def on_progress(p):
    progress.append(p)
    if p.get("stage") == "embedding":
        print(f"Embedding {p['embedded']}/{p['total']}...")

ingestor = RAGIngestor(
    embedder=embedder,
    vector_store=vs,
    batch_size=64,
    max_workers=4,
    progress_cb=on_progress,
)

# Run build in a background thread
th = threading.Thread(target=lambda: ingestor.build(), daemon=True)
th.start()

# Cancel after a moment (simulate user clicking 'Stop')
time.sleep(0.2)
ingestor.cancel_event.set()
th.join()
```

### Retrieve content (two ways)

**A) Via `RAGIngestor` helpers**

```python
hits = ingestor.search("what is the ingestion pipeline?", top_k=5)
for doc, score in hits:
    print(f"{score:.3f} | {doc.metadata.get('filename', doc.id)}")
    print(doc.text[:120], "...
")
```

**B) Directly via the vector store**

```python
# Prepare a query embedding
q = ingestor.embed_query("how do we chunk python code?")
# Use store's similarity_search with the query vector
matches = vs.similarity_search(q, top_k=5)
for doc, score in matches:
    print(doc.id, score)
```

---

## 🧪 End‑to‑end mini pipeline

```python
from neurosurfer.models.embedders.sentence_transformer import SentenceTransformerEmbedder
from neurosurfer.vectorstores import ChromaVectorStore
from neurosurfer.rag.ingestor import RAGIngestor

# 1) Components
emb = SentenceTransformerEmbedder("intfloat/e5-small-v2")
store = ChromaVectorStore(collection_name="proj-docs")
ing = RAGIngestor(embedder=emb, vector_store=store, batch_size=48, max_workers=4)

# 2) Intake
ing.add_directory("./docs")
ing.add_files(["README.md", "CHANGELOG.md"])
ing.add_texts(["This is a private note about deployment steps."], base_id="notes")

# 3) Build
summary = ing.build()
print("Added:", summary["added"])

# 4) Retrieve
for doc, score in ing.search("deployment steps", top_k=3):
    print(f"{score:.2f} | {doc.metadata.get('filename', doc.id)}")
    print(doc.text[:160], "...\n")
```

---

## ✅ Tips

- Prefer **batch sizes** of 32–128 for sentence‑transformers to balance throughput vs memory.
- Use **deduplication** (default on) when indexing mixed sources to avoid repeated chunks.
- Add `default_metadata` to `RAGIngestor(...)` to stamp common fields across all docs.
- Mind your **overlaps** in chunking — larger overlaps improve recall at a small cost.
- For very large repos, start with **directory filters** and a tighter set of `include_exts`.