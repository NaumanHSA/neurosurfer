# RAG Ingestor

> Module: `neurosurfer.rag.ingestor`  
> Depends on:
> - Reader: [`FileReader`](./filereader.md)
> - Chunking: [`Chunker`](./chunker.md)
> - Embeddings: **BaseEmbedder** ([Embedders](../models/embedders/base-embedder.md))
> - Vector DB: **BaseVectorDB** ([Vector Stores](../vectorstores/base-vectordb.md))

A production-grade **end-to-end ingestion pipeline** for RAG systems. `RAGIngestor` reads sources (files, folders, URLs, ZIPs, raw text), **chunks** them, **embeds** in batches, performs **content-hash deduplication**, and **indexes** into a vector store—with **progress callbacks**, **cancellation**, and **parallelism** built in.

---

## Features & Flow

`RAGIngestor` is a full-stack ingestion engine: it accepts **many source types** (files, directories, raw text, URLs via your fetcher, Git folders, ZIPs), runs **parallel chunking** (configurable `max_workers`), performs **batch embedding** (tuned by `batch_size` with optional vector normalization), and enforces **content deduplication** (SHA-256 over chunk text). Throughout the run it emits **structured progress events**, supports **cooperative cancellation** (`threading.Event`), **preserves metadata** (e.g., filename/URL/ZIP + your custom fields), and handles archives with **zip-slip–safe extraction** and defensive error handling.

### Typical flow:

1. **Queue sources** with `add_files`, `add_directory`, `add_texts`, `add_urls`, `add_git_folder`, or `add_zipfile`.
2. **Build** with `build()` which internally:
   - **Reads** content via [FileReader](./filereader.md)  
   - **Chunks** text via [Chunker](./chunker.md)  
   - **Deduplicates** on SHA-256 of chunk text  
   - **Embeds** using your **BaseEmbedder**  
   - **Indexes** into your **BaseVectorDB**
3. **Optionally verify** the index with `embed_query` and `search`.


---

## Constructor

```python
from neurosurfer.rag.ingestor import RAGIngestor
from neurosurfer.rag.filereader import FileReader
from neurosurfer.rag.chunker import Chunker
from neurosurfer.models.embedders import BaseEmbedder
from neurosurfer.vectorstores.base import BaseVectorDB

ingestor = RAGIngestor(
    embedder: BaseEmbedder,                 # required
    vector_store: BaseVectorDB,             # required
    file_reader: FileReader | None = None,  # default FileReader()
    chunker: Chunker | None = None,         # default Chunker()
    logger: logging.Logger | None = None,
    progress_cb: callable | None = None,    # def cb(d: dict) -> None
    cancel_event: threading.Event | None = None,
    batch_size: int = 64,
    max_workers: int = max(4, os.cpu_count() or 4),
    deduplicate: bool = True,
    normalize_embeddings: bool = True,
    default_metadata: dict | None = None,
    tmp_dir: str | None = None,             # defaults to "./tmp"
)
```

### Parameters

| Name | Type | Default | Description |
|---|---|---:|---|
| `embedder` | `BaseEmbedder` | — | Embedding backend used for `embed(texts, normalize_embeddings=...)`. |
| `vector_store` | `BaseVectorDB` | — | Vector DB with `add_documents(docs)` and `similarity_search(vec, top_k)`. |
| `file_reader` | `FileReader` | `FileReader()` | File → text loader. See [FileReader](./filereader.md). |
| `chunker` | `Chunker` | `Chunker()` | Text chunker. See [Chunker](./chunker.md). |
| `logger` | `logging.Logger` | module logger | Target for exceptions/warnings. |
| `progress_cb` | `Callable[[dict], None]` | `None` | Receives progress events (see **Progress Events**). |
| `cancel_event` | `threading.Event` | new event | When set, long steps return early with `"status": "cancelled"`. |
| `batch_size` | `int` | `64` | Texts per embed call. Tune to GPU/CPU throughput. |
| `max_workers` | `int` | `max(4, os.cpu_count() or 4)` | Thread workers for concurrent chunking. |
| `deduplicate` | `bool` | `True` | Drop duplicate chunk texts (by SHA-256). |
| `normalize_embeddings` | `bool` | `True` | Unit-normalize vectors before indexing (cosine friendly). |
| `default_metadata` | `dict` | `{}` | Base metadata merged into every queued record. |
| `tmp_dir` | `str` | `"./tmp"` | Working directory used for ZIP extraction, etc. |

---

## Input Queueing Methods

All queueing methods return `self` to enable **builder-style** chaining. Internal queue shape:  
`self._queue: List[Tuple[source_id: str, text: str, metadata: dict]]`

### `add_files(paths, include_exts=supported_file_types, *, root=None, extra_metadata=None)`

Reads individual files by extension and enqueues them.

- `paths`: `Sequence[str | Path]`
- `include_exts`: set of allowed extensions (defaults to `rag.constants.supported_file_types`)
- `root`: optional root to compute relative `filename` metadata
- `extra_metadata`: merged with `default_metadata`

**Metadata added**: `{"filename": <relative-or-absolute>, "source_type": "file"}`

```python
ingestor.add_files(["/data/README.md", "/data/notes.txt"], root="/data")
```

### `add_directory(directory, *, include_exts=supported_file_types, exclude_dirs=None, extra_metadata=None)`

Recursively walks a directory, applying `include_exts` and excluding common junk (default excludes: `.git`, `__pycache__`, `.venv`, `node_modules`, `dist`, `build`).

**Metadata added**: `{"filename": <relative-path>, "source_type": "file"}`

```python
ingestor.add_directory("./docs")
```

### `add_texts(texts, *, base_id="text", metadatas=None)`

Enqueue **raw text strings** (already in memory).

- `texts`: sequence of strings
- `base_id`: used to form `source_id` as `"{base_id}:{i}"`
- `metadatas`: optional list of dicts; aligned with `texts`

**Metadata added**: `{"source_type": "raw_text", "ordinal": i}`

```python
ingestor.add_texts(
    ["First text", "Second text"],
    metadatas=[{"tag": "a"}, {"tag": "b"}]
)
```

### `add_urls(urls, fetcher=None, extra_metadata=None)`

Add **URLs** using a user-provided `fetcher(url) -> str | None` that returns **clean text**. (This keeps the ingestor offline-friendly and testable.)

**Metadata added**: `{"url": url, "source_type": "url"}`

```python
import requests
from bs4 import BeautifulSoup

def fetcher(url: str) -> str | None:
    r = requests.get(url, timeout=20)
    soup = BeautifulSoup(r.text, "html.parser")
    return soup.get_text(separator="\n").strip()

ingestor.add_urls(["https://example.com"], fetcher=fetcher)
```

### `add_git_folder(repo_root, *, include_exts=None, extra_metadata=None)`

Index an **already-cloned** repository folder using the code-friendly exclude set from `rag.constants.exclude_dirs_in_code`. Internally delegates to `add_directory(...)`.

```python
ingestor.add_git_folder("/path/to/repo")
```

### `add_zipfile(zip_path, *, include_exts=supported_file_types, exclude_dirs=exclude_dirs_in_code, extra_metadata=None)`

Safely extracts a **single** `.zip` file into a temporary directory under `tmp_dir`, indexes it via `add_directory`, then deletes the temp directory (context manager). Protected against **zip-slip**.

**Metadata added** (in addition to others): `{"source_zip": "<zipname>"}`

```python
ingestor.add_zipfile("archive.zip", extra_metadata={"source": "upload"})
```

---

## Build / Ingest

### `build() -> dict`

Runs the full pipeline for all queued inputs:

1. **Chunk** (concurrent): convert `(source_id, text, metadata)` → `(source_id, chunk_text, metadata)`  
   - Uses [`Chunker`](./chunker.md).  
   - Thread-pooled via `ThreadPoolExecutor(max_workers=self.max_workers)`.

2. **Deduplicate**: drop duplicate `chunk_text` by SHA-256 hash, stored in `self._seen_hashes` (if `deduplicate=True`).  
   - Adds `"content_hash"` to chunk metadata.

3. **Embed** (batched): call `embedder.embed(texts, normalize_embeddings=self.normalize_embeddings)` in batches of `batch_size`.

4. **Index**: Create `Doc` records and call `vector_store.add_documents(docs)`.

5. **Done**: return stats dict (see **Return** below).

#### Return
```json
{
  "status": "ok",
  "sources": <int>,
  "chunks": <int>,
  "unique_chunks": <int>,
  "added": <int>,
  "finished_at": <unix_ts_float>
}
```
If cancelled mid-run: `{"status": "cancelled", ...}` with partial counts.

#### Document IDs

For each chunk, a stable `doc_id` is computed as:
```
doc_id = f"{content_hash[:16]}:{sha256(source_id)[:8]}"
```
This makes duplicates (same text) naturally collide while preserving source identity component.

---

## Progress Events

If `progress_cb` is provided, it receives dicts during `build()` like:

```json
{"stage": "start",    "queued_sources": 42}
{"stage": "chunking", "completed_sources": 10, "total_sources": 42, "progress_pct": 23}
{"stage": "dedupe",   "before": 1200, "after": 985}
{"stage": "embedding","embedded": 256, "total": 985}
{"stage": "done",     "added": 985}
```

> Use this to update UIs, logs, or metrics dashboards.

---

## Cancellation

Set `cancel_event` (a `threading.Event`) from another thread to **cooperatively stop** the ingestion. The ingestor checks it between major steps and inside the embed loop; on detection it returns a `"status": "cancelled"` summary payload.

```python
from threading import Event
stop = Event()

ingestor = RAGIngestor(..., cancel_event=stop)
# In another thread:
stop.set()
```

---

## Quick Start (End-to-End)

```python
from threading import Event
from neurosurfer.rag.ingestor import RAGIngestor
from neurosurfer.rag.filereader import FileReader
from neurosurfer.rag.chunker import Chunker
from neurosurfer.models.embedders.sentence_transformer import SentenceTransformerEmbedder
from neurosurfer.vectorstores.chroma import ChromaDB

stop = Event()

def progress(p):
    print(p)

ingestor = RAGIngestor(
    embedder=SentenceTransformerEmbedder("all-MiniLM-L6-v2"),
    vector_store=ChromaDB(collection="neurosurfer"),
    file_reader=FileReader(),
    chunker=Chunker(),
    progress_cb=progress,
    cancel_event=stop,
    batch_size=64,
    max_workers=4,
    deduplicate=True,
    normalize_embeddings=True,
    default_metadata={"project": "Neurosurfer"}
)

(ingestor
    .add_directory("./docs")
    .add_files(["README.md"])
    .add_texts(["Custom note"], metadatas=[{"source": "manual"}])
)

stats = ingestor.build()
print("Build stats:", stats)

# Smoke test:
print(ingestor.search("How do I run the server?", top_k=3))
```

---

## Vector Store Expectations

`BaseVectorDB` must provide:

- `add_documents(docs: list[Doc]) -> None`  
  Where `Doc` has: `id: str`, `text: str`, `embedding: list[float] | None`, `metadata: dict`.

- `similarity_search(query_vec: list[float], top_k: int = 5) -> list[tuple[Doc, float]]`  
  Returns `(Doc, score)` pairs (higher score = more similar).

See [Vector Stores](../vectorstores/base-vectordb.md).

---

## Embedder Expectations

`BaseEmbedder` must provide:

- `embed(texts: list[str], normalize_embeddings: bool = True) -> list[list[float]]`

If `normalize_embeddings=True`, return **unit vectors** (cosine similarity friendly). See [Embedders](../models/embedders/base-embedder.md).

---

## Performance Tuning & Tips

- **Batch size**: start with `64` for CPU sentence transformers; raise for GPUs until you saturate memory/throughput.
- **Workers**: chunking is CPU-bound but cheap; `max_workers=4–16` works well for large corpora.
- **Dedup**: keep enabled; it can reduce storage and query noise significantly for codebases & docs.
- **Metadata discipline**: include `filename`, `url`, `source_zip`, `commit`, `page`/`sheet` where available.
- **Normalization**: keep `normalize_embeddings=True` if your store uses cosine distance.
- **Backpressure**: vector stores sometimes perform better with **smaller bulks**; split `docs_to_add` if needed.

---

## Troubleshooting

- **My build crashes while embedding** → Wrap/inspect `embedder.embed`; verify VRAM/num threads; check `batch_size`.
- **I see many duplicate chunks** → Confirm `deduplicate=True`; ensure preprocessing (e.g., `Chunker` filters) is enabled.
- **ZIP ingestion missed files** → Check `include_exts` and `exclude_dirs`; verify `supported_file_types` in `rag.constants`.
- **Slow PDF extraction** → Consider pre-processing/cleaning PDFs, or adding an OCR path outside this module.
- **Error strings embedded** → `FileReader` returns `"Error reading ..."` text; filter these upstream (skip on `"Error reading"` prefix).

---

## Reference: Helper Functions

- `sha256_text(s: str) -> str` — UTF‑8 SHA‑256 hex digest for dedupe & IDs.  
- `now_ts() -> float` — `time.time()` convenience for `finished_at`.

---

## Internals (for contributors)

- Queue structure: `self._queue: List[Tuple[str, str, dict]]`
- Dedup state: `self._seen_hashes: set[str]`
- Temp workspace: `self.tmp_dir` used by `add_zipfile` (auto-created)
- Safe ZIP extraction mitigates **zip-slip** by validating paths against the temp root before writing.