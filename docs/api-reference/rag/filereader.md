# FileReader

> Module: `neurosurfer.rag.filereader`

A unified, production‑grade **file → text** loader for RAG pipelines. `FileReader` auto‑detects the file type by extension and applies the appropriate extractor to return **clean UTF‑8 text** that’s ready for downstream chunking ([Chunker](./chunker.md)) and ingestion ([RAG Ingestor](./ingestor.md)). It is defensive by design: optional dependencies are handled gracefully and errors are returned as descriptive strings instead of crashing your pipeline.

---

## Overview

`FileReader` exposes a single high‑level method, `read(path)`, which dispatches to format‑specific readers. It supports documents, data files, presentations, code/config/log formats, HTML pages, and more. When a format is not explicitly supported, it falls back to **plain‑text** reading with UTF‑8 (and `errors="ignore"`).

---

## Key Capabilities

- **Auto‑detection by extension** via `supported_types` mapping.
- **Broad format coverage** out of the box (see table below).
- **Graceful degradation** when optional libraries are missing (returns helpful messages).
- **Consistent plaintext output** suitable for embedding + retrieval.
- **Zero surprises**: reader methods never raise; you get text (or an “Error reading …” string).

---

## Supported Formats & Readers

| Category | Extensions | Reader method | Dependencies |
|---|---|---|---|
| **PDF** | `.pdf` | `_read_pdf` | `fitz` (PyMuPDF) |
| **HTML** | `.html`, `.htm` | `_read_html` | `bs4` (BeautifulSoup) |
| **DOCX** | `.docx` | `_read_docx` | `python-docx` |
| **CSV / TSV** | `.csv`, `.tsv` | `_read_csv` | `pandas` |
| **Excel** | `.xls`, `.xlsx` | `_read_excel` | `pandas` |
| **YAML** | `.yaml`, `.yml` | `_read_yaml` | `pyyaml` *(optional)* |
| **XML** | `.xml` | `_read_xml` | `xml.etree.ElementTree` (stdlib) |
| **PPTX** | `.pptx` | `_read_pptx` | `python-pptx` *(optional)* |
| **Plain‑text family** | `.txt`, `.md`, `.rtf`, `.doc`, `.odt`, `.json`, `.ppt`, `.py`, `.ipynb`, `.java`, `.js`, `.ts`, `.jsx`, `.tsx`, `.cpp`, `.c`, `.h`, `.cs`, `.go`, `.rb`, `.rs`, `.php`, `.swift`, `.kt`, `.sh`, `.bat`, `.ps1`, `.scala`, `.lua`, `.r`, `.env`, `.ini`, `.toml`, `.cfg`, `.conf`, `.properties`, `.log`, `.tex`, `.srt`, `.vtt` | `_read_txt` | none |

> Anything not in `supported_types` also falls back to `_read_txt` (UTF‑8, `errors="ignore"`).

---

## Dependencies

- **Required core libs (used if format encountered):**  
  - `fitz` (PyMuPDF) for PDF  
  - `docx` (python‑docx) for DOCX  
  - `pandas` for CSV/TSV/Excel  
  - `bs4` (BeautifulSoup) for HTML
- **Optional:**  
  - `pyyaml` for YAML (`yaml.safe_load`)  
  - `python-pptx` for PPTX  
  - `xml.etree.ElementTree` is from the standard library

When an optional dependency is missing, the corresponding reader returns a clear message (e.g., `"python-pptx not installed"`).

---

## Public API

### `class FileReader`

#### Attributes
- `supported_types: dict[str, Callable]` – mapping from extension (lowercase, with dot) to the concrete reader method.

#### Methods
- `read(file_path: str) -> str`  
  Auto‑detects by extension and dispatches to a concrete `_read_*` method. If no handler is registered, uses `_read_txt`. **Never raises**; errors are returned as readable strings.

##### Format‑specific readers
- `_read_pdf(path: str) -> str`  
  Page‑wise text extraction using `fitz`. On error returns `"Error reading PDF: <message>"`.
- `_read_txt(path: str) -> str`  
  Reads UTF‑8 with `errors="ignore"`. On error returns `"Error reading TXT: <message>"`.
- `_read_html(path: str) -> str`  
  Parses with `BeautifulSoup(..., "html.parser")` and returns visible text via `.get_text()`. On error returns `"Error reading HTML: <message>"`.
- `_read_docx(path: str) -> str`  
  Iterates `doc.paragraphs`, joins with newlines. On error returns `"Error reading DOCX: <message>"`.
- `_read_excel(path: str) -> str`  
  Uses `pandas.read_excel(..., sheet_name=None)` to load all sheets; renders with `.astype(str).to_string(index=False)` and sheet headers. On error returns `"Error reading Excel: <message>"`.
- `_read_csv(path: str) -> str`  
  Uses `pandas.read_csv`, stringifies and returns `.to_string(index=False)`. On error returns `"Error reading CSV/TSV: <message>"`.
- `_read_yaml(path: str) -> str`  
  Uses `yaml.safe_load` if `pyyaml` is available; otherwise returns `"PyYAML not installed"`. On error returns `"Error reading YAML: <message>"`.
- `_read_xml(path: str) -> str`  
  Uses `xml.etree.ElementTree.parse(...).getroot()` and `ET.tostring(..., encoding="unicode")`. If unavailable, returns `"XML parser not available"`. On error returns `"Error reading XML: <message>"`.
- `_read_pptx(path: str) -> str`  
  Uses `Presentation(path)`; concatenates `shape.text` for all shapes across slides. If library is missing, returns `"python-pptx not installed"`. On error returns `"Error reading PPTX: <message>"`.

---

## Behavior & Error Model

- **Non‑throwing**: all readers catch exceptions and return `"Error reading <FORMAT>: <message>"` to prevent ingestion crashes. You may choose to **skip** these records upstream.
- **Encoding**: text reading uses UTF‑8 with `errors="ignore"` to maximize robustness.
- **Best‑effort structure**: dataframes, DOCX paragraphs, and PPTX shape texts are stringified in a predictable, readable way.

---

## Usage Examples

### Basic

```python
from neurosurfer.rag.filereader import FileReader

reader = FileReader()

pdf_text = reader.read("report.pdf")
excel_text = reader.read("dataset.xlsx")
code_text = reader.read("script.py")
html_text = reader.read("page.html")
```

### With Chunker & Ingestor

```python
from neurosurfer.rag.filereader import FileReader
from neurosurfer.rag.chunker import Chunker
from neurosurfer.rag.ingestor import RAGIngestor
from neurosurfer.models.embedders.sentence_transformer import SentenceTransformerEmbedder
from neurosurfer.vectorstores.chroma import ChromaDB

reader = FileReader()
chunker = Chunker()
ingestor = RAGIngestor(
    embedder=SentenceTransformerEmbedder("all-MiniLM-L6-v2"),
    vector_store=ChromaDB(collection="neurosurfer")
)

text = reader.read("README.md")
chunks = chunker.chunk(text, file_path="README.md")
# or: ingestor.add_files(["README.md"]).build()
```

### Handling Errors

```python
txt = reader.read("possibly_corrupt.pdf")
if txt.startswith("Error reading"):
    # log and skip
    pass
```

---

## Extension Mapping Reference (`supported_types`)

Below is the canonical mapping initialized in `__init__` (extensions are lowercase). You can inspect it at runtime:

```python
reader = FileReader()
print(sorted(reader.supported_types.keys()))
```

**Registered as structured readers:** `.pdf`, `.html`, `.htm`, `.docx`, `.csv`, `.tsv`, `.xls`, `.xlsx`, `.xml`, `.yaml`, `.yml`, `.pptx`  
**Registered to read as plain‑text:** `.txt`, `.md`, `.rtf`, `.doc`, `.odt`, `.json`, `.ppt`, `.py`, `.ipynb`, `.java`, `.js`, `.ts`, `.jsx`, `.tsx`, `.cpp`, `.c`, `.h`, `.cs`, `.go`, `.rb`, `.rs`, `.php`, `.swift`, `.kt`, `.sh`, `.bat`, `.ps1`, `.scala`, `.lua`, `.r`, `.env`, `.ini`, `.toml`, `.cfg`, `.conf`, `.properties`, `.log`, `.tex`, `.srt`, `.vtt`  
**Anything else →** `_read_txt` (fallback)

---

## Production Notes & Best Practices

- **HTML**: `get_text()` strips tags; if you need DOM‑aware extraction (tables, links), post‑process the HTML separately and feed structured text to the Chunker.
- **PDF**: text extraction quality varies; consider adding an OCR fallback at a higher layer for scanned PDFs.
- **Excel/CSV**: very wide tables can produce long lines—rely on the Chunker’s char windows to split into manageable pieces.
- **YAML/XML**: readers return a serialized string; for semantic RAG over structured data, you may want to pre‑normalize into key/value lines.
- **Error strings**: treat them as loggable noise—skip during ingestion rather than embedding error messages.
- **Encoding**: if a file is known to be non‑UTF‑8, preconvert to UTF‑8 before invoking `read()` for best results.