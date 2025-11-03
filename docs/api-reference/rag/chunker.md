# Chunker

> Module: `neurosurfer.rag.chunker`

A production‑grade, extensible **document chunker** purpose‑built for RAG (Retrieval‑Augmented Generation) systems. It supports **AST‑aware** Python chunking, structure‑aware JS/TS chunking, header‑aware Markdown chunking, JSON object/array chunking, and robust fallbacks (line‑based for code‑like, char‑based for prose). It also includes **custom handler registration**, a **router** for dynamic strategy selection, **comment‑aware filtering**, **prompt‑block stripping**, and **safety caps** to prevent pathological outputs from custom logic.

---

## Key Capabilities

- **Strategy registry by file extension** (built‑in + custom)  
  - Built‑ins: `.py`, `.js`, `.ts`, `.tsx`, `.jsx`, `.json`, `.md`, `.txt`
- **AST‑aware Python chunking**: groups imports / defs / classes (with decorators) and then windows large blocks
- **JS/TS structure‑aware chunking**: coarse segmentation by `function`/`class` followed by cleanup
- **Markdown/README header‑aware chunking**: windows by headings/max lines
- **JSON chunking**: per top‑level object/array element with size caps; falls back to char windows if needed
- **Fallback heuristics**: `"auto"` detection prefers line‑based windows for code‑like text, char‑based for prose
- **Comment‑aware filtering**: collapses long comment blocks, skips fully‑commented chunks
- **Prompt‑block filtering** (triple‑quoted strings that look like prompts), reducing retrieval poisoning
- **Custom handler system**: register callable handlers by **name** and bind them to extensions or a **router**
- **Safety limits**: cap chunk counts and total output characters from custom handlers
- **Blacklist skip**: ignore common binary/config/infra paths early
- **Optional logging hook**

---

## Configuration (`ChunkerConfig`)

`ChunkerConfig` centralizes chunk sizes, overlaps, fallbacks, and safety limits.

| Name | Type | Default | Description |
|---|---|---:|---|
| `fallback_chunk_size` | `int` | `25` | **Line-based** window size (lines) for generic code splitting and Python/JS sub‑chunking. |
| `overlap_lines` | `int` | `3` | Line overlap between consecutive windows (context retention). |
| `max_chunk_lines` | `int` | `1000` | Hard cap on lines per chunk to avoid giant blocks. |
| `comment_block_threshold` | `int` | `4` | Consecutive comment‑only lines at/above this threshold form a “comment block” that can be dropped. |
| `char_chunk_size` | `int` | `1000` | **Char-based** window size (chars) for prose/unknown text. |
| `char_overlap` | `int` | `150` | Char overlap between consecutive char windows. |
| `readme_max_lines` | `int` | `30` | Max lines per Markdown/README chunk. |
| `json_chunk_size` | `int` | `1000` | Max chars per JSON sub‑chunk when pretty‑printing. |
| `fallback_mode` | `str` | `"char"` | What to do when no strategy: `"char"`, `"line"`, or `"auto"` (code‑like → line; else char). |
| `max_returned_chunks` | `int` | `500` | Hard limit on number of chunks returned by a custom handler (post‑sanitize). |
| `max_total_output_chars` | `int` | `1_000_000` | Hard limit on total characters returned by a custom handler (post‑sanitize). |
| `min_chunk_non_ws_chars` | `int` | `1` | Drop chunks that have fewer than this many **non‑whitespace** characters. |

**Example**

```python
from neurosurfer.rag.chunker import Chunker, ChunkerConfig

config = ChunkerConfig(
    fallback_chunk_size=30,
    overlap_lines=5,
    char_chunk_size=1000,
    comment_block_threshold=4,
    fallback_mode="auto"
)
chunker = Chunker(config)
```

---

## Blacklist / Skip Rules

`Chunker` avoids ingesting common non‑text or infra files via `_should_skip_file(file_path)` which tests against these regular‑expression patterns (examples):  

```
*.lock, .env*, .git*, node_modules, __pycache__, .DS_Store, Thumbs.db,
*.png, *.jpg, *.svg, *.ico, *.zip, *.tar.gz, *.mp4, *.mp3,
/dist/, /build/, .idea, .vscode, .eslintrc, .prettierrc, .editorconfig,
.gitignore, /LICENSE, /CODEOWNERS, /CONTRIBUTING.md, /CHANGELOG.md
```

> If a path matches any blacklist regex, `chunk(...)` returns `[]` immediately.

---

## Public API

### `class Chunker(config: ChunkerConfig = ChunkerConfig())`

#### Built‑in strategy registration

- `.register(exts: List[str], fn: StrategyFn) -> None`  
  Map file extensions to a **strategy function**: `fn(text: str, file_path: Optional[str]) -> List[str]`  
  Built‑ins are pre‑registered for:  
  - `.py` → `_chunk_python` (AST‑aware + line windows)  
  - `.js`, `.ts`, `.tsx`, `.jsx` → `_chunk_javascript`  
  - `.json` → `_chunk_json`  
  - `.md`, `.txt` → `_chunk_readme`

#### Custom handler registry

- `.register_custom(name: str, handler: CustomChunkHandler) -> None`  
  Registers a named handler. The `CustomChunkHandler` signature is:  
  `handler(text: str, *, file_path: Optional[str] = None, config: Optional[ChunkerConfig] = None) -> List[str]`

- `.unregister_custom(name: str) -> None`  
  Removes the handler and any extension mappings pointing to it.

- `.list_custom_handlers() -> List[str]`  
  Names of all registered custom handlers.

- `.use_custom_for_ext(exts: List[str], handler_name: str) -> None`  
  Route specific extensions to a named custom handler.

- `.clear_custom_for_ext(exts: List[str]) -> None`  
  Remove previous extension mappings.

- `.set_router(router: Optional[Callable[[Optional[str], str], Optional[str]]]) -> None`  
  Install a **router** invoked as `router(file_path, text) -> handler_name | None`. If it returns the name of a registered handler, that handler is used.

- `.list_ext_mappings() -> List[Tuple[str, str]]`  
  Returns current extension → custom handler mappings.

#### Logging (optional)

- `.set_logger(logger_fn: Callable[[str], None]) -> None`  
  Attach a logger callback. Internal helpers route info/warn/error strings to this function.

#### Main entry point

- `.chunk(text: str, *, source_id: str | None = None, file_path: str | None = None, k: int = 40, custom: str | CustomChunkHandler | None = None) -> List[str]`

**Priority order** used by `chunk(...)`:

1. **Explicit** `custom` handler (string name or callable)  
2. **Router** result (registered handler name)  
3. **Extension** mapping (registered handler name)  
4. **Built‑in** strategy for the extension  
5. **Heuristic fallback**: line windows if `_looks_like_code(text)` else char windows

Additional rules:

- If `file_path` matches the **blacklist**, returns `[]`.
- If `text` word count `<= k`, returns the **whole** text as a **single** chunk (if it meets `min_chunk_non_ws_chars`).

---

## Strategy Details

### Python (`_chunk_python`)
- Strips **prompt‑like triple‑quoted blocks** first (`_filter_prompt_like_blocks`).
- Parses AST; for each top‑level node (imports, defs/classes incl. decorators), collects its line range.
- Sub‑chunks large blocks by **line windows** with cleanup: `_split_into_chunks()` → `clean_chunk_lines()`
- Skips fully‑commented windows.
- Appends any **remaining lines** (non‑AST or trailing comments) via the same windowing logic.
- Fallback to `_line_windows` on parse failure.

### JavaScript/TypeScript (`_chunk_javascript`)
- Coarse split by `function ... {` or `class ... {` occurrences.
- Cleans each segment via `_clean_lines` and skips fully‑commented blocks.
- Falls back to `_line_windows` if no structural matches found.

### JSON (`_chunk_json`)
- Attempts `json.loads(text)`:
  - **dict**: chunk each `{"key": value}` pretty‑printed, truncating by `json_chunk_size`.
  - **list**: chunk each element pretty‑printed, capped by `json_chunk_size`.
- On parse error, falls back to `_char_windows(text, json_chunk_size, json_chunk_size * 0.2)`.

### Markdown/README (`_chunk_readme`)
- Windows by headings and `readme_max_lines`, ensuring manageable slices for embeddings.

### Fallbacks

- `_line_windows(text, window=fallback_chunk_size)`  
  Removes empty lines, applies line overlap (`overlap_lines`), cleans long comment blocks, skips fully‑commented windows.

- `_char_windows(text, size=char_chunk_size, overlap=char_overlap)`  
  Sliding char windows with overlap; empties removed inside each chunk.

### Heuristics & Filters

- `_looks_like_code(text)` – detects code‑likeness with braces/semicolons/keywords/indent patterns & short lines ratio.
- `_is_comment_line(line)` and `_is_fully_commented(lines)` – used to skip comment‑only chunks.
- `_clean_lines(chunk_lines, comment_block_threshold)` – collapses long comment blocks.
- `_is_prompt_like(text)` + `_filter_prompt_like_blocks(code)` – removes triple‑quoted strings that look like LLM prompts.

### Safety & Sanitization (custom handlers)

- `_sanitize_chunks(chunks)` enforces:  
  - `max_returned_chunks` and `max_total_output_chars`  
  - drop entries shorter than `min_chunk_non_ws_chars`  
  - trim trailing newlines

---

## Usage Examples

### Baseline usage

```python
from pathlib import Path
from neurosurfer.rag.chunker import Chunker, ChunkerConfig

chunker = Chunker(ChunkerConfig(fallback_mode="auto"))

py_text = Path("app/main.py").read_text()
py_chunks = chunker.chunk(py_text, file_path="app/main.py")

md_text = Path("README.md").read_text()
md_chunks = chunker.chunk(md_text, file_path="README.md")

raw = "A lot of prose..."
txt_chunks = chunker.chunk(raw, file_path="notes.unknown")  # auto → char windows
```

### Register a **custom** handler (by name)

```python
from typing import List, Optional
from neurosurfer.rag.chunker import Chunker, ChunkerConfig

def my_handler(text: str, *, file_path: Optional[str] = None, config: Optional[ChunkerConfig] = None) -> List[str]:
    # Example: split on double newlines and clamp ~1200 chars
    segs = [s.strip() for s in text.split("\n\n") if s.strip()]
    out, buf = [], ""
    for s in segs:
        if len(buf) + len(s) + 2 <= 1200:
            buf = f"{buf}\n\n{s}".strip()
        else:
            if buf: out.append(buf)
            buf = s
    if buf: out.append(buf)
    return out

chunker = Chunker()
chunker.register_custom("my_handler", my_handler)
chunker.use_custom_for_ext([".rst", ".tex"], "my_handler")

rst_text = open("paper.rst").read()
chunks = chunker.chunk(rst_text, file_path="paper.rst")
```

### Use an **ad‑hoc callable** for a single call

```python
def once(text: str, *, file_path=None, config=None):
    return [p for p in text.split("\n\n") if p.strip()]

chunks = chunker.chunk(big_text, custom=once)
```

### Install a **router** to decide dynamically

```python
def router(file_path, text):
    if file_path and file_path.endswith(".proto"):
        return "proto_chunks"   # must be registered via register_custom
    if "BEGIN:VCARD" in text:
        return "vcard_chunks"
    return None

chunker.set_router(router)
```

### Attach a logger

```python
chunker.set_logger(lambda msg: print(f"[chunker] {msg}"))
```

---

## Production Notes & Best Practices

- Prefer `fallback_mode="auto"` in mixed repos; it chooses line windows for code‑like content.
- Keep `overlap_lines` low (2–5) to reduce duplication in code while preserving local context.
- For long prose, start with `char_chunk_size ~ 800–1200` and `char_overlap ~ 100–200`.
- Treat JSON as **structured**: chunking at top‑level keys/elements often boosts retrieval precision.
- When adding custom handlers, rely on `_sanitize_chunks` (already applied) and consider your own local guards.
- The blacklist is conservative; extend/override upstream if needed before crawling a repo.
- Chunk **after** text normalization (decode, HTML → text, etc.) to keep windowing deterministic.