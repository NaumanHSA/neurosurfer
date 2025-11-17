# RAGAgent

**Module:** `neurosurfer.agents.rag`

## Overview

`RAGAgent` is a lightweight, modular retrieval core for Retrieval‑Augmented Generation (RAG). It is:

- **Vector‑store agnostic** — plugs into any store implementing the `BaseVectorDB` interface.  
- **Embedder agnostic** — works with any model implementing `BaseEmbedder`.  
- **LLM/tokenizer agnostic** — uses HuggingFace tokenizers when available, falls back to `tiktoken` if present, and otherwise applies a robust character‑based heuristic to keep prompts under the model’s context window.

It performs three primary steps:

1) **Retrieve**: embed the query and fetch the top‑K most similar chunks.  
2) **Build context**: convert retrieved docs into a joined context string (customizable).  
3) **Budget & trim**: fit the context into the LLM window while leaving room for generation; return a `RetrieveResult` with a safe `max_new_tokens` value.

Optionally, you can call `run(...)` to perform a **full generation** (retrieve → fill prompt → `llm.ask(...)`, streaming or not).

---

## Package layout

```
neurosurfer/agents/rag/
├─ __init__.py                 # exports RAGAgent, RAGAgentConfig, RetrieveResult
├─ config.py                   # dataclasses for config and results
├─ token_utils.py              # tokenizer/tokens fallback + trimming utilities
├─ context_builder.py          # format & join retrieved chunks
├─ picker.py                   # helper to pick top files by grouped hits
└─ agent.py                    # main agent implementation
```

---

## Constructor

```python
RAGAgent(
    llm: BaseChatModel,
    vectorstore: BaseVectorDB,
    embedder: BaseEmbedder,
    *,
    config: RAGAgentConfig | None = None,
    logger: logging.Logger | None = None,
    make_source: Callable[[Doc], str] | None = None,
)
```

| Parameter | Type | Description |
| --- | --- | --- |
| `llm` | [`BaseChatModel`](../models/chat-models/base-model.md) | Any supported chat model. Must expose `ask(...)` and (ideally) `max_seq_length`. A tokenizer is optional. |
| `vectorstore` | [`BaseVectorDB`](../vectorstores/base-vectordb.md) | Must expose `similarity_search(query_embedding, top_k, metadata_filter, similarity_threshold)`. |
| `embedder` | [`BaseEmbedder`](../models/embedders/base-embedder.md) | Must expose `embed(sequence[str], normalize_embeddings=bool) -> list[list[float]]`. |
| `config` | [`RAGAgentConfig`](#ragagentconfig) \| `None` | Retrieval, context‑formatting, and budgeting knobs (defaults used when `None`). |
| `logger` | `logging.Logger \| None` | Optional logger. |
| `make_source` | `Callable[[Doc], str] \| None` | Override how each doc’s “source” label is rendered in context (filename, URI, etc.). |

---

## `RAGAgentConfig`

```python
@dataclass
class RAGAgentConfig:
    # Retrieval
    top_k: int = 5
    similarity_threshold: float | None = None

    # Output budgeting
    fixed_max_new_tokens: int | None = None
    auto_output_ratio: float = 0.25
    min_output_tokens: int = 32
    safety_margin_tokens: int = 32

    # Context formatting
    include_metadata_in_context: bool = True
    context_separator: str = "\n\n---\n\n"
    context_item_header_fmt: str = "Source: {source}"
    normalize_embeddings: bool = True

    # Tokenizer fallbacks
    approx_chars_per_token: float = 4.0
```
### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `top_k` | `int` | Default number of chunks to fetch when `top_k` is not supplied to `retrieve(...)`. |
| `similarity_threshold` | `float \| None` | Global similarity floor for retrieval (override per call). |
| `fixed_max_new_tokens` | `int \| None` | Hard cap for generation. If `None`, the agent derives a cap dynamically after trimming. |
| `auto_output_ratio` | `float` | When no fixed cap is provided, initial portion of remaining window reserved for output before trimming (refined after trimming). |
| `min_output_tokens` | `int` | Guarantee at least this many tokens remain for generation. |
| `safety_margin_tokens` | `int` | Context margin to avoid overrunning model window due to tokenizer variance. |
| `include_metadata_in_context` | `bool` | When `True`, prefixes each chunk with `context_item_header_fmt.format(source=...)`. |
| `context_separator` | `str` | Separator between chunks in the final context string. |
| `context_item_header_fmt` | `str` | Format string for per‑chunk header line when metadata is included. |
| `normalize_embeddings` | `bool` | Passed to `embedder.embed(...)`; set `False` if your embedder already returns normalized vectors. |
| `approx_chars_per_token` | `float` | Heuristic used when the model has no tokenizer and `tiktoken` is unavailable. ~4 chars/token is a practical default. |

---

## Results object

### `RetrieveResult`

Returned by `retrieve(...)` and used by `run(...)` to fill prompts and set generation limits safely.

| Field | Type | Description |
| --- | --- | --- |
| `base_system_prompt` | `str` | System prompt provided to `retrieve(...)` |
| `base_user_prompt` | `str` | User prompt **template** (before inserting context). |
| `context` | `str` | **Trimmed** context actually used. |
| `max_new_tokens` | `int` | Recommended cap for generation after trimming. |
| `base_tokens` | `int` | Tokens for system + history + user (no context yet). |
| `context_tokens_used` | `int` | Tokens consumed by the trimmed context. |
| `token_budget` | `int` | Model’s total context window (`llm.max_seq_length` or default). |
| `generation_budget` | `int` | Remaining tokens for output. |
| `docs` | `list[Doc]` | Retrieved docs from the vector store. |
| `distances` | `list[float \| None]` | One per doc when the vector store returns distances. |
| `meta` | `dict` | Diagnostics (available_for_context, initial cap, margins, etc.). |

---

## Methods

### `retrieve(...) -> RetrieveResult`

```python
result = agent.retrieve(
    user_query="Explain vector databases in simple terms.",
    base_system_prompt="You are a helpful assistant.",
    base_user_prompt="Use the context to answer.\n\n{context}\n\nQuestion: {query}",
    chat_history=[{"role": "user", "content": "Hi!"}],
    top_k=8,
    metadata_filter={"collection": "docs"},
)
```

You can then use the result to build your own LLM call:

```python
filled = result.base_user_prompt.format(context=result.context, query="Explain vector databases in simple terms.")
response = llm.ask(
    system_prompt=result.base_system_prompt,
    user_prompt=filled,
    temperature=0.3,
    max_new_tokens=result.max_new_tokens,
)
```

### `run(...) -> Iterator[str] | str`

Runs **retrieve → fill → generate** for you. When `stream=True`, it yields generation chunks; otherwise it returns the full string.

```python
for token in agent.run(
    "List the main components of a RAG system.",
    base_system_prompt="You are concise.",
    base_user_prompt="Context:\n{context}\n\nQ: {query}\nA:",
    stream=True,
    temperature=0.2,
):
    print(token, end="")
```

**Arguments of note (subset):**

| Arg | Type | Description |
| --- | --- | --- |
| `stream` | `bool` | If `True`, yields streaming tokens from `llm.ask(..., stream=True)`. |
| `top_k` | `int \| None` | Overrides `config.top_k` for this call. |
| `metadata_filter` | `dict \| None` | Forwarded to the vector store. |
| `similarity_threshold` | `float \| None` | Per‑call similarity floor. |
| `temperature` | `float \| None` | Per‑call generation temperature (defaults to 0.3). |
| `max_new_tokens` | `int \| None` | Per‑call cap; if omitted, uses `RetrieveResult.max_new_tokens`. |
| `**llm_kwargs` | `Any` | Forwarded to `llm.ask(...)` (e.g., stop sequences). |

---

## Token handling & fallbacks

`TokenCounter` ensures prompts stay within the model window even without a tokenizer:

1. **HuggingFace path** — if `llm.tokenizer` exists, we use it (`apply_chat_template`, fast counting, and exact trimming).  
2. **tiktoken path** — if no tokenizer is available but `tiktoken` is installed, we use `cl100k_base` for robust counting & trimming.  
3. **Heuristic path** — otherwise, we estimate tokens via `len(text) / approx_chars_per_token` and binary‑search on character length to trim precisely enough.

This design guarantees safe budgeting across OpenAI‑style clients, HF models, and custom LLM wrappers.

---

## Context formatting

Context serialization is handled by `ContextBuilder`. By default it:

- Adds a header line like `Source: <label>` (when `include_metadata_in_context=True`), where `<label>` is produced by `make_source(doc)`.  
- Joins chunks with `context_separator` (default: `\n\n---\n\n`).

You can override `make_source` in the agent constructor or swap the builder if you need a different format (citations, bullet lists, etc.).

---

## File picking helper

When your index spans many files (e.g., a codebase), use the helper in `picker.py` to find the most promising files for deeper focus:

```python
from neurosurfer.agents.rag.picker import pick_files_by_grouped_chunk_hits

files = pick_files_by_grouped_chunk_hits(
    embedder=embedder,
    vector_db=vectorstore,
    section_query="vector similarity threshold",
    candidate_pool_size=200,
    n_files=5,
    file_key="filename",
)
```

This performs a wide similarity search, aggregates scores per file, and returns the top‑N paths.

---

## Best practices

- **Keep prompts short**: Your `base_user_prompt` is applied *before* context insertion. Excess preamble reduces space for retrieved evidence.  
- **Prefer normalized embeddings**: Set `normalize_embeddings=True` unless your embedder already normalizes vectors.  
- **Use `fixed_max_new_tokens` when needed**: For deterministic generations (e.g., latency budgeting), set an explicit cap.  
- **Log budgets**: `RetrieveResult.meta` and `generation_budget` make it easy to visualize headroom and trim behavior.  
- **Multi‑stage retrieval**: You can run `retrieve(...)` multiple times (e.g., coarse → re‑rank) and combine contexts before calling `run(...)` yourself.

---
