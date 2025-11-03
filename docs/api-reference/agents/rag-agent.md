# RAGRetrieverAgent

**Module:** `neurosurfer.agents.rag_retriever_agent`

## Overview

`RAGRetrieverAgent` is a composable retrieval core used by the rest of the framework. It does **not** call an LLM itself. Instead, it:

1. Embeds the query using the configured embedder
2. Pulls the most similar chunks from a vector store
3. Formats a context string (optionally including metadata headers)
4. Trims the context to fit within the model’s window
5. Computes the remaining token budget for generation

The result is a `RetrieveResult` dataclass that you can feed into any chat model (or hand over to another agent).

## Constructor

```python
RAGRetrieverAgent(
    llm: BaseModel,
    vectorstore: BaseVectorDB,
    embedder: BaseEmbedder,
    *,
    top_k: int = 5,
    similarity_threshold: float | None = None,
    logger: logging.Logger | None = None,
    fixed_max_new_tokens: int | None = None,
    auto_output_ratio: float = 0.25,
    min_output_tokens: int = 32,
    safety_margin_tokens: int = 32,
    include_metadata_in_context: bool = True,
    context_separator: str = "\n\n---\n\n",
    context_item_header_fmt: str = "Source: {source}",
    make_source: Callable[[Doc], str] | None = None,
    normalize_embeddings: bool = True,
)
```

| Parameter | Type | Description |
| --- | --- | --- |
| `llm` | [`BaseModel`](../models/chat-models/base-model.md) | Used only for tokenizer + context window information. |
| `vectorstore` | [`BaseVectorDB`](../vectorstores/base-vectordb.md) | Must expose `similarity_search(query_embedding, top_k, metadata_filter, similarity_threshold)`. |
| `embedder` | [`BaseEmbedder`](../models/embedders/base-embedder.md) | Must expose `embed(sequence[str], normalize_embeddings=bool) -> list[list[float]]`. |
| `top_k` | `int` | Default number of chunks to request when `top_k` is not supplied to `retrieve()`. |
| `similarity_threshold` | `float \| None` | Global similarity floor for retrieval (override per call). |
| `fixed_max_new_tokens` | `int \| None` | Optional hard cap for response generation. If `None`, the agent allocates a fraction of the window using `auto_output_ratio`. |
| `auto_output_ratio` | `float` | Portion of remaining tokens initially reserved for model output when no fixed cap is provided. |
| `min_output_tokens` | `int` | Guarantees at least this many tokens remain for generation. |
| `safety_margin_tokens` | `int` | Tokens left unused to avoid overshooting the model window. |
| `include_metadata_in_context` | `bool` | When `True`, prefixes each chunk with `context_item_header_fmt.format(source=...)`. |
| `context_separator` | `str` | Delimiter inserted between chunks. |
| `context_item_header_fmt` | `str` | Format string for chunk headers when metadata is included. |
| `make_source` | `Callable[[Doc], str] \| None` | Custom function to derive a human-readable source label. Defaults to filename/source/doc_id metadata. |
| `normalize_embeddings` | `bool` | Passed through to `embedder.embed(...)`. |

## Retrieval API

### `retrieve(...) -> RetrieveResult`

```python
result = agent.retrieve(
    user_query="Explain vector databases in simple terms.",
    base_system_prompt="You are a helpful assistant.",
    base_user_prompt="Answer the question using the context.\n\n{context}\n\nQuestion: {query}",
    chat_history=[{"role": "user", "content": "Hi!"}],
    top_k=8,
    metadata_filter={"collection": "docs"},
)
```

The returned `RetrieveResult` object contains everything you need to craft the final prompt:

| Field | Description |
| --- | --- |
| `base_system_prompt` | The system string you passed in (unchanged). |
| `base_user_prompt` | The user prompt template you provided (before injecting context). |
| `context` | Context string trimmed to fit within the model window. |
| `max_new_tokens` | Suggested `max_new_tokens` for the next generation call (after accounting for context length). |
| `base_tokens` | Token count for system + history + user prompt prior to context. |
| `context_tokens_used` | Tokens consumed by the trimmed context. |
| `token_budget` | Model’s total context window (`llm.max_seq_length`). |
| `generation_budget` | Tokens still available for generation after context insertion. |
| `docs` | List of [`Doc`](../vectorstores/base-vectordb.md#doc-dataclass) objects returned by the vector store. |
| `distances` | Optional similarity scores (one per doc) when provided by the vector store. |
| `meta` | Diagnostic data (available budget, initial guess, safety margin, etc.). |

Example usage to build the final model call:

```python
filled_user_prompt = result.base_user_prompt.format(
    context=result.context,
    query="Explain vector databases in simple terms.",
)

response = llm.ask(
    system_prompt=result.base_system_prompt,
    user_prompt=filled_user_prompt,
    temperature=0.3,
    max_new_tokens=result.max_new_tokens,
)
```

### `pick_files_by_grouped_chunk_hits(...) -> list[str]`

Utility to discover which files deserve focused attention (useful for large code bases). It performs a wide similarity search and returns the file keys with the highest concentration of hits.

```python
top_files = agent.pick_files_by_grouped_chunk_hits(
    section_query="vector similarity threshold",
    candidate_pool_size=200,
    n_files=5,
    file_key="filepath",
)
```

## Token budgeting helpers

`RAGRetrieverAgent` exposes a few static helpers that you can reuse in custom pipelines:

- `_calculate_max_new_tokens_prompts` — compute a safe `max_new_tokens` without any context.
- `_trim_context_by_token_limit` — replicate the trim-and-budget logic used internally.
- `_trim_text_to_tokens` — generic tokenizer-aware trimming routine.

Each accepts a `BaseModel` so the same heuristics work with Hugging Face, Unsloth, or OpenAI-compatible models.

## Implementation notes

- **Vector store contract**: the agent expects `similarity_search` to accept a raw embedding. Adapters are responsible for translating `metadata_filter` into the underlying database query.
- **Streaming compatibility**: the agent itself does not stream, but the recommended usage is to call `llm.ask(..., stream=True)` with the returned prompts and `max_new_tokens`.
- **Context formatting**: override `make_source` or subclass `_create_context` when you need custom formatting (citations, bullet lists, etc.).
- **Embedding normalisation**: set `normalize_embeddings=False` if your embedder already returns normalised vectors.
