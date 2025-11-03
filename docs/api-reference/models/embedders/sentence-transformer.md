# SentenceTransformerEmbedder

**Module:** `neurosurfer.models.embedders.sentence_transformer`  
**Inherits:** [`BaseEmbedder`](base-embedder.md)

## Overview

Wraps the `sentence-transformers` library to generate dense vectors for retrieval, clustering, or semantic search. The embedder optionally loads models with 8-bit quantisation using `BitsAndBytesConfig` for reduced memory consumption.

## Constructor

### `SentenceTransformerEmbedder.__init__`

```python
SentenceTransformerEmbedder(
    model_name: str,
    *,
    logger: logging.Logger | None = None,
    quantized: bool = True,
)
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `model_name` | `str` | – | Hugging Face id or local path recognised by `sentence-transformers`. |
| `logger` | `logging.Logger \| None` | `None` | Custom logger; falls back to a module-level logger. |
| `quantized` | `bool` | `True` | When `True`, wraps the Transformer with `BitsAndBytesConfig(load_in_8bit=True)` to cut memory usage. |

When `quantized=True`, the embedder builds a custom pipeline consisting of a quantised transformer block and a pooling layer. Otherwise it loads the plain `SentenceTransformer`.

## Embedding text

```python
from neurosurfer.models.embedders.sentence_transformer import SentenceTransformerEmbedder

embedder = SentenceTransformerEmbedder(model_name="intfloat/e5-large-v2")

# Single text → list[float]
query_vector = embedder.embed("What is retrieval augmented generation?")

# Multiple texts → list[list[float]]
doc_vectors = embedder.embed([
    "Neurosurfer ships a FastAPI server.",
    "Agents rely on OpenAI-style responses.",
])
```

### Method signature

```python
embed(
    query: str | list[str],
    *,
    convert_to_tensor: bool = False,
    normalize_embeddings: bool = True,
) -> list[float] | list[list[float]]
```

- Set `convert_to_tensor=True` to receive `torch.Tensor` objects (useful when chaining with PyTorch operations).
- Disable `normalize_embeddings` if you need raw vectors instead of unit-normalised outputs. By default it normalises to make cosine similarity equivalent to dot product.

## Tips

- Quantised models run efficiently on commodity GPUs; disable quantisation for CPU-only environments or when you require the original precision.
- Sentence-transformer checkpoints sometimes require authentication (e.g. private models). Run `huggingface-cli login` if necessary.
- Combine with `neurosurfer.vectorstores.ChromaVectorStore` or other vector stores through the shared `BaseEmbedder` API.

## Related embedders

- [`LlamaCppEmbedder`](llamacpp-embedder.md) – build embeddings with llama.cpp-based GGUF models.

*mkdocstrings output is temporarily disabled while import hooks are updated.*
