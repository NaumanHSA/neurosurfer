# LlamaCppEmbedder

**Module:** `neurosurfer.models.embedders.llamacpp`  
**Inherits:** [`BaseEmbedder`](base-embedder.md)

## Overview

`LlamaCppEmbedder` produces embeddings using the `llama-cpp-python` bindings. It is ideal when you already host GGUF models for chat and want embeddings from the same runtime without depending on external APIs.

## Constructor

### `LlamaCppEmbedder.__init__`

```python
LlamaCppEmbedder(
    config: dict,
    *,
    logger: logging.Logger | None = None,
)
```

The `config` dictionary is passed directly to `llama_cpp.Llama`. Typical keys include:

```python
config = {
    "model_path": "/weights/nomic-embed-text-v1.5.Q4_K_M.gguf",
    "n_threads": 8,
    "n_gpu_layers": 0,
    "main_gpu": 0,
    "embedding": True,  # required
}
```

`embedding=True` must be set so llama.cpp exposes the embedding endpoint. All other llama.cpp parameters (batch size, rope scaling, etc.) are also supported.

## Embedding API

```python
from neurosurfer.models.embedders.llamacpp import LlamaCppEmbedder

embedder = LlamaCppEmbedder(config=config)

vector = embedder.embed("What is retrieval augmented generation?")
print(len(vector))

vectors = embedder.embed([
    "Neurosurfer ships a FastAPI server.",
    "Agents reuse OpenAI-compatible responses.",
])
print(len(vectors), len(vectors[0]))
```

### Signature

```python
embed(
    query: str | list[str],
    *,
    normalize_embeddings: bool = True,
) -> list[float] | list[list[float]]
```

- When `normalize_embeddings=True`, each vector is L2-normalised.
- Passing a list returns a list of vectors; no separate `embed_batch` method is needed.

## Tips

- GGUF embedding checkpoints such as `nomic-embed-text-v1.5` are readily available from Hugging Face (`TheBloke` maintains many quantised variants).
- Increase `n_threads` for CPU-bound workloads; when GPU acceleration is available, set `n_gpu_layers` to offload part of the encoder.
- The wrapper returns native Python lists. Convert to `numpy`/`torch` as needed: `np.array(embedder.embed(texts))`.

## Related embedders

- [`SentenceTransformerEmbedder`](sentence-transformer.md) – higher-quality, transformer-based embeddings.

*mkdocstrings output is temporarily disabled while import hooks are updated.*
