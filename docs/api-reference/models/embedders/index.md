# Embedders API

Embedding backends turn text into dense vectors used by retrieval, clustering, and semantic search components.

## Available embedders

| Embedder | Backend | Ideal for |
| --- | --- | --- |
| [`SentenceTransformerEmbedder`](sentence-transformer.md) | `sentence-transformers` | High-quality multilingual embeddings with optional 8-bit quantisation |
| [`LlamaCppEmbedder`](llamacpp-embedder.md) | `llama-cpp-python` | Deployments that already rely on GGUF models and need fully offline embeddings |

## Quick start

### SentenceTransformer

```python
from neurosurfer.models.embedders.sentence_transformer import SentenceTransformerEmbedder

embedder = SentenceTransformerEmbedder(model_name="intfloat/e5-large-v2")

query_vector = embedder.embed("What is retrieval augmented generation?")
doc_vectors = embedder.embed([
    "Neurosurfer exposes an OpenAI-compatible API.",
    "Agents can run on top of multiple model backends.",
])
```

### LlamaCpp

```python
from neurosurfer.models.embedders.llamacpp import LlamaCppEmbedder

config = {
    "model_path": "/weights/nomic-embed-text-v1.5.Q4_K_M.gguf",
    "n_threads": 8,
    "embedding": True,
}

embedder = LlamaCppEmbedder(config=config)
vector = embedder.embed("Offline embeddings with llama.cpp are convenient.")
```

## Choosing an embedder

- Pick **SentenceTransformerEmbedder** when you prioritise accuracy and have access to the Hugging Face ecosystem.
- Pick **LlamaCppEmbedder** when you need an entirely local stack and already distribute GGUF models alongside chat backends.

## See also

- [BaseEmbedder](base-embedder.md)
- [RAG system](../../rag/index.md)
- [Vector stores](../../vectorstores/index.md)
- [Chat models](../chat-models/index.md)
