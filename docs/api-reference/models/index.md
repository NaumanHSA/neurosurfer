# Models API

Neurosurfer bundles two model families:

- **Chat models** (`neurosurfer.models.chat_models`) – large language models used by agents and the FastAPI server.
- **Embedders** (`neurosurfer.models.embedders`) – vector generators for RAG pipelines and semantic search.

Both families share consistent Pydantic responses and configuration patterns so you can swap providers without touching business logic.

## Navigation

<div class="grid cards" markdown>

-   :material-chat:{ .lg .middle } **Chat Models**

    ---

    Large language models with OpenAI-compatible responses.

    [:octicons-arrow-right-24: Explore chat models](chat-models/index.md)

-   :material-vector-triangle:{ .lg .middle } **Embedders**

    ---

    Sentence encoders for semantic search and retrieval.

    [:octicons-arrow-right-24: Explore embedders](embedders/index.md)

</div>

## Quick comparison

### Chat models

| Model | Backend | Best for |
| --- | --- | --- |
| `OpenAIModel` | Hosted API | OpenAI Cloud, LM Studio, or any OpenAI-compatible gateway |
| `TransformersModel` | Local PyTorch | GPU/CPU inference of Hugging Face checkpoints |
| `UnslothModel` | Unsloth CUDA | Fast inference for LoRA/QLoRA finetunes |
| `LlamaCppModel` | llama.cpp | CPU-first deployments and GGUF quantised weights |
| `VLLMModel` | HTTP client | Remote vLLM clusters with OpenAI-compatible APIs |

### Embedders

| Embedder | Backend | Best for |
| --- | --- | --- |
| `SentenceTransformerEmbedder` | sentence-transformers | High-quality open-source embeddings |
| `LlamaCppEmbedder` | llama.cpp | GGUF embedding models without GPU dependencies |

## Getting started

### Chat model example

```python
from neurosurfer.models.chat_models.openai import OpenAIModel

model = OpenAIModel(model_name="gpt-4o-mini")
reply = model.ask("Summarise the project in two sentences.")

print(reply.choices[0].message.content)
```

### Embedder example

```python
from neurosurfer.models.embedders.sentence_transformer import SentenceTransformerEmbedder

embedder = SentenceTransformerEmbedder(model_name="intfloat/e5-large-v2")
vector = embedder.embed("Explain retrieval augmented generation.")

print(len(vector))  # embedding dimensionality
```

## See also

- [Agents](../agents/index.md)
- [RAG system](../rag/index.md)
