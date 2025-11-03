# Chat Models API

Large language models shipped with Neurosurfer share a unified interface through [`BaseModel`](base-model.md). Choose the backend that matches your deployment constraints—cloud APIs, local GPUs, llama.cpp, or remote vLLM servers.

## Supported backends

| Model | Runtime | Deploy on | Ideal for |
| --- | --- | --- | --- |
| [`OpenAIModel`](openai-model.md) | Hosted API | OpenAI Cloud, LM Studio, Ollama, vLLM | Production-ready quality with minimal setup |
| [`TransformersModel`](transformers-model.md) | PyTorch | Local GPU/CPU | Running Hugging Face checkpoints directly |
| [`UnslothModel`](unsloth-model.md) | CUDA (Unsloth) | Local GPU | Fast inference for LoRA/QLoRA finetunes |
| [`LlamaCppModel`](llamacpp-model.md) | llama.cpp | CPU / lightweight GPU | GGUF quantised models with tiny footprint |

## Quick start examples

### 1. Hosted API (OpenAI-compatible)

```python
from neurosurfer.models.chat_models.openai import OpenAIModel

model = OpenAIModel(model_name="gpt-4o-mini")
completion = model.ask("Explain retrieval augmented generation in one sentence.")

print(completion.choices[0].message.content)
```

### 2. Hugging Face checkpoint (Transformers)

```python
from neurosurfer.models.chat_models.transformers import TransformersModel

model = TransformersModel(
    model_name="/weights/Qwen3-4B-unsloth-bnb-4bit",
    load_in_4bit=False,  # already quantised
)

response = model.ask("List three benefits of local inference.")
print(response.choices[0].message.content)
```

### 3. llama.cpp GGUF

```python
from neurosurfer.models.chat_models.llamacpp import LlamaCppModel

model = LlamaCppModel(
    model_path="/weights/llama-2-7b-chat.Q4_K_M.gguf",
    n_ctx=4096,
    n_threads=8,
)

answer = model.ask("What are GGUF files?")
print(answer.choices[0].message.content)
```

### 4. Unsloth accelerated inference

```python
from neurosurfer.models.chat_models.unsloth import UnslothModel

model = UnslothModel(
    model_name="/weights/Qwen3-4B-unsloth-bnb-4bit",
    enable_thinking=False,
)

reply = model.ask("Describe Unsloth in two bullet points.")
print(reply.choices[0].message.content)
```

## Choosing a backend

- Use **OpenAIModel** when you need the highest quality models or want to point at any OpenAI-compatible API.
- Use **TransformersModel** when you manage your own Hugging Face checkpoints and have GPU resources.
- Use **UnslothModel** for LoRA/QLoRA finetunes optimised with Unsloth’s runtime.
- Use **LlamaCppModel** on CPU-first deployments or when you rely on GGUF quantised weights.

## See also

- [Embedders](../embedders/index.md)
- [Agents](../../agents/index.md)
