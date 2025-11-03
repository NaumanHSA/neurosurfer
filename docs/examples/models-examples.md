# Models

This page shows how to use Neurosurfer's **chat models** (OpenAI-compatible and local Transformers) and **embedders** with a small, practical API. All chat models share the same `BaseModel.ask(...)` interface and produce **OpenAI‑compatible** Pydantic responses, so you can swap backends without changing your app code.

---

## 🔎 At a glance

```python
from neurosurfer.models.chat_models.openai_model import OpenAIModel

model = OpenAIModel(model_name="gpt-4o-mini")  # or any OpenAI-compatible name
res = model.ask("Say hi in one sentence.", temperature=0.2)
print(res.choices[0].message.content)
```

---

## 💬 Chat models

Neurosurfer unifies different backends behind a single abstract class:

- **`BaseModel`** — defines the interface (non-stream + stream, stop/interrupt, stop‑words, thinking tag suppression)
- **`OpenAIModel`** — uses OpenAI/compatible servers (OpenAI Cloud, LM Studio, vLLM, Ollama, etc.)
- **`TransformersModel`** — runs local Hugging Face models with `transformers`, optional 4‑bit loading
- **`UnslothModel`** — runs local and Hugging Face models with `unsloth`
- **`LlamaCppModel`** — runs local and Hugging Face GGUF models with `llamacpp`

### Return types (OpenAI-style)

- **Non-streaming:** `ChatCompletionResponse` → `response.choices[0].message.content`
- **Streaming:** generator of `ChatCompletionChunk` → `chunk.choices[0].delta.content`

---

### Non‑streaming completion (OpenAI‑compatible)

```python
from neurosurfer.models.chat_models.openai_model import OpenAIModel

# Works with: OpenAI Cloud, LM Studio, vLLM OpenAI server, Ollama (OpenAI compat)
model = OpenAIModel(
    model_name="gpt-4o-mini",
    # base_url=None uses OpenAI cloud; set base_url for self-hosted OpenAI-compatible servers
    # base_url="http://localhost:8000/v1",  # vLLM example
    # api_key="sk-...",                      # or "ollama"/"lm-studio" for local compat servers
)

resp = model.ask(
    user_prompt="Explain RAG in 2 lines.",
    temperature=0.3,
    max_new_tokens=256,
)
print(resp.choices[0].message.content)
```

**Tip:** You can keep a single code path while switching providers just by changing `base_url` and `api_key`.

---

### Streaming completion (token‑by‑token)

```python
from neurosurfer.models.chat_models.openai_model import OpenAIModel

model = OpenAIModel(model_name="gpt-4o-mini")

for chunk in model.ask("Stream me a haiku about GPUs.", stream=True):
    delta = chunk.choices[0].delta.content or ""
    print(delta, end="", flush=True)
print()  # newline
```

---

### System prompt + chat history

```python
from neurosurfer.models.chat_models.openai_model import OpenAIModel

model = OpenAIModel(model_name="gpt-4o-mini")

history = [
    {"role": "user", "content": "Who won the 2018 FIFA World Cup?"},
    {"role": "assistant", "content": "France."},
]

resp = model.ask(
    user_prompt="And who was the top scorer? Reply in 1 short sentence.",
    system_prompt="You are a concise sports assistant.",
    chat_history=history,
    temperature=0.2,
)
print(resp.choices[0].message.content)
```

---

### Stop words & reasoning / thinking suppression

```python
from neurosurfer.models.chat_models.openai_model import OpenAIModel
from neurosurfer.config import config

# Example: cut generation as soon as model emits any of these substrings
model = OpenAIModel(
    model_name="gpt-4o-mini",
    stop_words=["\n\n", "<END>", "###"],
    # Strip/suppress internal chain-of-thought style tags if present
    # (e.g., <think>...</think>, <analysis>...</analysis>, etc.)
    # To suppress: set enable_thinking=False via BaseModel init (configurable)
)

text = "Give me a step-by-step plan to brew coffee at home."
resp = model.ask(text, temperature=0.4, max_new_tokens=300)
print(resp.choices[0].message.content)
```

> Behind the scenes, `BaseModel` scans streamed text for the first matching stop token and truncates output. When thinking is disabled, recognized tags like `<think>...</think>` are removed from the stream.

---


### Local Transformers (CPU/GPU), optional 4‑bit

```python
from neurosurfer.models.chat_models.transformers_model import TransformersModel

# Example model: a small instruct model from HF (pick something that fits your GPU/CPU)
lm = TransformersModel(
    model_name="microsoft/Phi-3-mini-4k-instruct",   # change as needed
    max_seq_length=4096,
    load_in_4bit=True,                               # try 4-bit to save VRAM if supported
    verbose=True,
)

# Non‑streaming
out = lm.ask("Summarize attention mechanisms in one paragraph.", temperature=0.3)
print(out.choices[0].message.content)

# Streaming
for ch in lm.ask("Stream 2 facts about Transformers (the model).", stream=True):
    print(ch.choices[0].delta.content or "", end="")
print()
```

> The `TransformersModel` picks `cuda` if available, otherwise CPU, and selects a sensible dtype (`bfloat16` on GPU, `float32` on CPU).

---

## 🔢 Embedders (for RAG & similarity)

Neurosurfer also standardizes embedding models with a simple `BaseEmbedder` → `embed(...)` API.

### SentenceTransformer (HF) quick start

```python
from neurosurfer.models.embedders.sentence_transformer import SentenceTransformerEmbedder

emb = SentenceTransformerEmbedder("intfloat/e5-small-v2")  # pick a model that fits your device

# Single text → one vector
v = emb.embed("vector search is fun")
print(len(v))  # e.g., 384 or 768 depending on the model

# Batch texts → list of vectors
B = emb.embed(["first text", "second text"])
print(len(B), len(B[0]))
```

> Tip: pass `quantized=True` (default) to attempt 8‑bit loading where supported, or `quantized=False` for full precision.

---

## 📌 Patterns & tips

- **Swap backends, keep code:** both `OpenAIModel` and `TransformersModel` return the same Pydantic types.
- **Streaming UI:** iterate chunks and read `chunk.choices[0].delta.content`.
- **Safety & control:** use `stop_words`, `enable_thinking=False`, and the stop signal to keep outputs tidy.
- **Latency/Cost:** set smaller `max_new_tokens` and lower `temperature` for fast, deterministic flows.
- **Reusability:** create one model instance and share it across requests where possible.

---
