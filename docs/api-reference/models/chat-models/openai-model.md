# OpenAIModel

**Module:** `neurosurfer.models.chat_models.openai`  
**Inherits:** [`BaseModel`](base-model.md)

## Overview

`OpenAIModel` connects Neurosurfer to any OpenAI-compatible chat endpoint. It ships with sensible defaults for OpenAI Cloud but also works with LM Studio, vLLM, Ollama OpenAI bridges, and other self-hosted gateways. Responses are normalised to OpenAI's `ChatCompletionResponse`/`ChatCompletionChunk` schema so agents can consume them transparently.

### Highlights

- Supports both non-streaming and streaming chat completions
- Accepts `stop_words` for client-side truncation without leaking the sequence
- Optional `strip_reasoning` helper to hide `<think>` or `<analysis>` blocks returned by newer models
- Attempts to load a tokenizer to estimate token usage when the server omits usage fields

## Constructor

### `OpenAIModel.__init__`

```python
from neurosurfer.config import config

OpenAIModel(
    model_name: str = "gpt-4o-mini",
    *,
    base_url: str | None = None,
    api_key: str | None = os.getenv("OPENAI_API_KEY"),
    timeout: float | None = 120.0,
    stop_words: list[str] | None = config.base_model.stop_words,
    strip_reasoning: bool = config.base_model.strip_reasoning,
    max_seq_length: int = config.base_model.max_seq_length,
    verbose: bool = config.base_model.verbose,
    logger: logging.Logger = logging.getLogger(),
    **kwargs,
)
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `model_name` | `str` | `"gpt-4o-mini"` | Target deployment on the OpenAI-compatible endpoint. |
| `base_url` | `str \| None` | `None` | Override API base URL (e.g. `http://localhost:8000/v1`). Leave `None` for OpenAI Cloud. |
| `api_key` | `str \| None` | `OPENAI_API_KEY` env var | Bearer token used by the OpenAI SDK. Can be omitted for proxies that ignore authentication. |
| `timeout` | `float \| None` | `120.0` | Maximum seconds to wait for a response. |
| `stop_words` | `list[str] \| None` | `config.base_model.stop_words` | Optional stop sequence list checked on the client side. |
| `strip_reasoning` | `bool` | `config.base_model.strip_reasoning` | If `True`, remove reasoning/thinking blocks before returning content. |
| `max_seq_length` | `int` | `config.base_model.max_seq_length` | Advertised context window, forwarded to `BaseModel`. |
| `verbose` | `bool` | `config.base_model.verbose` | Emit detailed logs during initialisation and errors. |
| `logger` | `logging.Logger` | `logging.getLogger()` | Logger instance reused across operations. |

The constructor immediately calls `init_model()` which instantiates the OpenAI client and tries to load a Hugging Face tokenizer matching `model_name` to estimate token usage when the server response omits it.

!!! tip
    `config` is imported from `neurosurfer.config`. Please see the [configuration](../../configuration.md#base-model-defaults) section for more details.
    
## Usage

### Non-streaming completion

```python
from neurosurfer.models.chat_models.openai import OpenAIModel

model = OpenAIModel(model_name="gpt-4o-mini")
response = model.ask("Explain retrieval-augmented generation in one paragraph.")

print(response.choices[0].message.content)
print(response.usage.total_tokens)
```

### Streaming completion

```python
stream = model.ask(
    "List three benefits of streaming responses.",
    system_prompt="Keep answers concise.",
    stream=True,
    temperature=0.4,
)

for chunk in stream:
    delta = chunk.choices[0].delta.content or ""
    print(delta, end="", flush=True)
```

### Working with stop words and reasoning blocks

```python
model.set_stop_words(["\nObservation:"])

# Hides <think>...</think> and similar reasoning tags
model.strip_reasoning = True

completion = model.ask("Return thought+answer with explicit markers.")
print(completion.choices[0].message.content)
```

Allowed passthrough parameters (forwarded directly to the OpenAI API) include `top_p`, `presence_penalty`, `frequency_penalty`, `logit_bias`, `user`, `n`, `response_format`, `seed`. Provide them as keyword arguments in the `ask` call.

## Error handling

API errors from `openai` (connection issues, rate limits, API status errors) are caught and returned as textual messages inside the response body so that your application can surface them without crashing. Enable `verbose=True` to emit detailed stack traces to logs.

## Environment variables

- `OPENAI_API_KEY` – default API key used when `api_key` is not supplied.
- Set `OPENAI_BASE_URL` alongside `base_url` in infrastructure tooling if you proxy requests.

Example `.env` snippet:

```bash
OPENAI_API_KEY=sk-your-key
OPENAI_MODEL_NAME=gpt-4o-mini
```

## Related APIs

- [`BaseModel`](base-model.md) – shared helpers and lifecycle hooks
- [`TransformersModel`](transformers-model.md) – local Hugging Face backend

*mkdocstrings output is temporarily disabled while import hooks are updated.*
