# BaseModel

**Module:** `neurosurfer.models.chat_models.base`  
**Type:** Abstract base class

## Overview

`BaseModel` is the foundation for every chat model shipped with Neurosurfer. It standardises request handling, streaming, token accounting, and OpenAI-compatible responses so that downstream agents can work with any backend without branching.

### Core capabilities

- Creates OpenAI-style `ChatCompletionResponse` / `ChatCompletionChunk` objects via `ask`
- Handles streaming vs. non-streaming in a single entry point
- Tracks prompt/completion token usage (when provided or estimated)
- Provides helper methods that subclasses can reuse to build deltas, stop chunks, and final responses
- Exposes `stop_generation`, `reset_stop_signal`, and `set_stop_words` hooks that concrete implementations respond to

## Constructor

### `BaseModel.__init__`

```python
from neurosurfer.config import config

BaseModel(
    *,
    max_seq_length: int = config.base_model.max_seq_length,
    verbose: bool = config.base_model.verbose,
    logger: logging.Logger = logging.getLogger(),
    **kwargs,
)
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `max_seq_length` | `int` | `config.base_model.max_seq_length` | Maximum context window supported by the backend. |
| `verbose` | `bool` | `config.base_model.verbose` | When `True`, subclasses should emit additional debug logs. |
| `logger` | `logging.Logger` | `logging.getLogger()` | Logger instance reused across helper methods. |
| `**kwargs` | any | – | Forwarded to subclass initialisation; stored on subclasses as needed. |

Subclasses must call `super().__init__` so shared state (`model_name`, `lock`, `max_seq_length`, etc.) is initialised before loading a backend in `init_model()`.

!!! tip
    `config` is imported from `neurosurfer.config`. Please see the [configuration](../../configuration.md#base-model-defaults) section for more details.

## Public interface

### `ask`

```python
ask(
    user_prompt: str,
    *,
    system_prompt: str = config.base_model.system_prompt,
    chat_history: list[dict] = [],
    temperature: float = config.base_model.temperature,
    max_new_tokens: int = config.base_model.max_new_tokens,
    stream: bool = False,
    **kwargs,
) -> ChatCompletionResponse | Generator[ChatCompletionChunk, None, None]
```

Routes the request to `_call` (non-streaming) or `_stream` (streaming) and returns OpenAI-compatible Pydantic models. All subclasses must implement `_call` and `_stream`.

Usage:

```python
response = model.ask("Explain RAG in one sentence")
print(response.choices[0].message.content)

for chunk in model.ask("Summarise the docs", stream=True):
    delta = chunk.choices[0].delta.content or ""
    print(delta, end="")
```

### `stop_generation`

Sets an implementation-defined flag instructing the current generation to stop at the next opportunity. Concrete models should respect this flag inside their generation loops (see the built-in integrations for patterns).

### `reset_stop_signal`

Clears any stop signal before a new request runs. Typically called internally by `ask`.

### `set_stop_words`

Stores a stop word list which helpers such as `_transformers_consume_stream` use to truncate responses without emitting the terminator tokens.

## Hooks for subclass authors

Concrete implementations **must** provide the following methods:

- `init_model(self) -> None`: load the underlying model/client/tokenizer.
- `_call(...) -> ChatCompletionResponse`: run a blocking generation and return the final response.
- `_stream(...) -> Generator[ChatCompletionChunk, None, None]`: yield partial chunks followed by a terminal stop chunk.
- `stop_generation(self) -> None`: honour stop requests from callers.

Optional helpers supplied by `BaseModel`:

- `_delta_chunk(call_id, model, content)` – build incremental streaming chunks.
- `_stop_chunk(call_id, model, finish_reason)` – emit the terminating streaming chunk.
- `_final_nonstream_response(call_id, model, content, prompt_tokens, completion_tokens)` – return a complete `ChatCompletionResponse`.
- `_transformers_consume_stream(streamer)` – consume tokens from a `TextIteratorStreamer`, applying stop words and `<think>` filtering.

## Creating a custom model

```python
from neurosurfer.models.chat_models.base import BaseModel
from neurosurfer.server.schemas import ChatCompletionResponse

class CustomModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.init_model()

    def init_model(self):
        self.model = load_backend_somehow()
        self.tokenizer = load_tokenizer()
        self.model_name = "my-custom-model"

    def _call(self, **kwargs) -> ChatCompletionResponse:
        # Run the backend, produce text, then build the response
        content = self._run_backend(**kwargs)
        return self._final_nonstream_response(
            call_id=self.call_id,
            model=self.model_name,
            content=content,
            prompt_tokens=0,
            completion_tokens=0,
        )

    def _stream(self, **kwargs):
        for token in self._backend_stream(**kwargs):
            yield self._delta_chunk(self.call_id, self.model_name, token)
        yield self._stop_chunk(self.call_id, self.model_name, finish_reason="stop")

    def stop_generation(self):
        self.backend.cancel_current_request()
```

## See also

- [OpenAIModel](openai-model.md)
- [TransformersModel](transformers-model.md)
- [Chat models index](index.md)

*mkdocstrings output is temporarily disabled while import hooks are updated.*
