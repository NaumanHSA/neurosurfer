# TransformersModel

**Module:** `neurosurfer.models.chat_models.transformers`  
**Inherits:** [`BaseModel`](base-model.md)

## Overview

`TransformersModel` wraps any Hugging Face causal language model for local inference. It handles device placement, optional 4-bit quantisation via `bitsandbytes`, stop word filtering, and streaming support through `TextIteratorStreamer`.

### Highlights

- Works with local checkpoints or remote Hub ids (`trust_remote_code=True`)
- Auto-selects GPU (`cuda`) when available, otherwise falls back to CPU
- Detects pre-quantised models to avoid double-quantising
- Supports `<think>` suppression for Qwen/Qwen2 style models when `enable_thinking=False`
- Exposes graceful cancellation through `stop_generation`

## Constructor

### `TransformersModel.__init__`

```python
from neurosurfer.config import config

TransformersModel(
    model_name: str = "openai/gpt-oss-20b",
    *,
    max_seq_length: int = config.base_model.max_seq_length,
    load_in_4bit: bool = config.base_model.load_in_4bit,
    enable_thinking: bool = config.base_model.enable_thinking,
    stop_words: list[str] | None = config.base_model.stop_words,
    verbose: bool = config.base_model.verbose,
    logger: logging.Logger = logging.getLogger(),
    **kwargs,
)
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `model_name` | `str` | `"openai/gpt-oss-20b"` | Hugging Face id or local path passed to `from_pretrained`. |
| `max_seq_length` | `int` | `config.base_model.max_seq_length` | Logical context window (recorded on the instance). |
| `load_in_4bit` | `bool` | `config.base_model.load_in_4bit` | Enable 4-bit quantisation via `BitsAndBytesConfig` if the model is not already quantised. |
| `enable_thinking` | `bool` | `config.base_model.enable_thinking` | When `False`, `<think>` sections are stripped and Qwen-style prompts append `/nothink`. |
| `stop_words` | `list[str] \| None` | `config.base_model.stop_words` | Optional stop sequence list enforced client-side. |
| `verbose` | `bool` | `config.base_model.verbose` | Emit additional logging during model load and streaming. |
| `logger` | `logging.Logger` | `logging.getLogger()` | Logger shared with helper methods. |
| `**kwargs` | any | – | Forwarded to `AutoModelForCausalLM.generate`. |

During initialisation:

1. Device (`cuda` vs `cpu`) and dtype are selected automatically.
2. `AutoTokenizer.from_pretrained` and `AutoModelForCausalLM.from_pretrained` are called with `trust_remote_code=True`.
3. If `load_in_4bit` is `True`, a `BitsAndBytesConfig` is attached unless the checkpoint already contains `quantization_config`.

!!! tip
    `config` is imported from `neurosurfer.config`. Please see the [configuration](../../configuration.md#base-model-defaults) section for more details.

## Usage

### Basic completion

```python
from neurosurfer.models.chat_models.transformers import TransformersModel

model = TransformersModel(
    model_name="/home/nomi/workspace/Model_Weights/Qwen3-4B-unsloth-bnb-4bit",
    load_in_4bit=False,  # already quantised
)

reply = model.ask("Summarise the features of Neurosurfer in two sentences.")
print(reply.choices[0].message.content)
```

### Streaming completion

```python
stream = model.ask(
    "Provide three bullet points about vector databases.",
    system_prompt="Answer as a helpful engineer.",
    stream=True,
    temperature=0.3,
)

for chunk in stream:
    delta = chunk.choices[0].delta.content or ""
    print(delta, end="")
```

### Cancelling generation

```python
stream = model.ask("Write a very long story about transformers.", stream=True)

for chunk in stream:
    if should_cancel_now():
        model.stop_generation()
        break
    print(chunk.choices[0].delta.content, end="")
```

## Tips

- **Quantised checkpoints**: For weights already saved in 4-bit/8-bit format, leave `load_in_4bit=False`. The loader inspects `config.json` to avoid double quantisation.
- **Thinking tags**: Qwen/Qwen2 models output `<think>...</think>` content. Set `enable_thinking=False` (default) to strip it; set `True` to keep the reasoning trace.
- **Custom kwargs**: Pass `top_p`, `top_k`, `repetition_penalty`, etc. directly to `ask` and they propagate to `generate`.
- **Stop words**: Use `model.set_stop_words(["Observation:", "\nResult:"])` to truncate outputs before those markers.

## Related models

- [`UnslothModel`](unsloth-model.md) – similar API built on Unsloth's `FastLanguageModel`
- [`LlamaCppModel`](llamacpp-model.md) – run GGUF weights via llama.cpp

*mkdocstrings output is temporarily disabled while import hooks are updated.*
