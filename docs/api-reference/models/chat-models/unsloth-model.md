# UnslothModel

**Module:** `neurosurfer.models.chat_models.unsloth`  
**Inherits:** [`BaseChatModel`](base-model.md)

## Overview

`UnslothModel` integrates Unsloth's `FastLanguageModel` runtime, giving you accelerated inference for LoRA/QLoRA checkpoints without rewriting application code. It mirrors the `TransformersModel` interface but leans on Unsloth's optimisations for NVIDIA GPUs.

### Highlights

- Optimised CUDA kernels with optional 4-bit/8-bit quantisation
- Thread-safe streaming with stop signal support
- Optional `<think>` suppression for Qwen-style models
- Compatible with checkpoints produced by the Unsloth finetuning workflow

## Constructor

### `UnslothModel.__init__`

```python
from neurosurfer.config import config

UnslothModel(
    model_name: str,
    *,
    max_seq_length: int = config.base_model.max_seq_length,
    load_in_4bit: bool = config.base_model.load_in_4bit,
    load_in_8bit: bool = False,
    full_finetuning: bool = config.base_model.full_finetuning,
    enable_thinking: bool = config.base_model.enable_thinking,
    stop_words: list[str] | None = config.base_model.stop_words,
    verbose: bool = config.base_model.verbose,
    logger: logging.Logger = logging.getLogger(),
    **kwargs,
)
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `model_name` | `str` | – | Local path or Hugging Face id recognised by `FastLanguageModel`. |
| `max_seq_length` | `int` | `config.base_model.max_seq_length` | Context window passed to Unsloth; must align with how the model was trained. |
| `load_in_4bit` | `bool` | `config.base_model.load_in_4bit` | Load weights in 4-bit mode for memory savings. |
| `load_in_8bit` | `bool` | `False` | Load weights in 8-bit mode instead of 4-bit. |
| `full_finetuning` | `bool` | `config.base_model.full_finetuning` | Enable full-parameter finetuning mode (not needed for inference). |
| `enable_thinking` | `bool` | `config.base_model.enable_thinking` | Keep `<think>` reasoning traces instead of stripping them. |
| `stop_words` | `list[str] or None` | `config.base_model.stop_words` | Optional stop sequence list enforced client-side. |
| `verbose` | `bool` | `config.base_model.verbose` | Emit additional logs from the wrapper. |
| `logger` | `logging.Logger` | `logging.getLogger()` | Logger shared across helper methods. |

Additional keyword arguments are forwarded to `FastLanguageModel.from_pretrained`.

!!! tip
    `config` is imported from `neurosurfer.config`. Please see the [configuration](../../configuration.md#base-model-defaults) section for more details.

## Usage

### Non-streaming reply

```python
from neurosurfer.models.chat_models.unsloth import UnslothModel

model = UnslothModel(
    model_name="/weights/Qwen2.5-7B-Instruct-bnb-4bit",
    enable_thinking=False,
)

response = model.ask("Summarise the README in two bullet points.")
print(response.choices[0].message.content)
```

### Streaming with stop control

```python
stream = model.ask(
    "List three benefits of Unsloth.",
    stream=True,
    temperature=0.6,
)

for chunk in stream:
    delta = chunk.choices[0].delta.content or ""
    print(delta, end="")
    if "<END>" in delta:
        model.stop_generation()
```

### Updating stop words at runtime

```python
model.set_stop_words(["\nObservation:"])
reply = model.ask("Respond using the format 'Answer: ...' and 'Observation: ...'.")
print(reply.choices[0].message.content)
```

## Notes

- Unsloth currently targets CUDA; running on CPU is not supported.
- The wrapper uses a background thread plus `TextIteratorStreamer` to support streaming and stop signals—call `stop_generation()` to cancel long generations.
- Token counts are approximated via the tokenizer when possible; failures fall back to a word-count heuristic.

## Related models

- [`TransformersModel`](transformers-model.md) – direct Hugging Face integration
- [`LlamaCppModel`](llamacpp-model.md) – run GGUF weights via llama.cpp

*mkdocstrings output is temporarily disabled while import hooks are updated.*
