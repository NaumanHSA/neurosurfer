# LlamaCppModel

**Module:** `neurosurfer.models.chat_models.llamacpp`  
**Inherits:** [`BaseModel`](base-model.md)

## Overview

`LlamaCppModel` loads GGUF checkpoints through the `llama-cpp-python` bindings. It enables fast local inference on CPU or GPU with a pure Python interface while retaining the standard Neurosurfer response format.

### Highlights

- Supports both local GGUF files and Hugging Face repositories
- Works on CPU-only systems or with CUDA/OpenCL acceleration via `n_gpu_layers`
- Streams tokens incrementally with stop-word enforcement and cancellation support
- Tracks prompt/completion token usage when provided by llama.cpp

## Constructor

### `LlamaCppModel.__init__`

```python
LlamaCppModel(
    model_path: str | None = None,
    *,
    repo_id: str | None = None,
    filename: str | None = None,
    n_ctx: int = 2048,
    n_threads: int = 4,
    main_gpu: int = 0,
    n_gpu_layers: int = -1,
    max_seq_length: int = 2048,
    stop_words: list[str] | None = None,
    verbose: bool = False,
    logger: logging.Logger = logging.getLogger(),
    **kwargs,
)
```

| Parameter | Type | Description |
| --- | --- | --- |
| `model_path` | `str \| None` | Path to a local `.gguf` file. Required unless `repo_id` + `filename` is provided. |
| `repo_id` | `str \| None` | Hugging Face repo id for `Llama.from_pretrained`. |
| `filename` | `str \| None` | Specific filename within the repo (usually the GGUF file). |
| `n_ctx` | `int` | Context window size passed to llama.cpp (`max_seq_length` mirrors this). |
| `n_threads` | `int` | CPU worker threads. |
| `main_gpu` | `int` | Primary GPU index when offloading layers. |
| `n_gpu_layers` | `int` | Number of layers to place on GPU (`-1` = as many as possible). |
| `stop_words` | `list[str] \| None` | Optional list of stop words enforced client-side. |
| `verbose` | `bool` | Enable verbose llama.cpp logging. |
| `logger` | `logging.Logger` | Logger used for wrapper messages. |

If `model_path` ends with `.gguf`, the file is loaded directly. Otherwise supply `repo_id` and `filename` to download and cache automatically.

## Usage

### Basic completion

```python
from neurosurfer.models.chat_models.llamacpp import LlamaCppModel

model = LlamaCppModel(
    model_path="/weights/llama-2-7b-chat.Q4_K_M.gguf",
    n_ctx=4096,
    n_threads=8,
)

response = model.ask("Explain retrieval-augmented generation.")
print(response.choices[0].message.content)
```

### Streaming with GPU acceleration

```python
gpu_model = LlamaCppModel(
    model_path="/weights/Qwen2.5-7B-Instruct-Q3_K_L.gguf",
    n_ctx=4096,
    n_gpu_layers=40,  # offload as many layers as possible
)

stream = gpu_model.ask("List three pros of llama.cpp.", stream=True)
for chunk in stream:
    delta = chunk.choices[0].delta.content or ""
    print(delta, end="")
```

### Loading from Hugging Face

```python
hf_model = LlamaCppModel(
    repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    filename="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    n_ctx=4096,
)
```

## Tips

- **Stop words**: `model.set_stop_words(["Observation:", "\nFinal answer:"])` prevents the response from emitting tool markers.
- **Cancellation**: Call `model.stop_generation()` while streaming to interrupt long generations.
- **GPU selection**: `main_gpu` selects the device; use `CUDA_VISIBLE_DEVICES` when running inside multi-GPU nodes.
- **Thread count**: On CPU-only setups tune `n_threads` to match available cores for optimal throughput.

## Related models

- [`TransformersModel`](transformers-model.md) – load PyTorch checkpoints directly
- [`UnslothModel`](unsloth-model.md) – GPU-optimised inference via Unsloth

*mkdocstrings output is temporarily disabled while import hooks are updated.*
