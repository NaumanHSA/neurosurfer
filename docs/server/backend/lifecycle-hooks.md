---
title: Lifecycle Hooks
description: Startup and shutdown hooks for model warmup, caches, RAG orchestration, resource hygiene, graceful termination, and testability.
---

# Lifecycle Hooks

Neurosurfer exposes **decorator-driven lifecycle hooks** so you can prepare and tear down resources cleanly around the server’s life. Hooks are lightweight, composable, and support both **sync** and **async** functions.

- `@app.on_startup` — register one or more functions that run **once** when the app starts.  
- `@app.on_shutdown` — register functions that run **once** on **graceful termination** (SIGTERM / server stop).  
- `@app.stop_generation` — register a **cooperative cancellation** handler for in-flight generations.

You can attach multiple functions to each hook; they run in **registration order**. Both **sync** and **async** functions are supported.

---

## Minimal Example

```python
from neurosurfer.server import NeurosurferApp

app = NeurosurferApp()

@app.on_startup
async def warmup():
    # Create loggers, test GPU, load models, hydrate caches
    ...

@app.on_shutdown
def cleanup():
    # Close pools, delete temp dirs, flush telemetry
    ...

@app.stop_generation
def on_stop():
    # Cooperatively ask the active model(s) to stop
    ...
```

---

## Reference Example (End-to-End)

Below mirrors a production setup with model load, embedder + RAG orchestration, a warmup inference, cleanup, and a cooperative stop signal.

```python
from typing import List
import logging, shutil, os

from neurosurfer.server import NeurosurferApp
from neurosurfer.config import config

app = NeurosurferApp(
    app_name=config.app.app_name,
    cors_origins=config.app.cors_origins,
    enable_docs=config.app.enable_docs,
    host=config.app.host_ip,
    port=config.app.host_port,
    reload=config.app.reload,
    log_level=config.app.logs_level,
    workers=config.app.workers,
)

BASE_DIR = "./tmp/code_sessions"; os.makedirs(BASE_DIR, exist_ok=True)

LLM = None
EMBEDDER = None
LOGGER = logging.getLogger("neurosurfer")

@app.on_startup
async def load_everything():
    # 1) Logging & hardware probe
    logging.basicConfig(level=config.app.logs_level.upper())
    try:
        import torch
        LOGGER.info("GPU available: %s", torch.cuda.is_available())
        if torch.cuda.is_available():
            LOGGER.info("GPU name: %s", torch.cuda.get_device_name(0))
    except Exception:
        LOGGER.warning("Torch not available; running on CPU")

    # 2) Models (LLM + embedder)
    from neurosurfer.models.chat_models.transformers import TransformersModel
    from neurosurfer.models.embedders.sentence_transformer import SentenceTransformerEmbedder

    global LLM, EMBEDDER
    LLM = TransformersModel(
        model_name=config.model.model_name,
        max_seq_length=config.model.max_seq_length,
        load_in_4bit=getattr(config.model, "load_in_4bit", False),
        enable_thinking=getattr(config.model, "enable_thinking", False),
        stop_words=config.model.stop_words,
        logger=LOGGER,
    )

    # 3) Model registry (for UI pickers and client feature gating)
    app.model_registry.add(
        llm=LLM,
        family="qwen3",
        provider="Qwen",
        description="Local Transformers model",
    )

    EMBEDDER = SentenceTransformerEmbedder(
        model_name=config.model.embedder, logger=LOGGER
    )

    # 4) Warmup inference (reduces first-token latency)
    ping = LLM.ask(
        user_prompt="ping", system_prompt=config.model.system_prompt, stream=False
    )
    LOGGER.info("LLM warmup OK: %s", ping.choices[0].message.content)

@app.on_shutdown
def tear_down():
    # Always resilient: do not raise; log and continue.
    shutil.rmtree(BASE_DIR, ignore_errors=True)
    LOGGER.info("Cleaned temp workspace at %s", BASE_DIR)

@app.stop_generation
def stop_handler():
    # Called when a user/client requests to stop a running generation
    if LLM:
        LLM.stop_generation()
```

---

!!! tip
    Startup must be **fast**. If something heavy can be deferred, do it lazily on the **first request** (but keep a small warmup for UX).

## Multiple Hooks & Ordering

You can register **several** startup/shutdown functions. They run in the order you declare them:

```python
@app.on_startup
def init_logging(): ...

@app.on_startup
async def init_models(): ...

@app.on_startup
def init_caches(): ...
```

---

## Async vs Sync

- Use **async** when calling network I/O (object storage, vector DB, model gateway).  
- Use **sync** for pure CPU initialization (loading local weights, creating directories).  
- Avoid long `await` chains that can stall the server; consider **background tasks** after the app is accepting traffic.

---

## Graceful Shutdown & In-Flight Streams

Streaming clients (SSE) deserve graceful termination:

1. Register a `@app.stop_generation` handler that calls your model’s cooperative **stop**.  
2. On shutdown, the server sends stop signals, then waits briefly for streams to complete.  
3. Keep proxy **read timeouts** high enough so clients get the final event rather than a TCP reset (see *Deployment*).

```python
@app.stop_generation
def stop_all():
    if LLM:
        LLM.stop_generation()
```

---

## Troubleshooting

- **Server boots but first request is slow** → add a **warmup** call in startup.  
- **Hangs on shutdown** → ensure cooperative `@stop_generation` handler and raise proxy timeouts.  
- **RAG never triggers** → inspect thresholds and log top similarity; adjust implicit/explicit gates.  
- **Out of memory on startup** → lower quantization/precision or delay heavy loads to **first request**.

---
