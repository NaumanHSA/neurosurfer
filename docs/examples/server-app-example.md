---
title: Server Application
description: A Neurosurfer server app, explained step‑by‑step—app init, startup wiring, RAG, chat handler, shutdown, cooperative stop, and running the server.
---

# Server Application

This is a **guided walk‑through** of a Neurosurfer server app. Instead of one giant snippet, we’ll build it piece by piece and explain how each section fits together. At the end, you’ll find the complete example in one block for easy copy‑paste.

> See also: [Lifecycle Hooks](../server/backend/lifecycle-hooks.md) • [Chat Handlers](../server/backend/chat-handlers.md) • [Configuration](../api-reference/configuration.md) • [Auth & Users](../server/backend/auth.md)

---

## App Initialization

Create the app from configuration so ports, origins, logging, and worker counts stay environment‑driven—not hard‑coded.

```python
import os, logging
from neurosurfer.server import NeurosurferApp
from neurosurfer.config import config

logging.basicConfig(level=config.app.logs_level.upper())

nm = NeurosurferApp(
    app_name=config.app.app_name,
    api_keys=[],  # add static API keys here if you need header-based auth
    enable_docs=config.app.enable_docs,
    cors_origins=config.app.cors_origins,
    host=config.app.host_ip,
    port=config.app.host_port,
    reload=config.app.reload,
    log_level=config.app.logs_level,
    workers=config.app.workers
)
```

!!! tip
    Configure CORS in **Configuration** if your frontend runs on a different origin. That’s often the cause of “works with curl, fails in browser.”

---

## Global State

Keep shared components explicit—model, embedder, logger, RAG, and a temp directory for file ops.

```python
from neurosurfer.models.embedders.base import BaseEmbedder
from neurosurfer.models.chat_models import BaseChatModel as BaseChatModel
from neurosurfer.server.services.rag_orchestrator import RAGOrchestrator

BASE_DIR = "./tmp/code_sessions"; os.makedirs(BASE_DIR, exist_ok=True)
LLM: BaseChatModel = None
EMBEDDER_MODEL: BaseEmbedder = None
LOGGER: logging.Logger = None
RAG: RAGOrchestrator | None = None
```

!!! note
    Neurosurfer keeps the surface minimal: the handler orchestrates; heavy lifting lives in services (models, RAG, DB).

---

## Startup Hook

Load the chat model and embedder, boot the RAG orchestrator, register the model for `/v1/models`, and warm up for low first‑token latency.

```python
from neurosurfer.rag.chunker import Chunker
from neurosurfer.rag.filereader import FileReader

@nm.on_startup
async def load_model():
    global EMBEDDER_MODEL, LOGGER, LLM, RAG
    LOGGER = logging.getLogger("neurosurfer")

    try:
        import torch
        LOGGER.info(f"GPU Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            LOGGER.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
    except Exception:
        LOGGER.warning("Torch not found...")

    from neurosurfer.models.chat_models.transformers import TransformersModel
    from neurosurfer.models.embedders.sentence_transformer import SentenceTransformerEmbedder

    LLM = TransformersModel(
        model_name="/path/to/weights/Qwen3-4B-unsloth-bnb-4bit",
        max_seq_length=8192,
        load_in_4bit=True,
        enable_thinking=False,
        stop_words=[],
        logger=LOGGER,
    )

    nm.model_registry.add(
        llm=LLM,
        family="qwen3",
        provider="Qwen",
        description="Local Qwen3 served by Transformers backend"
    )

    EMBEDDER_MODEL = SentenceTransformerEmbedder(
        model_name="/path/to/weights/e5-large-v2",
        logger=LOGGER
    )

    RAG = RAGOrchestrator(
        embedder=EMBEDDER_MODEL,
        chunker=Chunker(),
        file_reader=FileReader(),
        persist_dir=config.app.database_path,
        max_context_tokens=2000,
        top_k=15,
        min_top_sim_default=0.35,
        min_top_sim_when_explicit=0.15,
        min_sim_to_keep=0.20,
        logger=LOGGER,
    )

    # Warmup
    ready = LLM.ask(user_prompt="Say hi!", system_prompt=config.model.system_prompt, stream=False)
    LOGGER.info(f"LLM ready: {ready.choices[0].message.content}")
```

!!! tip
    Registering the model exposes it via `GET /v1/models`—handy for UIs and clients to pick a model by id.

---

## Chat Handler

The heart of the app. It reads the request, compacts history, optionally applies RAG per thread, and calls `LLM.ask(...)`. The same call supports streaming and non‑streaming.

```python
from typing import List, Generator
from neurosurfer.server.schemas import ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChunk
from neurosurfer.server.runtime import RequestContext

@nm.chat()
def handler(request: ChatCompletionRequest, ctx: RequestContext) -> ChatCompletionResponse | Generator[ChatCompletionChunk, None, None]:
    global LLM, RAG

    actor_id = (getattr(ctx, "meta", {}) or {}).get("actor_id", 0)
    thread_id = request.thread_id

    user_msgs: List[str] = [m["content"] for m in request.messages if m["role"] == "user"]
    system_msgs = [m["content"] for m in request.messages if m["role"] == "system"]
    system_prompt = system_msgs[0] if system_msgs else config.model.system_prompt
    user_query = user_msgs[-1] if user_msgs else ""
    chat_history = request.messages[-10:-1]

    temperature = request.temperature if (request.temperature and 2 > request.temperature > 0) else config.model.temperature
    max_tokens = request.max_tokens if (request.max_tokens and request.max_tokens > 512) else config.model.max_new_tokens

    if RAG and thread_id is not None:
        rag_res = RAG.apply(
            actor_id=actor_id,
            thread_id=thread_id,
            user_query=user_query,
            files=[f.model_dump() for f in (request.files or [])],
        )
        user_query = rag_res.augmented_query
        if rag_res.used:
            LOGGER.info(f"[RAG] used context (top_sim={rag_res.meta.get('top_similarity', 0):.3f})")

    return LLM.ask(
        user_prompt=user_query,
        system_prompt=system_prompt,
        chat_history=chat_history,
        stream=request.stream,
        temperature=temperature,
        max_new_tokens=max_tokens,
    )
```

!!! note
    The handler returns either a **single completion** or a **generator of SSE chunks**, depending on `request.stream`. The router already formats tokens as OpenAI‑style delta chunks.

---

## Shutdown Cleanup

Remove transient state so subsequent restarts are clean.

```python
import shutil

@nm.on_shutdown
def cleanup():
    shutil.rmtree(BASE_DIR, ignore_errors=True)
```

---

## Cooperative Stop

Expose a stop hook so UIs can cancel long generations gracefully.

```python
@nm.stop_generation()
def stop_handler():
    global LLM
    LLM.stop_generation()
```

---

## Full Example (All Together)

```python
from typing import List, Generator
import os, shutil, logging

from neurosurfer.models.embedders.base import BaseEmbedder
from neurosurfer.models.chat_models import BaseChatModel as BaseChatModel
from neurosurfer.rag.chunker import Chunker
from neurosurfer.rag.filereader import FileReader

from neurosurfer.server import NeurosurferApp
from neurosurfer.server.schemas import ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChunk
from neurosurfer.server.runtime import RequestContext
from neurosurfer.server.services.rag_orchestrator import RAGOrchestrator

from neurosurfer.config import config
logging.basicConfig(level=config.app.logs_level.upper())

nm = NeurosurferApp(
    app_name=config.app.app_name,
    api_keys=[],
    enable_docs=config.app.enable_docs,
    cors_origins=config.app.cors_origins,
    host=config.app.host_ip,
    port=config.app.host_port,
    reload=config.app.reload,
    log_level=config.app.logs_level,
    workers=config.app.workers
)

BASE_DIR = "./tmp/code_sessions"; os.makedirs(BASE_DIR, exist_ok=True)
LLM: BaseChatModel = None
EMBEDDER_MODEL: BaseEmbedder = None
LOGGER: logging.Logger = None
RAG: RAGOrchestrator | None = None

@nm.on_startup
async def load_model():
    global EMBEDDER_MODEL, LOGGER, LLM, RAG
    LOGGER = logging.getLogger("neurosurfer")

    try:
        import torch
        LOGGER.info(f"GPU Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            LOGGER.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
    except Exception:
        LOGGER.warning("Torch not found...")

    from neurosurfer.models.chat_models.transformers import TransformersModel
    from neurosurfer.models.embedders.sentence_transformer import SentenceTransformerEmbedder

    LLM = TransformersModel(
        model_name="/path/to/weights/Qwen3-4B-unsloth-bnb-4bit",
        max_seq_length=8192,
        load_in_4bit=True,
        enable_thinking=False,
        stop_words=[],
        logger=LOGGER,
    )

    nm.model_registry.add(
        llm=LLM,
        family="qwen3",
        provider="Qwen",
        description="Local Qwen3 served by Transformers backend"
    )

    EMBEDDER_MODEL = SentenceTransformerEmbedder(
        model_name="/path/to/weights/e5-large-v2",
        logger=LOGGER
    )

    RAG = RAGOrchestrator(
        embedder=EMBEDDER_MODEL,
        chunker=Chunker(),
        file_reader=FileReader(),
        persist_dir=config.app.database_path,
        max_context_tokens=2000,
        top_k=15,
        min_top_sim_default=0.35,
        min_top_sim_when_explicit=0.15,
        min_sim_to_keep=0.20,
        logger=LOGGER,
    )

    ready = LLM.ask(user_prompt="Say hi!", system_prompt=config.model.system_prompt, stream=False)
    LOGGER.info(f"LLM ready: {ready.choices[0].message.content}")

@nm.on_shutdown
def cleanup():
    shutil.rmtree(BASE_DIR, ignore_errors=True)

@nm.chat()
def handler(request: ChatCompletionRequest, ctx: RequestContext) -> ChatCompletionResponse | Generator[ChatCompletionChunk, None, None]:
    global LLM, RAG

    actor_id = (getattr(ctx, "meta", {}) or {}).get("actor_id", 0)
    thread_id = request.thread_id

    user_msgs: List[str] = [m["content"] for m in request.messages if m["role"] == "user"]
    system_msgs = [m["content"] for m in request.messages if m["role"] == "system"]
    system_prompt = system_msgs[0] if system_msgs else config.model.system_prompt
    user_query = user_msgs[-1] if user_msgs else ""
    chat_history = request.messages[-10:-1]

    temperature = request.temperature if (request.temperature and 2 > request.temperature > 0) else config.model.temperature
    max_tokens = request.max_tokens if (request.max_tokens and request.max_tokens > 512) else config.model.max_new_tokens

    if RAG and thread_id is not None:
        rag_res = RAG.apply(
            actor_id=actor_id,
            thread_id=thread_id,
            user_query=user_query,
            files=[f.model_dump() for f in (request.files or [])],
        )
        user_query = rag_res.augmented_query

    return LLM.ask(
        user_prompt=user_query,
        system_prompt=system_prompt,
        chat_history=chat_history,
        stream=request.stream,
        temperature=temperature,
        max_new_tokens=max_tokens,
    )

@nm.stop_generation()
def stop_handler():
    global LLM
    LLM.stop_generation()

if __name__ == "__main__":
    nm.run()
```