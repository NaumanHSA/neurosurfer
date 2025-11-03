---
title: Neurosurfer API (Backend)
description: The Neurosurfer backend—OpenAI‑compatible chat API, decorators for handlers and endpoints, auth, and RAG integration.
---

# Neurosurfer API (Backend)

Neurosurfer’s backend is a **FastAPI** server that exposes an **OpenAI‑compatible** surface for chat completions and a small set of ergonomic decorators to register your logic. You get streaming, JWT/cookie auth, a model registry, thread‑scoped RAG, and typed custom routes—without wiring FastAPI by hand.

---

## What you’ll find here

- A compact **request flow** for `/v1/chat/completions` (sync/async, streaming or not)
- **Decorator‑driven** APIs: `@app.chat()` and `@app.endpoint(...)`
- Built‑in **auth** (bcrypt + JWT) that works for browsers (cookies) and API clients (bearer)
- Integration points for **RAG** and model registries

For deeper topics, jump directly to the dedicated pages linked below.

---

## Components

<div class="grid cards" markdown>

-   **Configuration**

    ---

    Centralized settings via Pydantic (models, CORS, DB paths, RAG, server flags). Start here to wire your environment.

    [:octicons-arrow-right-24: Documentation](../api-reference/configuration.md)

-   **Lifecycle Hooks**

    ---

    Start/stop callbacks to load models, warm resources, initialize RAG, and clean up.

    [:octicons-arrow-right-24: Documentation](./lifecycle-hooks.md)

-   **Chat Handlers**

    ---

    Your entry point for `/v1/chat/completions`. Register once, stream or return a final message, and compose RAG/services inside.

    [:octicons-arrow-right-24: Documentation](./chat-handlers.md)

-   **Custom Endpoints**

    ---

    Add typed REST routes with request/response models and dependencies—perfect for utilities, admin, and service façades.

    [:octicons-arrow-right-24: Documentation](./custom-endpoints.md)

-   **Auth & Users**

    ---

    bcrypt passwords, JWT tokens (header or HttpOnly cookie), and slim dependencies for required/optional auth.

    [:octicons-arrow-right-24: Documentation](./auth.md)

</div>

---

## Quick start

### Minimal handler

```python
from neurosurfer.server.app import NeurosurferApp

app = NeurosurferApp()

@app.chat()
def handle_chat(request, ctx):
    return f"You said: {request.messages[-1]['content']}"
```

### With a model registry

```python
app = NeurosurferApp()

app.model_registry.add(
    id="qwen3",
    family="Qwen",
    provider="Qwen",
    context_length=8192,
)

@app.chat()
async def handle_chat(request, ctx):
    # stream an answer (pseudo)
    for token in generate_tokens(request):
        yield {"choices":[{"delta":{"content": token}}]}
```

### With RAG orchestration

```python
from neurosurfer.server.services.rag_orchestrator import RAGOrchestrator

rag = RAGOrchestrator(embedder=your_embedder, persist_dir="./vector_store", top_k=8)

@app.chat()
def handle_chat(request, ctx):
    user_query = request.messages[-1]["content"]
    # rag.apply(...) may augment the query based on thread uploads
    # then your LLM answers with the augmented prompt
    return your_llm.generate(user_query)
```

---

## Where next?

- **Handlers:** learn streaming, request shapes, and optional tool prompts → [Chat Handlers](./chat-handlers.md)  
- **Typed routes:** build utility/admin APIs → [Custom Endpoints](./custom-endpoints.md)  
- **Auth:** headers, cookies, dependencies → [Auth & Users](./auth.md)  
- **Boot sequence:** models, RAG, warmups → [Lifecycle Hooks](./lifecycle-hooks.md)  
- **Configure it all:** env vars, CORS, ports → [Configuration](../api-reference/configuration.md)