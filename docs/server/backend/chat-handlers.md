---
title: Chat Handlers
description: The Neurosurfer chat handler – endpoint contract, streaming, request/response models, what you can do inside the handler, and a complete example. Includes an optional, minimal tooling pattern.
---

# Chat Handlers

The **chat handler** is the core extension point of the backend. With a single decorator, you implement how your app responds to `/v1/chat/completions` in both **streaming** and **non‑streaming** modes. Handlers are intentionally thin and composable: use them to orchestrate models, apply RAG, enforce policy, and shape the final response.

---

## Why Chat Handlers?

- **OpenAI‑compatible** surface: clients and SDKs integrate without custom glue.
- **One place for logic**: prompt strategy, RAG, safety checks, and formatting live together.
- **Streaming‑first UX**: return tokens as they’re generated (SSE) or a single final message.
- **Minimal ceremony**: sync or async function; return string, dict, or a generator.
- **Composability**: call services (RAG, toolkits, DB) without leaking internals to clients.

---

## Endpoint

- **URL:** `/v1/chat/completions`  
- **Method:** `POST`  
- **Auth:** API key header or cookie‑based session (see Auth)  
- **Streaming:** `stream=true` → Server‑Sent Events (SSE)

---

## Request Model (Essentials)

| Field | Type | Notes |
|---|---|---|
| `model` | string | `required` Model identifier from the server model registry. |
| `messages` | array | `required` OpenAI‑style chat messages (`role`, `content`). |
| `stream` | boolean | `true` enables SSE token streaming. |
| `temperature` | number | 0–2; defaults from configuration. |
| `max_tokens` | integer | Generation cap; defaults from configuration. |
| `thread_id` | string | Scopes RAG and uploads to a conversation thread. |
| `files` | array | Base64 attachments for RAG ingestion. |
| `stop` | string/array | Stop sequences. |

### Message Format

Minimal example:

```json
{
  "messages": [
    { "role": "system", "content": "You are a helpful assistant." },
    { "role": "user", "content": "Summarize the uploaded PDF." }
  ]
}
```

Roles allowed: `system`, `user`, `assistant`, `tool` (your app may add `tool` messages as part of a follow‑up round).

---

## Response Model

### Non‑Streaming

```json
{
  "id": "cmpl_...",
  "object": "chat.completion",
  "created": 1710000000,
  "model": "qwen3",
  "choices": [{
    "index": 0,
    "message": { "role": "assistant", "content": "Hello!" },
    "finish_reason": "stop"
  }],
  "usage": { "prompt_tokens": 25, "completion_tokens": 7, "total_tokens": 32 }
}
```

### Streaming (SSE)

- **Content‑Type:** `text/event-stream`  
- Server emits JSON **chunks** until `[DONE]`:

```
event: message
data: {"object":"chat.completion.chunk","choices":[{"delta":{"content":"Hel"}}]}

event: message
data: {"object":"chat.completion.chunk","choices":[{"delta":{"content":"lo"}}]}

data: [DONE]
```

Concatenate `choices[].delta.content` to reconstruct the final text.

---

## What Can I Do Inside the Handler?

- **Read the request**: messages, `stream`, sampling params, `thread_id`, `files`.
- **Shape prompts**: pick a `system_prompt`, extract the latest user input, trim history.
- **Run RAG**: ingest thread files, retrieve relevant chunks, **augment** the user query.
- **Call your model**: via `LLM.ask(user_prompt=..., system_prompt=..., stream=...)`.
- **Stream**: yield OpenAI‑compatible **delta** chunks as tokens arrive.
- **Respect stop**: integrate with your model’s cooperative `stop_generation` handler.
- **Return types**: a **string**, an OpenAI‑shaped **dict**, or a **generator** (for SSE).
- **Log & meter**: attach `op_id`, `thread_id`, timing, token counts, RAG usage.

!!! tip
    Keep the handler **thin**. Put heavy lifting (RAG indexing, DB work, tool execution) into services. The handler glues these together and formats the response.

---

## Complete Example (Sync; streaming‑aware)

```python
from typing import List
from neurosurfer.server import NeurosurferApp
from neurosurfer.server.schemas import ChatCompletionRequest
from neurosurfer.server.runtime import RequestContext
from neurosurfer.config import config

app = NeurosurferApp()

# Optional: RAG orchestrator/service may be initialized in startup hooks
RAG = None
LLM = None

@app.on_startup
def boot():
    global LLM, RAG
    # Load your model(s), embedder, and RAG orchestrator here.
    # Add the model to app.model_registry for client discovery.
    ...

@app.chat()
def handle_chat(request: ChatCompletionRequest, ctx: RequestContext):
    # 1) Extract prompts
    user_msgs = [m["content"] for m in request.messages if m["role"] == "user"]
    system_msgs = [m["content"] for m in request.messages if m["role"] == "system"]
    user_prompt = user_msgs[-1] if user_msgs else ""
    system_prompt = system_msgs[-1] if system_msgs else config.model.system_prompt

    # 2) Optional: compact recent chat history
    chat_history = request.messages[-10:-1]

    # 3) Optional: RAG augmentation per thread
    if RAG and request.thread_id is not None:
        rag_res = RAG.apply(
            actor_id=(getattr(ctx, "meta", {}) or {}).get("actor_id", 0),
            thread_id=request.thread_id,
            user_query=user_prompt,
            files=[f.model_dump() for f in (request.files or [])]
        )
        user_prompt = rag_res.augmented_query  # falls back to original if not used

    # 4) Model call (stream or not)
    kwargs = {
        "temperature": request.temperature or config.model.temperature,
        "max_new_tokens": request.max_tokens or config.model.max_new_tokens,
        "stream": request.stream,
    }

    result = LLM.ask(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        chat_history=chat_history,
        **kwargs
    )

    # 5) Return directly for non-streaming
    if not request.stream:
        return result

    # 6) For streaming, wrap yielded tokens into OpenAI delta chunks if needed
    def to_stream(gen):
        for chunk in gen:
            # If your model already yields OpenAI-shaped deltas, just yield them.
            # Otherwise adapt each token to {"choices":[{"delta":{"content": token}}]}
            yield chunk
    return to_stream(result)
```

---

## Error Handling

Return compact, user‑friendly errors (4xx for client, 5xx for server). In streaming mode, send an error event and close cleanly.

```json
{
  "error": { "code": "bad_request", "message": "Missing 'messages'" }
}
```

---

## Observability

Track: `op_id`, `actor_id`, `thread_id`, `model`, `stream`, `latency_ms`, `rag_used`, token usage, and error rates. Export metrics for dashboards and alerts.

---

## Optional: Minimal Tooling Pattern (Prompt‑Injected)

Tooling is **optional**. If you add a toolkit at startup, you may allow clients to pass a **tool name** and **inputs**, or you can **prompt** the model with your tool catalog and ask it to use one tool. So there are two ways to use tools:

### 1) Explicit Tool Invocation (Deterministic)

**When to use:** Buttons, admin flows, or any UX where you want a specific tool to run with known inputs—no model reasoning required.

**How it works:**

- Client sends `tool` and `tool_input` in the request payload.
- Handler looks up the tool in `toolkit.registry`, validates inputs via `tool.spec.check_inputs`, executes, and returns/streams the result.

```python
# Assume 'toolkit' and 'LLM' (and optionally 'EMBEDDER', 'RAG') were initialized in startup.
@app.chat()
def handler(request, ctx):
    tool_name = getattr(request, "tool", None)
    tool_input = getattr(request, "tool_input", {}) or {}

    if tool_name:
        tool = toolkit.registry.get(tool_name)
        if not tool:
            return {"error": {"code": "unknown_tool", "message": f"Tool '{tool_name}' not registered"}}

        try:
            data = tool.spec.check_inputs(tool_input)
        except Exception as e:
            return {"error": {"code": "invalid_tool_input", "message": str(e)}}

        out = tool(**data, llm=LLM, embedder=EMBEDDER, rag=RAG, stream=request.stream)
        obs = out.results

        if request.stream and hasattr(obs, "__iter__") and not isinstance(obs, (str, bytes)):
            for token in obs:
                yield {"choices": [{"delta": {"content": str(token)}}]}
        else:
            return {"choices": [{"message": {"role": "assistant", "content": str(obs)}}]}

    # …otherwise fall back to your standard chat flow (RAG + LLM).
    ...
```

### 2) Prompt-Driven Selection (Model Decides)

**When to use:** Natural language requests where the model should decide whether a tool is needed.

**Contract:** Inject a tool catalog into the system prompt (from toolkit.get_tools_description()), and instruct the model:

- If it can answer directly → output only the final answer.
- If it needs a tool → output only strict JSON (no backticks, no prose).

```json
{"call_tool": {"name": "<tool_name>", "args": {...}}}
```

**Minimal handler pattern:**

```python
import json

@app.chat()
def handler(request, ctx):
    user_last = next((m["content"] for m in reversed(request.messages) if m["role"] == "user"), "")
    history = request.messages[-10:-1]

    menu = toolkit.get_tools_description()
    system_prompt = (
        "You are Neurosurfer. Decide if a tool is needed.\n\n"
        "TOOLS:\n" + menu + "\n\n"
        "INSTRUCTIONS:\n"
        "- If you can answer directly, output ONLY the final answer text.\n"
        "- If you need a tool, output ONLY this JSON (no backticks, no extra text):\n"
        '  {\"call_tool\": {\"name\": \"<tool_name>\", \"args\": { ... }}}\n'
        "- Use exact parameter names/types from the tool spec."
    )

    # First pass (non-streaming) to inspect if tool JSON is returned
    first = LLM.ask(
        system_prompt=system_prompt,
        user_prompt=user_last,
        chat_history=history,
        temperature=request.temperature or 0.7,
        max_new_tokens=request.max_tokens or 1024,
        stream=False,
    )

    text = first.choices[0].message.content or ""
    try:
        call = json.loads(text).get("call_tool")
    except Exception:
        call = None

    # No tool requested → return (or optionally re-ask in streaming mode)
    if not call:
        if request.stream:
            return LLM.ask(
                system_prompt=system_prompt,
                user_prompt=user_last,
                chat_history=history,
                temperature=request.temperature or 0.7,
                max_new_tokens=request.max_tokens or 1024,
                stream=True,
            )
        return first

    # Tool requested → validate and run
    name, args = call.get("name"), (call.get("args") or {})
    tool = toolkit.registry.get(name)
    if not tool:
        return {"error": {"code": "unknown_tool", "message": f"Tool '{name}' is not registered"}}

    try:
        data = tool.spec.check_inputs(args)
    except Exception as e:
        return {"error": {"code": "invalid_tool_input", "message": str(e)}}

    out = tool(**data, llm=LLM, embedder=EMBEDDER, rag=RAG, stream=request.stream)
    obs = out.results

    if request.stream and hasattr(obs, "__iter__") and not isinstance(obs, (str, bytes)):
        for token in obs:
            yield {"choices": [{"delta": {"content": str(token)}}]}
    else:
        return {"choices": [{"message": {"role": "assistant", "content": str(obs)}}]}
```

**Notes & Tips**

- Keep the tool JSON contract strict to avoid ambiguous parsing.
- Use tool.spec.check_inputs(...) for strong validation (required fields, types, no extras).
- Cap multi-step tool loops if you extend this to ReAct-style behaviors (e.g., max 3 rounds).

---