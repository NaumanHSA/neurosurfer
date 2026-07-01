# Hooks

Hooks intercept the gateway's request/response lifecycle — for auth, prompt injection, logging,
redaction, or reshaping payloads. A `Hook` overrides any of three methods:

```python
from neurosurfer.app.server import Hook, HookContext

class Hook:
    async def before_chat(self, ctx: HookContext, req: dict) -> dict: ...   # mutate the request
    async def after_chat(self, ctx: HookContext, resp: dict) -> dict: ...    # mutate a full response
    async def stream_chunk(self, ctx: HookContext, chunk: dict) -> dict: ... # mutate each SSE chunk
```

Each method returns the (possibly modified) payload. `HookContext` carries `request_id`, `model`,
and optional `user` / `client_ip`.

Register hooks in order; they run as a pipeline:

```python
server.add_hook(MyAuthHook())
server.add_hook(MyLoggingHook())
```

## Built-in hooks

Two ready-made hooks ship in `neurosurfer.app.server`:

```python
from neurosurfer.app.server import StripReasoningHook, SystemPromptInjectorHook

# Remove <think>…</think> reasoning from responses and streamed chunks:
server.add_hook(StripReasoningHook())

# Prepend a system prompt to every request:
server.add_hook(SystemPromptInjectorHook("You are a concise, helpful assistant."))
```

## A custom hook

```python
from neurosurfer.app.server import Hook, HookContext

class TagResponsesHook(Hook):
    async def before_chat(self, ctx: HookContext, req: dict) -> dict:
        # e.g. enforce a max token budget
        req.setdefault("max_tokens", 1024)
        return req

    async def after_chat(self, ctx: HookContext, resp: dict) -> dict:
        resp["system_fingerprint"] = "neurosurfer-gateway"
        return resp

server.add_hook(TagResponsesHook())
```

## Auth

Two ways to require credentials:

- **Simple bearer keys** — pass `api_keys=[...]` to `NeurosurferServer(...)`; requests must present a
  matching `Authorization: Bearer <key>`.
- **Custom auth** — implement it in a `before_chat` hook (inspect `ctx.user` / headers you attach,
  look up a database, raise `OpenAIHTTPError` to reject) for anything beyond static keys.

```python
from neurosurfer.app.server import OpenAIHTTPError

class ApiKeyHook(Hook):
    async def before_chat(self, ctx: HookContext, req: dict) -> dict:
        if not _is_authorized(ctx.user):
            raise OpenAIHTTPError(status_code=401, message="Unauthorized")
        return req
```
