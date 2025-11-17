---
title: Custom Endpoints
description: Define typed API routes with one decorator — request/response validation, dependency injection, and OpenAPI docs included.
---

# Custom Endpoints

Custom endpoints let you add **purpose‑built REST routes** next to chat completions without wiring FastAPI by hand. With a single decorator you get request parsing, response serialization, dependency execution, and OpenAPI documentation — all while keeping your business logic as a clean Python function. If you’ve used FastAPI before, this will feel familiar; `@app.endpoint(...)` simply standardizes the patterns we use across the Neurosurfer server.

---

## The Decorator

Use `@app.endpoint(path, method="post", request=..., response=..., dependencies=...)` to register a route. You can supply Pydantic models for the body and for the response, or omit them and return any JSON‑serializable value.

```python
@app.endpoint(
  path: str,
  *,
  method: "get" | "post" | "put" | "patch" | "delete" = "post",
  request: BaseChatModel | None = None,
  response: BaseChatModel | None = None,
  dependencies: list[Callable] | None = None,
)
def handler(...): ...
```

Under the hood, the decorator validates the HTTP verb, wraps your function in a small adapter, and attaches it to the main router with the appropriate `response_model` and `dependencies`. Handlers may be **sync or async**.

---

## Minimal Examples

### Health check (GET, no models)

A tiny route is just a function — no request/response models required:

```python
@app.endpoint("/health", method="get")
def health():
    return {"status": "ok"}
```

### Typed POST with request/response models

When you want strict validation and a documented schema, add Pydantic models. Neurosurfer will parse the body into your request type and serialize the return value to your response type.

```python
from pydantic import BaseModel

class SummarizeRequest(BaseModel):
    text: str
    max_length: int = 100

class SummarizeResponse(BaseModel):
    summary: str
    original_length: int
    summary_length: int

@app.endpoint(
    "/summarize",
    method="post",
    request=SummarizeRequest,
    response=SummarizeResponse
)
def summarize(req: SummarizeRequest):
    summary = summarize_fn(req.text, req.max_length)
    return SummarizeResponse(
        summary=summary,
        original_length=len(req.text),
        summary_length=len(summary),
    )
```

---

## Dependencies

Dependencies are regular FastAPI callables that run **before** your handler. They’re ideal for authentication, database sessions, rate limits, or feature flags. Add them with `dependencies=[...]` and keep the handler focused on the core logic.

```python
def require_api_key():
    # raise HTTPException(401) on failure
    ...

@app.endpoint("/admin/stats", method="get", dependencies=[require_api_key])
def admin_stats():
    return {"users": 120, "threads": 980}
```

---

## Request & Response Behavior

If you pass a `request=Model`, the incoming JSON body is validated and injected as the first parameter of your function. If you pass a `response=Model`, your return value is serialized to match that schema. Omit both and you can return plain JSON (dicts, lists, primitives). For custom status codes or headers, return a FastAPI `JSONResponse` or raise `HTTPException` — the decorator doesn’t constrain you.

Async is supported: declare `async def` and freely `await` network calls or I/O.

```python
@app.endpoint("/async-example", method="post", request=SummarizeRequest, response=SummarizeResponse)
async def async_summarize(req: SummarizeRequest):
    result = await summarize_async(req.text, req.max_length)
    return SummarizeResponse(summary=result, original_length=len(req.text), summary_length=len(result))
```

---

## Patterns You’ll Use Often

Small “utility” services (embeddings, text cleaning, conversions), lightweight admin panels (stats, cache purge), content operations (upload → process → return IDs), and model control hooks (warmup, reload). Keep each endpoint **small and pure**; move heavy lifting to services so the route stays testable.

---

!!! tip "Tip"
    Prefer Pydantic models when you want clear validation errors and self‑documenting APIs in `/docs`. Use `dependencies` instead of ad‑hoc checks so you can reuse auth and context setup. Name routes descriptively — they surface in OpenAPI, and good names make your API pleasant to explore. If you’ll call endpoints from a browser app, configure CORS in **Configuration**.


## End‑to‑End Example

Below is a compact endpoint that extracts keywords with simple auth, strong typing, and a clean response shape.

```python
from pydantic import BaseModel

class KeywordsIn(BaseModel):
    text: str
    top_k: int = 5

class KeywordsOut(BaseModel):
    keywords: list[str]

def require_user():
    # auth check, attach user to context if you maintain one
    ...

@app.endpoint(
    "/keywords",
    method="post",
    request=KeywordsIn,
    response=KeywordsOut,
    dependencies=[require_user],
)
def extract_keywords(data: KeywordsIn):
    kws = extract_keywords_fn(data.text, data.top_k)
    return KeywordsOut(keywords=kws)
```

---