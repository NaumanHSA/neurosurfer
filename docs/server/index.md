# Server / Gateway

`NeurosurferServer` is a drop-in **OpenAI-compatible gateway**. It exposes the routes every OpenAI
client already speaks — `/v1/models`, `/v1/chat/completions` (with SSE streaming), and `/health` —
and lets you serve two kinds of backends behind them:

- **[Upstream backends](backends.md#upstream-backends)** — proxy an existing OpenAI-compatible server
  (vLLM, LM Studio, Ollama, OpenAI).
- **[Native agent backends](backends.md#agent-backends)** — expose a Neurosurfer `AgenticLoop` /
  `ReactAgent` / `Agent` as a model ID.

Install the extra:

```bash
pip install "neurosurfer[serve]"
```

## Quickstart

```python
from neurosurfer.app.server import NeurosurferServer, UpstreamBackend

server = NeurosurferServer(port=8000)
server.register_backend(UpstreamBackend(name="local", base_url="http://localhost:8001/v1"))
server.run()   # blocking → http://localhost:8000/v1/chat/completions
```

Point any OpenAI client at it:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
resp = client.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(resp.choices[0].message.content)
```

## Serve an agent as a model

```python
from neurosurfer.agents import AgenticLoop
from neurosurfer.app.server import NeurosurferServer

server = NeurosurferServer()
server.register_agent(AgenticLoop(provider=provider), model_id="my-agent")
server.run()
```

Now `model="my-agent"` routes chat completions through your agent — streaming included. See
[Backends](backends.md) for the full registration API and [Hooks](hooks.md) for auth, prompt
injection, and response filtering.

## Configuration

`NeurosurferServer(...)` reads defaults from the environment / `.env` (via `ServerSettings`) and
applies any explicit overrides you pass. Common options:

| Argument | Purpose |
|---|---|
| `host`, `port` | Bind address (default `0.0.0.0:8000`). |
| `api_keys` | Require one of these bearer keys on requests. |
| `enable_docs` | Toggle the `/docs` UI. |
| `cors_origins`, `cors_allow_credentials` | CORS policy. |
| `reload`, `workers`, `log_level` | Uvicorn runtime settings. |

For ASGI deployment (Gunicorn/Uvicorn workers, containers), grab the FastAPI app with
`server.create_app()` instead of calling `run()`.

## From the CLI

You don't need Python to start a gateway — the [CLI](../cli.md) wraps it:

```bash
neurosurfer serve --host 0.0.0.0 --port 8000
# proxy an upstream backend:
neurosurfer serve --upstream-url http://localhost:1234/v1
```
