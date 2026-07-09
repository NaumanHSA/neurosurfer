# Deployment

Run the gateway in production — from a one-liner CLI launch to a containerised stack with tracing.

## The `serve` command

With the `serve` extra installed, `neurosurfer serve` starts the gateway without writing any Python:

```bash
pip install -U "neurosurfer[serve]"
neurosurfer serve --host 0.0.0.0 --port 8000
```

| Flag | Default | Purpose |
|---|---|---|
| `--host` | `0.0.0.0` | Bind host. |
| `--port` | `8000` | Bind port. |
| `--workers` | `1` | Uvicorn worker processes. |
| `--reload` | off | Auto-reload for development. |
| `--log-level` | `info` | Uvicorn log level. |
| `--no-docs` | docs on | Disable the `/docs` UI. |
| `--upstream-url` | — | Proxy to an upstream OpenAI-compatible endpoint. |
| `--upstream-api-key` | — | API key for that upstream. |

Server settings also read from the environment with the **`NS_`** prefix — `NS_PORT`, `NS_API_KEYS`,
`NS_CORS_ORIGINS`, `NS_ENABLE_DOCS`, etc. (see [Configuration](../guides/configuration.md)). CLI
flags override env.

## Programmatic app (ASGI)

For a custom process manager, build the FastAPI app yourself and hand it to any ASGI server:

```python
# app.py
from neurosurfer.app.server import NeurosurferServer
server = NeurosurferServer(api_keys=["sk-local-123"])
server.register_agent(build_agent(), model_id="my-agent")
app = server.create_app()      # a FastAPI instance
```

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

## Containerized deployment

Neurosurfer doesn't ship a Docker image — the gateway is a plain pip-installed service, so a minimal
image is all you need:

```dockerfile
FROM python:3.12-slim
RUN pip install --no-cache-dir "neurosurfer[serve]"
EXPOSE 8000
CMD ["neurosurfer", "serve", "--host", "0.0.0.0", "--port", "8000"]
```

To run it alongside a self-hosted [Langfuse](../observability/index.md) so traces stay local, compose
the gateway with the Langfuse services and point the tracing env at the internal Langfuse host:

```yaml
services:
  gateway:
    build: .          # the Dockerfile above
    environment:
      - NEUROSURFER_EXPORTERS=langfuse
      - LANGFUSE_HOST=http://langfuse-web:3000
      - LANGFUSE_PUBLIC_KEY=pk-lf-...
      - LANGFUSE_SECRET_KEY=sk-lf-...
    ports: ["8000:8000"]
    depends_on: { langfuse-web: { condition: service_healthy } }
  # langfuse-web / worker / postgres / clickhouse / redis / minio …
```

The gateway reaches Langfuse over the internal Docker network (`langfuse-web:3000`), so tracing works
without exposing Langfuse publicly. See [Observability](../observability/index.md) for the full
tracing setup, and the [Configuration reference](../guides/configuration.md) for every env var.

## Production checklist

- **Auth** — set `NS_API_KEYS` (or `api_keys=`) so `/v1/*` requires a bearer token.
- **CORS** — restrict `NS_CORS_ORIGINS` to your front-ends instead of `*`.
- **Headless IO** — served agents must use an auto-approving handler + tight guardrails
  ([Serving Agents](agents.md)).
- **Docs** — consider `--no-docs` in prod if you don't want the interactive `/docs` exposed.
- **Workers** — scale with `--workers`; keep agent state per-request, not global.
