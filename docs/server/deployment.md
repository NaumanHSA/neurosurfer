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
`NS_CORS_ORIGINS`, `NS_ENABLE_DOCS`, etc. (see [Configuration](../reference/configuration.md)). CLI
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

## Docker + observability stack

To run the gateway alongside a self-hosted [Langfuse](../observability/index.md) so traces stay
local, compose the gateway with the Langfuse services and point the tracing env at the internal
Langfuse host:

```yaml
services:
  gateway:
    image: your-neurosurfer-image
    command: ["neurosurfer", "serve", "--host", "0.0.0.0", "--port", "8000"]
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
tracing setup, and the [Configuration reference](../reference/configuration.md) for every env var.

!!! warning "The bundled `docker-compose.yml` references a launcher that isn't shipped"
    The repository's `docker-compose.yml` sets
    `NEUROSURFER_BACKEND_APP=neurosurfer.examples.quickstart_app:ns`, but no `neurosurfer.examples`
    module ships in the package. Use `neurosurfer serve` or the programmatic ASGI app above as your
    entrypoint, or provide your own module that builds a `NeurosurferServer` and registers your
    agents.

## Production checklist

- **Auth** — set `NS_API_KEYS` (or `api_keys=`) so `/v1/*` requires a bearer token.
- **CORS** — restrict `NS_CORS_ORIGINS` to your front-ends instead of `*`.
- **Headless IO** — served agents must use an auto-approving handler + tight guardrails
  ([Serving Agents](agents.md)).
- **Docs** — consider `--no-docs` in prod if you don't want the interactive `/docs` exposed.
- **Workers** — scale with `--workers`; keep agent state per-request, not global.
