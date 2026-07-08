# Configuration

Neurosurfer is configured from the **environment** (optionally via a `.env` file). `load_config()`
reads these into a `Config` dataclass with namespaced sub-configs; most code just relies on the env
being set. Nothing here is required to *import* the package — unset values fall back to sensible
defaults.

```python
from neurosurfer.config import load_config
cfg = load_config()          # reads env (+ .env if present)
print(cfg.redacted())         # safe to log — secrets masked
```

## LLM provider

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `anthropic` or `openai` (the OpenAI-compatible adapter). |
| `MODEL` | provider default | Model id (e.g. `claude-sonnet-4`, `qwen/qwen3.5-9b`). |
| `ANTHROPIC_API_KEY` | — | Required when `LLM_PROVIDER=anthropic`. |
| `OPENAI_BASE_URL` | `http://localhost:1234/v1` | OpenAI-compatible endpoint — **this is how you point at vLLM / Ollama / LM Studio / llama.cpp**. |
| `OPENAI_API_KEY` | `not-needed` | Key for the OpenAI-compatible endpoint (local servers usually ignore it). |
| `CONTEXT_WINDOW` | `200000` | Override the model's context window (local servers should set this). |

See [Providers](../guides/providers.md) for the routing details and provider profiles.

## Runtime & storage

| Variable | Default | Purpose |
|---|---|---|
| `NEUROSURFER_LOG_LEVEL` | `INFO` | Log verbosity. |
| `NEUROSURFER_STATE_DIR` | `./.neurosurfer` | Per-run transcripts / state directory. |
| `NEUROSURFER_HOME` | `~/.neurosurfer` | User config home (profiles, MCP servers). |
| `NEUROSURFER_SERVICE_NAME` | `neurosurfer` | Service name surfaced to trace backends. |

## Observability

Tracing is **auto-on**: set a backend's connection vars and it activates on the next run (no code
change). See [Observability](../observability/index.md).

| Variable | Activates | Purpose |
|---|---|---|
| `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY` | Langfuse exporter | Credentials (both required). |
| `LANGFUSE_HOST` | — | Langfuse endpoint (cloud or self-hosted); read by the Langfuse SDK. |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OpenTelemetry exporter | OTLP endpoint (Honeycomb, Phoenix, Grafana, …). |
| `OTEL_EXPORTER_OTLP_HEADERS` | — | OTLP auth headers (e.g. `x-honeycomb-team=…`); read by the OTel SDK. |
| `NEUROSURFER_EXPORTERS` | override | Comma-separated exporter set (`langfuse,otel`), or `none` to force off. |

## Web search

| Variable | Purpose |
|---|---|
| `SERPAPI_API_KEY` | Enables the SerpAPI search backend for the `search` tool. Without it, the free DuckDuckGo backend is used (needs the `search` extra). |

## Server / gateway (`NS_` prefix)

The [gateway](../server/index.md) reads its own settings with the **`NS_`** prefix:

| Variable | Default | Purpose |
|---|---|---|
| `NS_HOST` / `NS_PORT` | `0.0.0.0` / `8000` | Bind address. |
| `NS_API_KEYS` | `[]` | Bearer tokens required on `/v1/*` (CSV or JSON). |
| `NS_ENABLE_DOCS` | `true` | Serve the `/docs` UI. |
| `NS_CORS_ORIGINS` | `*` | Allowed CORS origins (CSV or JSON). |
| `NS_CORS_ALLOW_CREDENTIALS` | `false` | Allow credentialed CORS. |
| `NS_WORKERS` | `1` | Uvicorn workers. |
| `NS_LOG_LEVEL` | `info` | Uvicorn log level. |

CLI flags to [`neurosurfer serve`](../server/deployment.md) override these.

## Using a `.env` file

Neurosurfer loads a `.env` from the working directory if present. Keep secrets out of source control
and pass `load_config(env_file="path/to/.env")` to load a specific file.
