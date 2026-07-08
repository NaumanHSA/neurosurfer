# Installation

Neurosurfer needs **Python 3.11+**. The core install is light; heavier capabilities are opt-in
**extras** whose imports are lazy — a base install never pulls in torch, chromadb, or a browser.

```bash
pip install -U neurosurfer
```

## Extras

Install only what you use. Combine them in one bracket, e.g. `pip install -U "neurosurfer[search,serve]"`.

| Extra | Adds | Pulls in |
|---|---|---|
| `search` | Free web search (`search` tool, no API key) | `ddgs`, `beautifulsoup4`, `lxml`, `rank-bm25` |
| `browser` | Headless-browser `browse` tool | `playwright` (then run `playwright install chromium`) |
| `rag` | Ingest/retrieve, vector store, file readers | `chromadb`, `sentence-transformers`, `SQLAlchemy`, `PyMuPDF`, `python-docx`, `python-pptx` |
| `serve` | OpenAI-compatible gateway + `neurosurfer serve` | `fastapi`, `uvicorn` |
| `mcp` | Model Context Protocol client | `mcp` |
| `observability` | Langfuse + OpenTelemetry trace exporters | `langfuse`, `opentelemetry-sdk`, `opentelemetry-exporter-otlp` |
| `local` | Accurate token counting for local models | `tiktoken` |
| `dev` | Test/lint/build toolchain | `pytest`, `ruff`, `mypy`, `build`, `twine` |

!!! tip "Pick extras by task, not by fear of missing out"
    A chat agent against a hosted model needs **nothing** beyond the core. Add `search` for web
    tools, `serve` to expose an OpenAI API, `rag` for retrieval, `observability` for tracing.

## Verify the install

```bash
neurosurfer doctor      # checks config + provider reachability (and MCP, if configured)
```

`doctor` reports the resolved provider, model, and whether the configured endpoint answers — the
fastest way to catch a missing key or a wrong base URL before your first run.

## Next

- [Quickstart](quickstart.md) — your first agent, the REPL, and the gateway in a few minutes.
- [Configuration](../reference/configuration.md) — every environment variable Neurosurfer reads.
