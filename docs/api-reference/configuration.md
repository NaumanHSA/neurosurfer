---
title: Configuration
description: One `config` to rule them all — environment‑first settings for app/server, base model defaults, external DB, and RAG chunking.
---

# Configuration

Neurosurfer exposes a single import, **`config`**, that aggregates settings from environment variables (and `.env`) with safe defaults. Use these values directly, or pass them into constructors via `to_dict()` where available.

---

## App settings

Application/server behavior (host/port, CORS, logging) and derived paths/URLs.

| Setting | Env var | Default | Description |
|---|---|---|---|
| `app_name` | `APP_APP_NAME` | `Neurosurfer` | Display name for logs/metadata. |
| `dev_version` | `APP_DEV_VERSION` | `1.0.0` | Version shown in non‑prod. |
| `prod_version` | `APP_PROD_VERSION` | `1.0.0` | Version shown in prod. |
| `description` | `APP_DESCRIPTION` | `Neurosurfer` | Human‑readable description. |
| `host_ip` | `APP_HOST_IP` | `0.0.0.0` | Bind IP for the server. |
| `host_port` | `APP_HOST_PORT` | `8081` | Bind port for the server. |
| `host_protocol` | `APP_HOST_PROTOCOL` | `http` | Protocol used in `host_url`. |
| `logs_level` | `APP_LOGS_LEVEL` | `info` | Global logging level. |
| `cors_origins` | `APP_CORS_ORIGINS` | `["*"]` | Allowed browser origins. |
| `reload` | `APP_RELOAD` | `false` | Auto‑reload in dev. |
| `workers` | `APP_WORKERS` | `1` | Uvicorn/Gunicorn workers. |
| `enable_docs` | `APP_ENABLE_DOCS` | `true` | Enable `/docs` UI. |
| `temp_path` | `APP_TEMP_PATH` | `temp` | Scratch dir for temp files. |
| `logs_path` | `APP_LOGS_PATH` | `logs` | Log file directory. |
| `database_path` | `APP_DATABASE_PATH` | `./db_storage` | Root for SQLite & vectors. |
| `is_docker` | `APP_IS_DOCKER` | `false` | Hint to adjust runtime behavior. |
| `host_url` (derived) | — | `http://0.0.0.0:8081` | Computed from protocol/ip/port. |
| `vector_store_path` (derived) | — | `./db_storage/codemind_chroma` | Vector DB folder. |
| `database_url` (derived) | — | — | SQLite DSN used by server. |

!!! tip
    For LAN deployments, `config.app.get_dynamic_host_ip()` tries to pick a non‑loopback interface IP automatically.

---

## Base model defaults

Shared generation parameters for all chat model backends. Use `config.base_model.to_dict()` to unpack into model initializers.

| Setting | Env var | Default | Description |
|---|---|---|---|
| `temperature` | `TEMPERATURE` | `0.7` | Sampling temperature (0‑2). |
| `max_seq_length` | `MAX_SEQ_LENGTH` | `4096` | Model context length. |
| `max_new_tokens` | `MAX_NEW_TOKENS` | `2000` | Max generation tokens. |
| `top_k` | `TOP_K` | `4` | Top‑k sampling. |
| `load_in_4bit` | `LOAD_IN_4BIT` | `false` | Enable 4‑bit quant. |
| `enable_thinking` | `ENABLE_THINKING` | `false` | Enable “thinking mode”. |
| `stop_words` | `STOP_WORDS` | `null` | Hard stop sequences. |
| `system_prompt` | `SYSTEM_PROMPT` | `You are a helpful assistant...` | Default system prompt. |
| `verbose` | `VERBOSE` | `false` | Verbose model logging. |

---

## External database config

Credentials for tooling/agents that talk to SQL Server (etc.).

| Setting | Env var | Default | Description |
|---|---|---|---|
| `server` | `DB_SERVER` | `localhost` | Hostname or IP. |
| `database` | `DB_DATABASE` | `""` | Database/schema. |
| `username` | `DB_USERNAME` | `""` | Login username. |
| `password` | `DB_PASSWORD` | `""` | Login password. |
| `driver` | `DB_DRIVER` | `ODBC Driver 17 for SQL Server` | ODBC driver name. |
| `port` | `DB_PORT` | `1433` | TCP port. |

!!! tip
    Keep secrets in your platform’s secret manager (or a private `.env` in development). Never commit secrets.

---

## Chunker (RAG) config

Chunking rules for documents and code, used by the RAG pipeline.

| Setting | Env var | Default | Description |
|---|---|---|---|
| `fallback_chunk_size` | `CHUNKER_FALLBACK_CHUNK_SIZE` | `25` | Lines per chunk (line mode). |
| `overlap_lines` | `CHUNKER_OVERLAP_LINES` | `3` | Line overlap between chunks. |
| `max_chunk_lines` | `CHUNKER_MAX_CHUNK_LINES` | `1000` | Safety cap for line chunks. |
| `comment_block_threshold` | `CHUNKER_COMMENT_BLOCK_THRESHOLD` | `4` | Treat N comment lines as a block. |
| `char_chunk_size` | `CHUNKER_CHAR_CHUNK_SIZE` | `1000` | Chars per chunk (char mode). |
| `char_overlap` | `CHUNKER_CHAR_OVERLAP` | `150` | Char overlap between chunks. |
| `readme_max_lines` | `CHUNKER_README_MAX_LINES` | `30` | Lines per README/Markdown chunk. |
| `json_chunk_size` | `CHUNKER_JSON_CHUNK_SIZE` | `1000` | Chars per JSON chunk. |
| `fallback_mode` | `CHUNKER_FALLBACK_MODE` | `char` | `char`/`line`/`auto` strategy. |
| `max_returned_chunks` | `CHUNKER_MAX_RETURNED_CHUNKS` | `500` | Post‑sanitize chunk cap. |
| `max_total_output_chars` | `CHUNKER_MAX_TOTAL_OUTPUT_CHARS` | `1000000` | Post‑sanitize char cap. |
| `min_chunk_non_ws_chars` | `CHUNKER_MIN_CHUNK_NON_WS_CHARS` | `1` | Drop nearly empty chunks. |

!!! note
    `fallback_mode="auto"` is a good default for mixed code/docs repositories.

---

## Using `config` in code

Import once and pass settings to the pieces that need them.

```python
from neurosurfer.config import config
from neurosurfer.models.chat_models.transformers import TransformersModel

logger = config.get_logger("neurosurfer")

# Feed base model defaults straight into the backend
llm = TransformersModel(
    **{**config.base_model.to_dict(), "model_name": "/models/Qwen3-4B"},
    logger=logger,
)

print(config.app.host_url)          # server URL
print(config.app.database_url)      # SQLite DSN
print(config.app.vector_store_path) # vector store path
```

### Overriding specific values

**Option A — environment variables** (preferred):
```dotenv
# .env
APP_HOST_PORT=8080
APP_CORS_ORIGINS=["https://studio.example.com"]
TEMPERATURE=0.6
CHUNKER_FALLBACK_MODE=auto
```

**Option B — in code** (rare; useful for quick experiments):
```python
from neurosurfer.config import config

config.app.host_port = 8080
config.base_model.temperature = 0.6
config.chunker.fallback_mode = "auto"
```

!!! tip
    The `config` initializer ensures required directories exist (`temp`, `logs`, `db_storage`, and the vector store path), so clean installs work out of the box.

---
