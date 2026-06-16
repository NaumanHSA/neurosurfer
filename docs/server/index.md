---
title: Server
description: End-to-end Neurosurfer server layer — OpenAI-compatible FastAPI gateway, deployment recipes, and a worked example application.
---

# Server

Neurosurfer’s **Server** layer is an **OpenAI‑compatible FastAPI gateway** — `/v1/models` and
`/v1/chat/completions` — that you can point any OpenAI client (Open-WebUI, etc.) at. You can
keep the defaults and ship fast, or swap/extend chat handlers, tools, and routes for a fully
custom stack.

> **What you get:** OpenAI‑compatible endpoints, RAG‑ready flows, and streaming/tool‑calling
> support.

## 🚀 Quick Navigation

<div class="grid cards" markdown>

-   :material-cog-outline:{ .lg .middle } **Neurosurfer API (Backend)**

    ---

    OpenAI‑style `/v1/chat/completions`, lifecycle hooks, chat handlers, custom endpoints, auth/users.

    [:octicons-arrow-right-24: Read the Backend guide](./backend/index.md)

-   :material-flask-outline:{ .lg .middle } **Example Application**

    ---

    A minimal but complete reference app: startup/shutdown hooks, a custom chat handler, drop‑in RAG, and a sample tool/endpoint.

    [:octicons-arrow-right-24: Walk through the Example](./example-app.md)

</div>

---

## Usage Overview

Start the server via the **CLI** and use Docker/Compose or a reverse proxy in staging/production. The **backend** is a FastAPI app (exported as `NeurosurferApp`).

### Start the backend (dev)

```bash
neurosurfer serve
```

- Backend binds to `NEUROSURFER_BACKEND_HOST` / `NEUROSURFER_BACKEND_PORT` (from config).

**Common variants**

```bash
neurosurfer serve --backend-host 0.0.0.0 --backend-port 8000

# Serve your own app file (must expose a NeurosurferApp instance)
neurosurfer serve --backend-app ./app.py --backend-reload

# Serve a module with an instance or factory
neurosurfer serve --backend-app mypkg.myapp:ns
neurosurfer serve --backend-app mypkg.myapp:create_app()
```

!!! info "Backend app resolution"
    `--backend-app` accepts a module path with `:attr` or `:factory()`, or a Python file containing a `NeurosurferApp` instance. Don’t call `app.run()` at import time; the CLI orchestrates the process.

---

## Deployment

You can containerize the backend and put a reverse proxy in front of it. Below are pragmatic patterns that work well in practice.

### Docker & Compose

**Dockerfile (backend)**
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . /app

# Install core + LLM extras if you’re serving models locally
RUN pip install -U pip \
 && pip install -e . \
 && pip install 'neurosurfer[torch]'

ENV NEUROSURFER_SILENCE=1 \
    NEUROSURFER_BACKEND_HOST=0.0.0.0 \
    NEUROSURFER_BACKEND_PORT=8000

EXPOSE 8000

# Run your app module or file; replace with your backend entry
CMD ["neurosurfer", "serve", "--backend-app", "neurosurfer.examples.quickstart_app:ns"]
```

**docker-compose.yml (backend + proxy skeleton)**
```yaml
version: "3.9"
services:
  api:
    build: .
    image: neurosurfer-api:latest
    environment:
      NEUROSURFER_SILENCE: "1"
      NEUROSURFER_BACKEND_HOST: "0.0.0.0"
      NEUROSURFER_BACKEND_PORT: "8000"
    ports:
      - "8000:8000"
    restart: unless-stopped

  # Optional: Caddy / Nginx as reverse proxy in front of the gateway
  # proxy:
  #   image: caddy:2
  #   volumes:
  #     - ./Caddyfile:/etc/caddy/Caddyfile:ro
  #   ports:
  #     - "80:80"
  #     - "443:443"
  #   restart: unless-stopped
```

### Reverse Proxy (Nginx/Caddy)

In a reverse proxy, route `/v1/*` to the backend (and `/docs` if you expose FastAPI docs). Ensure CORS settings in the backend match your deployed origins.

- Nginx: `location /v1/ { proxy_pass http://api:8000/v1/; }`
- Caddy: `reverse_proxy /v1/* api:8000`

!!! warning "TLS and WebSockets"
    If you use streaming or WebSockets, confirm your proxy is configured to forward upgrade headers. Both Nginx and Caddy can do this with their standard reverse_proxy settings.

---

## Configuration & Environment

Configuration lives with the backend and is also influenced by CLI/env. For a deeper dive, see **[Backend → Configuration](../api-reference/configuration.md)**.

**Common environment variables**

- `NEUROSURFER_PUBLIC_HOST` — used to craft the public-facing backend URL shown in the ready banner when binding `0.0.0.0`/`::`
- `NEUROSURFER_SILENCE=1` — suppress banner/optional‑deps warnings on import  
- `NEUROSURFER_BACKEND_HOST` / `NEUROSURFER_BACKEND_PORT` — default bind for the API  
- `NEUROSURFER_BACKEND_LOG`, `NEUROSURFER_BACKEND_WORKERS`, `NEUROSURFER_BACKEND_WORKER_TIMEOUT` — backend behavior

!!! tip "Model/runtime dependencies"
    The server core is light. Install the full LLM stack when you need local inference/finetuning:
    ```bash
    pip install -U 'neurosurfer[torch]'
    ```
    For CUDA wheels (Linux x86_64):
    ```bash
    pip install -U torch --index-url https://download.pytorch.org/whl/cu124
    ```
    or CPU‑only:
    ```bash
    pip install -U torch --index-url https://download.pytorch.org/whl/cpu
    ```

---

## Worked Example

The **Example Application** shows: startup hooks (model load, RAG wiring), a custom chat handler, file uploads driving RAG, and a typed server that streams completions.

- Read it here → **[Example Application](./example-app.md)**
- Looking for custom handlers? → **[Chat Handlers](./backend/chat-handlers.md)**
- Want to wire tools or extra routes? → **[Custom Endpoints](./backend/custom-endpoints.md)**

---