# CLI & Dev Server Guide

The **Neurosurfer CLI** boots your backend gateway with a single command. Use it for
day-to-day development, local demos, and quick sanity checks of your app.

The CLI command is `neurosurfer`. Its primary subcommand is `serve`, which starts the
backend API. The backend binds to host/port from your configuration or flags.

Use `--help` to see available flags:
```bash
neurosurfer --help
neurosurfer serve --help
```

!!! info "Dependencies"
    Before running the CLI, make sure you have environment ready with dependencies installed. For the default UI, cli requires `npm`, `nodejs` and `serve` to be installed on your system.

### What happens if nothing is specified?

If you run `neurosurfer serve` with no flags, the CLI boots **the built-in example
backend** (a ready-to-use `NeurosurferServer` gateway) on your configured host/port. For
details, see: [Built-in Example App](./examples/server-app-example.md).

---

## Key Options

- **`--backend-app`**: points to the backend app (see *App Resolution* below)
- **`--backend-host`**, **`--backend-port`**
- **`--backend-log-level`**
- **`--backend-reload`**
- **`--backend-workers`**
- **`--backend-worker-timeout`**

---

## Backend App Resolution

When you provide `--backend-app`, the CLI can interpret three forms and resolve them into
a runnable [NeurosurferServer](./server/example-app.md) instance.

### 1) Default (no `--backend-app`)

If omitted, the CLI runs the built‑in example shipped with the package:
```
neurosurfer.examples.quickstart_app:ns
```

### 2) Module path (`pkg.module[:attr_or_factory]`)

Point to a module and an attribute/factory yielding a `NeurosurferServer`:

```bash
neurosurfer serve --backend-app mypkg.api:ns
neurosurfer serve --backend-app mypkg.api:create_app()
```

### 3) File path (`/path/to/app.py`)

Pass a Python file that defines a `NeurosurferServer` instance; the CLI executes the file
in an isolated namespace and scans for the instance:

```bash
neurosurfer serve --backend-app ./app.py --backend-reload
```

!!! warning "File mode expectations"
    Ensure your file **creates** a `NeurosurferServer` instance but does **not** call
    `app.run()` at import time. The CLI will run it. Keep direct runs behind
    `if __name__ == '__main__':`.

---

## Recipes & Examples

### Serve a module attribute

```bash
neurosurfer serve --backend-app mypkg.myapp:ns --backend-reload
```

### Serve a factory

```bash
neurosurfer serve --backend-app mypkg.myapp:create_app()
```

### Serve a local file

```bash
neurosurfer serve --backend-app ./app.py --backend-reload
```

---

## Security & Operations

### CORS & Auth

Control CORS via config. For private deployments, pair CORS with a reverse proxy (Nginx,
Traefik) and your preferred auth. See **Configuration** for the complete set.

### Reverse Proxy & TLS

Terminate TLS at a reverse proxy and forward to the backend. Keep ports non‑public where
possible. If you must expose the backend directly, enforce strong auth and restrict IP
ranges.

### Logs & Levels

Use `--backend-log-level`. During local debugging, `debug` can be handy; `info` is the
sane default. Backend child logs are piped with an `[api]` prefix. A **single banner**
summarizes the URL after readiness checks.

### Graceful Shutdown

We register `SIGINT`/`SIGTERM` handlers. On shutdown, the backend is terminated with a
short grace period before a hard kill to avoid lingering sockets.

!!! tip "Non‑interactive runs (CI)"
    Set `NEUROSURFER_SILENCE=1` to suppress banners in CI, and pin ports carefully to
    avoid collisions.

---

## Environment & Flags Reference

| Setting | Flag | Env Var | Default | Description |
|---|---|---|---|---|
| Backend app | `--backend-app` | `NEUROSURFER_BACKEND_APP` | `neurosurfer.examples.quickstart_app:ns` | Module attr/factory or file path that yields a `NeurosurferServer`. |
| Host | `--backend-host` | `NEUROSURFER_BACKEND_HOST` | from config | Bind address for API. |
| Port | `--backend-port` | `NEUROSURFER_BACKEND_PORT` | from config | Bind port for API. |
| Log level | `--backend-log-level` | `NEUROSURFER_BACKEND_LOG` | from config | Logging verbosity (`debug`, `info`, etc.). |
| Reload | `--backend-reload` | – | `false` | Auto‑reload for dev. |
| Workers | `--backend-workers` | `NEUROSURFER_BACKEND_WORKERS` | from config | Number of worker processes. |
| Worker timeout (s) | `--backend-worker-timeout` | `NEUROSURFER_BACKEND_WORKER_TIMEOUT` | from config | Worker timeout in seconds. |

### Cross‑cutting

| Setting | Env Var | Default | Description |
|---|---|---|---|
| Silence banners & optional‑deps warnings | `NEUROSURFER_SILENCE` | `0` | Set `1` to reduce noise (e.g., CI). |
| Eager runtime assert (deps) | `NEUROSURFER_EAGER_RUNTIME_ASSERT` | `0` | Set `1` to fail fast on missing optional LLM deps. |

---

## Ready to launch?

```bash
neurosurfer serve --backend-host 0.0.0.0 --backend-port 8081
```
