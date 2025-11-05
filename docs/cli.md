# CLI & Dev Server Guide (Enhanced)

The **Neurosurfer CLI** is a single command that boots your developer stack: it starts the FastAPI backend, optionally brings up the NeurowebUI. Use it for day-to-day development, local demos, and quick sanity checks of your app and UI.

The CLI command is `neurosurfer`. Its primary subcommand is `serve`, which starts the backend API and, if desired, the NeurowebUI dev server. The **backend** binds to host/port from your configuration or flags. The **UI** is auto‑discovered (or provided via `--ui-root`) and runs either as a **Vite dev server** or a **static build server** (if you point at a build dir).

Use `--help` to see available flags:
```bash
neurosurfer --help
neurosurfer serve --help
```

### What happens if nothing is specified?

If you run `neurosurfer serve` with no flags, the CLI boots **the built-in example backend** (a ready-to-use NeurosurferApp) on your configured host/port and then looks for a **bundled UI build**; if it finds one, it serves it automatically, otherwise it runs **backend-only**. For details, see: [Built-in Example App](./examples/server-app-example.md) and [Bundled UI Build](./server/neurosurferui.md).


---

## 🚀 Quick Navigation

<div class="grid cards" markdown>

-   :material-console:{ .lg .middle } **Overview**

    ---

    What the CLI does, command structure, and where it runs.

    [:octicons-arrow-right-24: Read Overview](#overview)

-   :material-cube-outline:{ .lg .middle } **App Resolution**

    ---

    How files, modules, and `NeurosurferApp` instances are discovered.

    [:octicons-arrow-right-24: Learn Resolution](#backend-app-resolution)

-   :material-view-dashboard:{ .lg .middle } **NeurosurferUI**

    ---

    How `--ui-root` is detected, when installs run, Vite dev server args.

    [:octicons-arrow-right-24: UI Details](#neurosurferui)

-   :material-rocket-launch:{ .lg .middle } **Recipes & Examples**

    ---

    Common scenarios—backend only, custom modules, public host URLs, and more.

    [:octicons-arrow-right-24: Try Recipes](#recipes--examples)

-   :material-shield-lock:{ .lg .middle } **Security & Ops**

    ---

    CORS, reverse proxy, environment variables, graceful shutdown.

    [:octicons-arrow-right-24: Ops Notes](#security--operations)

</div>

---

## Key Options

Below is a quick reference of the most useful flags. Full reference tables are in **[Environment & Flags](#environment--flags-reference)**.

### Backend flags (essentials)

- **`--backend-app`**: points to the backend app (see *App Resolution* below)
- **`--backend-host`**, **`--backend-port`**
- **`--backend-log-level`**
- **`--backend-reload`**
- **`--backend-workers`**
- **`--backend-worker-timeout`**

### UI flags (essentials)

- **`--ui-root`**: path to NeurowebUI (folder with `package.json`) or a prebuilt static dir with `index.html`
- **`--ui-host`**, **`--ui-port`**
- **`--ui-strict-port`**
- **`--ui-open`** *(default: **true**; pass `--ui-open=0` or `--ui-open=false` to disable)*
- **`--npm-install {auto|always|never}`** *(default: `auto`)*
- **`--only-backend`** / **`--only-ui`**

> **What’s new?** The CLI now **opens the browser only after the UI is reachable**, so your “open UI” message isn’t buried under backend logs.

---

## Backend App Resolution

When you provide `--backend-app`, the CLI can interpret three forms and resolve them into a runnable [NeurosurferApp](./server/example-app.md) instance.

### 1) Default (no `--backend-app`)

If omitted, the CLI runs the built‑in example shipped with the package:
```
neurosurfer.examples.quickstart_app:ns
```

### 2) Module path (`pkg.module[:attr_or_factory]`)

Point to a module and an attribute/factory yielding a `NeurosurferApp`:

```bash
neurosurfer serve --backend-app mypkg.api:ns
neurosurfer serve --backend-app mypkg.api:create_app()
```

### 3) File path (`/path/to/app.py`)

Pass a Python file that defines a `NeurosurferApp` instance; the CLI executes the file in an isolated namespace and scans for the instance:

```bash
neurosurfer serve --backend-app ./app.py --backend-reload
```

!!! warning "File mode expectations"
    Ensure your file **creates** a `NeurosurferApp` instance but does **not** call `app.run()` at import time. The CLI will run it. Keep direct runs behind `if __name__ == '__main__':`.

---

## NeurosurferUI

The NeurosurferUI is a Vite‑powered dev server for development. The CLI can also serve a **built** UI directory statically.

### Root discovery

If `--ui-root` is omitted, we try common relative paths (e.g., `neurosurferui`), or read `NEUROSURF_UI_ROOT`. If not found, the CLI runs **backend only** with a helpful log.

```bash
neurosurfer serve --ui-root /path/to/neurowebui
```

### First‑run dependency install

With `--npm-install auto` (default), the CLI runs `npm install --force` **only if** `node_modules` is missing. Use `always` to force install, or `never` to skip.

### Backend URL for the UI

If the backend binds to `0.0.0.0` or `::`, browsers can’t use that literal host. The CLI injects **`VITE_BACKEND_URL`** using `NEUROSURF_PUBLIC_HOST` (or `127.0.0.1` by default). Set it explicitly if you’re exposing the backend on a LAN or public IP:

```bash
export NEUROSURF_PUBLIC_HOST=192.168.1.25
neurosurfer serve
```

!!! tip "Cross‑origin requests"
    Use permissive CORS in local dev; in production, restrict to your domains. See **Configuration** for knobs.

---

## Recipes & Examples

### Backend only

```bash
neurosurfer serve --only-backend --backend-host 0.0.0.0 --backend-port 8081
```

### UI only (Vite dev)

```bash
neurosurfer serve --only-ui --ui-root /path/to/neurowebui
```

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

### Use a public backend URL for the UI

```bash
export NEUROSURF_PUBLIC_HOST=192.168.1.25
neurosurfer serve
```

### Force UI dependency install

```bash
neurosurfer serve --npm-install always
```

### Strict UI port usage

```bash
neurosurfer serve --ui-strict-port --ui-port 5173
```

---

## Security & Operations

### CORS & Auth

Control CORS via config. For private deployments, pair CORS with a reverse proxy (Nginx, Traefik) and your preferred auth. See **Configuration** for the complete set.

### Reverse Proxy & TLS

Terminate TLS at a reverse proxy and forward to the backend. Keep ports non‑public where possible. If you must expose the backend directly, enforce strong auth and restrict IP ranges.

### Logs & Levels

Use `--backend-log-level`. During local debugging, `debug` can be handy; `info` is the sane default. UI and backend child logs are piped with `[ui]` and `[api]` prefixes. A **single banner** summarizes URLs after readiness checks.

### Graceful Shutdown

We register `SIGINT`/`SIGTERM` handlers. On shutdown, both UI and backend are terminated with a short grace period before a hard kill to avoid lingering watchers/sockets.

!!! tip "Non‑interactive runs (CI)"
    Set `NEUROSURF_SILENCE=1` to suppress banners in CI, and pin ports carefully to avoid collisions.

---

## Environment & Flags Reference

### Backend (flags & env)

| Setting | Flag | Env Var | Default | Description |
|---|---|---|---|---|
| Backend app | `--backend-app` | – | `neurosurfer.examples.quickstart_app:ns` | Module attr/factory or file path that yields a `NeurosurferApp`. |
| Host | `--backend-host` | `NEUROSURF_BACKEND_HOST` | from config | Bind address for API. |
| Port | `--backend-port` | `NEUROSURF_BACKEND_PORT` | from config | Bind port for API. |
| Log level | `--backend-log-level` | `NEUROSURF_BACKEND_LOG` | from config | Logging verbosity (`debug`, `info`, etc.). |
| Reload | `--backend-reload` | – | `false` | Auto‑reload for dev. |
| Workers | `--backend-workers` | `NEUROSURF_BACKEND_WORKERS` | from config | Number of worker processes. |
| Worker timeout (s) | `--backend-worker-timeout` | `NEUROSURF_BACKEND_WORKER_TIMEOUT` | from config | Worker timeout in seconds. |

### UI (flags & env)

| Setting | Flag | Env Var | Default | Description |
|---|---|---|---|---|
| UI root | `--ui-root` | `NEUROSURF_UI_ROOT` | auto‑detect | Path to Vite project (with `package.json`) or a build dir (`index.html`). |
| UI host | `--ui-host` | `NEUROSURF_UI_HOST` | from config | Bind host for Vite/static server. |
| UI port | `--ui-port` | `NEUROSURF_UI_PORT` | from config | Bind port for Vite/static server. |
| Strict port | `--ui-strict-port` | – | `false` | Fail if port is in use (Vite). |
| Open UI in browser | `--ui-open` | `NEUROSURF_UI_OPEN` | **`true`** | Auto‑open browser when UI becomes reachable. Accepts `1/0`, `true/false`. |
| NPM install policy | `--npm-install` | `NEUROSURF_NPM_INSTALL` | `auto` | `auto`: only if missing `node_modules`; `always`: force; `never`: skip. |
| Only backend | `--only-backend` | – | `false` | Run API only. |
| Only UI | `--only-ui` | – | `false` | Run UI only. |

### Cross‑cutting

| Setting | Env Var | Default | Description |
|---|---|---|---|
| Public host for URL composition | `NEUROSURF_PUBLIC_HOST` | `127.0.0.1` | Used to craft `VITE_BACKEND_URL` when API binds `0.0.0.0`/`::`. |
| Silence banners & optional‑deps warnings | `NEUROSURF_SILENCE` | `0` | Set `1` to reduce noise (e.g., CI). |
| Eager runtime assert (deps) | `NEUROSURF_EAGER_RUNTIME_ASSERT` | `0` | Set `1` to fail fast on missing optional LLM deps. |

> **Note on `--ui-open` default:** The modular implementation treats UI auto‑open as **enabled by default**. You can disable it with `--ui-open=false`, `--ui-open=0`, or by setting `NEUROSURF_UI_OPEN=0`.

---

## Troubleshooting

- **Port in use**: Use `--ui-strict-port` or pick a free `--ui-port`. For the API, change `--backend-port`.
- **UI can’t reach API**: Ensure `VITE_BACKEND_URL` resolves. Set `NEUROSURF_PUBLIC_HOST` when binding `0.0.0.0`.
- **`npm` not found**: Install Node.js/npm or run `--only-backend`.
- **Long password error on register (bcrypt 72‑byte limit)**: Use a shorter password or configure truncation in your auth layer.
- **Static UI**: If you point `--ui-root` at a build folder (must contain `index.html`), the CLI will serve it via a lightweight static server.

---

## Ready to launch?

```bash
# Backend + UI (auto-open, default)
neurosurfer serve --backend-host 0.0.0.0 --backend-port 8081 --ui-root ./neurosurferui

# Disable auto-open
neurosurfer serve --ui-open=0

# Backend only
neurosurfer serve --only-backend --backend-port 8081
```
