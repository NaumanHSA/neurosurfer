# CLI & Dev Server Guide

The **Neurosurfer CLI** is your single entry point for launching the backend API and (optionally) the NeurowebUI dev server. This page explains how the CLI resolves your backend app, how the UI is discovered and started, and how to tune behavior with flags and environment variables.

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

    How `--ui-root` is detected, when `npm install` runs, Vite dev server args.

    [:octicons-arrow-right-24: UI Details](#neurosurferui)

-   :material-rocket-launch:{ .lg .middle } **Recipes & Examples**

    ---

    Common scenarios—backend only, custom modules, public host URLs, and more.

    [:octicons-arrow-right-24: Try Recipes](#recipes-examples)

-   :material-shield-lock:{ .lg .middle } **Security & Ops**

    ---

    CORS, reverse proxy, environment variables, graceful shutdown.

    [:octicons-arrow-right-24: Ops Notes](#security--operations)

</div>

---

## Overview

The CLI command is `neurosurfer`. Its most important subcommand is `serve`, which starts the backend API and optionally the NeurowebUI Vite dev server. By default, the backend will bind to host/port values from your configuration; the UI will be auto‑detected and started if present (you can opt out via flags).

Use `--help` to see available flags:
```bash
neurosurfer --help
neurosurfer serve --help
```

!!! tip "Lightweight by default"
    The CLI itself does not require the full LLM stack. You can run API scaffolding and the UI even without GPU/model packages installed. When you actually load models, the runtime will guide you if extras are missing.

---

## Key Options

Most users start with `neurosurfer serve` and refine behavior by setting host/port, turning on reload, or pointing at a specific app file/module. The following options are the essentials you’ll use day‑to‑day.

### Backend flags

- **`--backend-app`**: where to find your backend application. This accepts:
  - *(default)* `neurosurfer.examples.quickstart_app:ns`
  - A **module path** with an attribute or factory: `pkg.module:ns` or `pkg.module:create_app()`
  - A **file path** to a Python script that defines a `NeurosurferApp` instance
- **`--backend-host`**, **`--backend-port`**: where to bind the API service (defaults come from your config).
- **`--backend-log-level`**: logging level (e.g., `info`, `debug`).
- **`--backend-reload`**: enable auto‑reload for development.
- **`--backend-workers`**: number of workers for serving requests.
- **`--backend-worker-timeout`**: worker timeout seconds.

### UI flags

- **`--ui-root`**: path to the NeurowebUI project (folder with `package.json`). If omitted, the CLI tries common locations or reads `NEUROSURF_UI_ROOT`.
- **`--ui-host`**, **`--ui-port`**: where to bind the Vite dev server.
- **`--ui-strict-port`**: fail fast if the port is already in use.
- **`--ui-open`**: open the browser when the UI starts.
- **`--npm-install {auto|always|never}`**: control first‑run dependency install.
- **`--only-backend`** / **`--only-ui`**: run a single side.

!!! info "UI dependency management"
    With `--npm-install auto` (default), the CLI runs `npm install` only when it detects a missing `node_modules` directory. Use `always` to force installation, or `never` if you manage dependencies yourself.

---

## Backend App Resolution

When you provide `--backend-app`, the CLI can interpret three forms and resolve them into a runnable `NeurosurferApp` instance. The goal is developer convenience while keeping behavior explicit.

### 1) Default (no `--backend-app`)

If you omit the flag, the CLI runs the built‑in example shipped with the package:
```
neurosurfer.examples.quickstart_app:ns
```
The instance `ns` is a preconfigured `NeurosurferApp` designed for quick testing.

### 2) Module path (`pkg.module[:attr_or_factory]`)

You can point to a module and the attribute/factory that yields your app instance. For example:
```bash
neurosurfer serve --backend-app mypkg.myapi:ns
neurosurfer serve --backend-app mypkg.myapi:create_app()
```
The CLI imports your module, resolves the attribute or calls the factory, and runs the resulting `NeurosurferApp`.

### 3) File path (`/path/to/app.py`)

You can also pass a Python file that defines a `NeurosurferApp` instance. The CLI executes the file in an isolated namespace and scans for an instance of `NeurosurferApp`. If it finds one, it runs it.

```bash
neurosurfer serve --backend-app ./app.py --backend-reload
```

!!! warning "File mode expectations"
    Ensure your file **creates** a `NeurosurferApp` instance but does **not** call `app.run()` at import time. The CLI will handle running. Prefer `if __name__ == '__main__': ns.run()` in files intended for direct execution.

---

## NeurosurferUI

The NeurosurferUI is a Vite‑powered dev server used during development. The CLI can launch it in parallel with the backend.

### Root discovery

If `--ui-root` is not specified, the CLI tries common relative paths (e.g., `neurosurferui` next to your package) or reads `NEUROSURF_UI_ROOT`. If the folder is not found, the CLI will exit with a helpful message.

```bash
neurosurfer serve --ui-root /path/to/neurowebui
```

### First‑run install

The first time you run the UI, dependencies may be missing. With `--npm-install auto`, the CLI runs `npm install --force` if it detects a missing `node_modules` directory. You can override this with `--npm-install always` or `--npm-install never`.

### Backend URL for the UI

The UI needs to talk to the backend. If the backend binds to `0.0.0.0` or `::`, browsers can’t use that literal host; so the CLI injects **`VITE_BACKEND_URL`** using `NEUROSURF_PUBLIC_HOST` (or `127.0.0.1` by default). Set it explicitly if you’re exposing the backend on a LAN or public IP:

```bash
export NEUROSURF_PUBLIC_HOST=your.ip.addr
neurosurfer serve
```

!!! tip "Cross‑origin requests"
    The backend exposes CORS settings via config. When developing locally, keep origins permissive; for production, restrict to your actual UI domain(s). See **[Configuration](./api-reference/configuration.md)** for details.

---

## Recipes & Examples

Use these scenarios as a starting point and adapt to your environment (ports, hosts, etc.).

### Backend only

```bash
neurosurfer serve --only-backend --backend-host 0.0.0.0 --backend-port 8000
```

### UI only

```bash
neurosurfer serve --only-ui --ui-root /path/to/neurosurferui
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

Good defaults help in development, but production requires a bit more care—especially around ports, proxies, and graceful shutdown.

### CORS & Authentication

The backend supports CORS via configuration, allowing you to control which origins can access your API. For private deployments, pair CORS with a reverse proxy (e.g., Nginx, Traefik) and standard authentication. See **[Configuration](./api-reference/configuration.md)** for the complete set of knobs.

### Reverse Proxy & TLS

In production, terminate TLS at a reverse proxy and forward to the backend. Keep ports non‑public where possible. If you must expose the backend directly, prefer strong auth and limited IP ranges.

### Logs & Levels

Use `--backend-log-level` to switch verbosity. During local debugging, `debug` can be handy; otherwise `info` is usually sufficient. UI logs stream to the console with `[ui]` and backend logs as `[api]` prefixes in the CLI runner.

### Graceful Shutdown

The CLI registers handlers for `SIGINT` and `SIGTERM`. On shutdown, both UI and backend child processes are asked to terminate, with a short grace period before a hard kill. This avoids lingering watchers or socket binds in dev.

!!! tip "Non‑interactive runs (CI)"
    In CI or background jobs, set `NEUROSURF_SILENCE=1` to suppress banners/warnings, and pin ports carefully to avoid conflicts with runners.

---

## Environment Reference

You can run everything via flags, or set environment variables to define defaults (useful for containers/CI).

- `NEUROSURF_UI_ROOT` — path to the NeurowebUI root (folder with `package.json`)
- `NEUROSURF_PUBLIC_HOST` — concrete host/IP used to craft `VITE_BACKEND_URL` for the UI when backend binds `0.0.0.0`/`::`
- `NEUROSURF_SILENCE=1` — suppress banner and optional‑deps warnings on import
- `NEUROSURF_EAGER_RUNTIME_ASSERT=1` — fail fast at import if LLM deps are missing

Backend defaults (often mirrored by flags within your config layer):
- `NEUROSURF_BACKEND_HOST`, `NEUROSURF_BACKEND_PORT`
- `NEUROSURF_BACKEND_LOG`, `NEUROSURF_BACKEND_WORKERS`, `NEUROSURF_BACKEND_WORKER_TIMEOUT`
- `NEUROSURF_UI_HOST`, `NEUROSURF_UI_PORT`

---

## Troubleshooting

If something doesn’t start, read the error message—the CLI prints a single consolidated message when it can’t find a UI root or backend app. For environment mismatches, these quick checks help:

- **Port in use**: use `--ui-strict-port` to fail fast, or pick another `--ui-port`. For backend, change `--backend-port`.
- **UI cannot reach backend**: ensure `VITE_BACKEND_URL` points to a reachable host/IP (set `NEUROSURF_PUBLIC_HOST` if backend bound `0.0.0.0`).
- **LLM packages missing**: install the extra—`pip install -U 'neurosurfer[torch]'`—or see the Installation page for CUDA/MPS guidance.
- **`npm` not found**: install Node.js/npm or run with `--only-backend`.

!!! warning "Don’t call `app.run()` in your code when using the CLI"
    The CLI orchestrates child processes itself. If your file calls `ns.run()` at import time, you’ll see unexpected behavior. Keep run logic behind `if __name__ == '__main__':` or expose a factory like `create_app()`.

---

## Related Documentation

- **Getting Started** — installation and basic usage [:octicons-arrow-up-right-24:](./getting-started.md)
- **Configuration** — API keys, models, server settings & env vars [:octicons-arrow-up-right-24:](./api-reference/configuration.md)
- **API Reference** — classes, methods, schemas [:octicons-arrow-up-right-24:](./api-reference/index.md)

---

**Ready to launch?** Try:
```bash
neurosurfer serve --host 0.0.0.0 --port 8081
```