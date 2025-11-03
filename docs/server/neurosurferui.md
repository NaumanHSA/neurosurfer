---
title: Frontend (React)
description: NeurosurferUI — a React client for chat, threads, uploads for RAG, typed API client, and an extensible component system.
---

# NeurosurferUI

**NeurosurferUI** is a streamlined React frontend purpose-built for the Neurosurfer FastAPI backend. It speaks the backend’s chat and thread APIs, streams tokens over SSE, and manages predictable client state for conversations, uploads, and follow-ups. It favors OpenAPI-described endpoints and standard auth so you can extend capabilities without custom protocols or UI rewrites.

- **⚡ Typed client**: thin, type-safe wrappers for auth, models, chats, messages, and streaming
- **🧑‍🤝‍🧑 Per-user threads**: isolate histories, titles, and actions per account
- **💬 Chat UX**: durable threads, regenerate last answer, smooth token streaming, quick follow-ups
- **🧾 PDF export**: one-click export of any conversation with title and timestamps
- **📎 RAG uploads**: per-message file attach, progress chips, optional ingest for retrieval
- **🔐 Session-safe**: HttpOnly cookie auth, fast bootstrap from the current user endpoint, clean logout
- **🧩 Extensible**: plug custom renderers, tool panels, model pickers; feature-detect and adapt

---

## UI Tour (Coming Soon)

- Place a short demo GIF at `docs/assets/ui-tour.gif` showing login → new chat → streaming reply → follow-ups → PDF export.  
- Embed it via `![UI Tour](assets/ui-tour.gif)` so MkDocs copies it during build.  
- Capture path: open sidebar, create/select “New Chat,” send a prompt, watch streaming, click a follow-up, export from the chat menu.

!!! tip
    Keep the asset small (≤ 2–3 MB) for fast loads; prefer short clips over full sessions.

---

## How the UI talks to the backend

The UI uses `import.meta.env.VITE_BACKEND_URL` at build/dev time. If not provided, it falls back to:

```
${window.location.protocol}//${window.location.hostname}:8081
```

So by default the UI expects the API at **port 8081**. Override with:

```bash
# dev
VITE_BACKEND_URL=http://127.0.0.1:9000 npm run dev

# production build
VITE_BACKEND_URL=http://api.example.com:8081 npm run build
```

**TypeScript typings**: add `src/vite-env.d.ts`

```ts
/// <reference types="vite/client" />
interface ImportMetaEnv { readonly VITE_BACKEND_URL?: string; }
interface ImportMeta { readonly env: ImportMetaEnv; }
```

Ensure `tsconfig.json` includes `"types": ["vite/client"]` and includes that file.

---

## Development vs Production

### Dev (hot reload, Vite)
Use when iterating on UI code. Calls your running backend.

```bash
cd neurosurferui
npm ci
VITE_BACKEND_URL=http://127.0.0.1:8081 npm run dev
# UI at http://localhost:5173
```

### Production (prebuilt static assets)
Build once, then serve the static output via Neurosurfer CLI (recommended for users of the pip wheel):

```bash
cd neurosurferui
npm ci
VITE_BACKEND_URL=http://127.0.0.1:8081 npm run build   # outputs ./dist
```

Then copy the compiled assets into the Python package (see next section) so they ship inside the wheel.

---

## Shipping the UI inside the wheel (no Node required at runtime)

We provide a helper script that builds the React app and syncs its compiled assets into the Python package so the wheel contains `neurosurfer/ui_build/**`.

### 1) Use the helper script

From the repo root:

```bash
chmod +x scripts/build_ui.sh
./scripts/build_ui.sh         # auto-detects vite output (dist) and syncs -> neurosurfer/ui_build
```

Options:
- `--no-install` — skip `npm ci`
- `--no-clean` — avoid deleting removed files in the target (when rsync unavailable)
- `--out-dir=dist|build` — override output detection if needed

### 2) Include assets in packaging

**pyproject.toml**
```toml
[tool.setuptools]
include-package-data = true

[tool.setuptools.package-data]
neurosurfer = ["py.typed", "ui_build/**"]
```

**MANIFEST.in**
```
graft neurosurfer/ui_build
global-exclude *.map __pycache__ *.py[cod]
```

Rebuild & install:

```bash
rm -rf dist build *.egg-info
python -m build
pip install dist/neurosurfer-*.whl
```

Now the installed package contains `neurosurfer/ui_build/` with your React build.

---

## Serving the UI with the Neurosurfer CLI

The CLI supports three modes. It selects the best one automatically, but you can control it with flags.

### 1) Packaged static UI (default if present)
If your wheel contains `neurosurfer/ui_build/index.html`, the CLI can serve it via a lightweight static server (no FastAPI mounting).

```bash
neurosurfer serve --ui-port 5173
# Backend at http://0.0.0.0:8081, UI at http://0.0.0.0:5173
```

Under the hood, the CLI runs `npx serve -s neurosurfer/ui_build -l tcp://<ui_host>:<ui_port>` (falls back to global `serve` if `npx` is unavailable).

!!! note
    By default the UI **runs on a separate port** (`ui_port`) from the API (`backend_port`).  
    Ensure the backend allows CORS from the UI origin when they differ.

### 2) Dev mode from a source folder
Point `--ui-root` at your Vite project to run `npm run dev`:

```bash
neurosurfer serve --ui-root ./neurosurferui --ui-port 5173
# CLI injects VITE_BACKEND_URL -> http://<backend_host>:<backend_port> if not set
```

### 3) Static directory from a path
If you have a prebuilt directory (e.g., `dist` or any folder with `index.html`), you can serve it directly:

```bash
neurosurfer serve --ui-root ./neurosurferui/dist --ui-port 5173
```

The CLI will detect it’s a build folder and start `serve -s` on that path.

---

## CORS (when UI and API run on different ports)

Add this middleware to your FastAPI app factory:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        # add deployed UI origins here
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

If you deploy both behind the same domain/port or reverse proxy, you typically don’t need CORS.

---

## Troubleshooting

### “address already in use”
You tried to bind both API and static UI to the same port. Use different ports:
```bash
neurosurfer serve --ui-port 5173
```

### UI can’t reach backend
- Verify `VITE_BACKEND_URL` points to your API (scheme, host, port).
- If using different ports, enable CORS on the API (see above).

### TypeScript error: `import.meta.env` not found
Add `src/vite-env.d.ts` and `"types": ["vite/client"]` in `tsconfig.json` (see earlier section).

### NPM warnings about peer deps / deprecations
They’re typically transitive and not blocking. For Tailwind plugins, match the plugin’s version with your Tailwind major (e.g., `tailwind-scrollbar@^3` for Tailwind 3).

---

## Recommended workflow

1. **During UI development**
   ```bash
   cd neurosurferui
   VITE_BACKEND_URL=http://127.0.0.1:8081 npm run dev
   ```
2. **Before releasing a wheel**
   ```bash
   ./scripts/build_ui.sh   # writes to neurosurfer/ui_build
   python -m build
   twine upload dist/*
   ```
3. **User runs everything**
   ```bash
   pip install neurosurfer
   neurosurfer serve --ui-port 5173
   # open http://localhost:5173
   ```

---

## File structure (reference)

```
repo-root/
├─ neurosurfer/               # Python package (backend)
│  ├─ ui_build/             # (generated) copied from neurosurferui/dist
│  └─ ...
├─ neurosurferui/             # React app (Vite)
│  ├─ src/
│  ├─ public/
│  ├─ dist/                 # (generated) vite build output
│  └─ ...
├─ scripts/
│  └─ build_ui.sh           # helper to build & sync UI into neurosurfer/ui_build
└─ ...
```

With this setup, users get a single `pip install neurosurfer` and `neurosurfer serve` experience—no Node required at runtime—while you still have a great dev loop with Vite.