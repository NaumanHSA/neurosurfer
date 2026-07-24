# neurosurfer studio

A visual studio for neurosurfer workflows — see, run, and trace graphs in the
browser. ComfyUI-inspired dark, graph-first UI built on React Flow, riding the
neurosurfer Phase-2 workflow API.

> **Status: M0 (Foundations).** Read-only: browse registered workflows and
> render any one as a graph with control-flow visual language (routers, loops,
> maps, error edges). M1 adds the node inspector; M2 adds live run + per-node
> traces. See `ARCHITECT_V2_PLAN.md` Phase 7 for the roadmap.

## Stack

React 18 · Vite 6 · TypeScript · [@xyflow/react](https://reactflow.dev) (canvas).
No backend of its own — it talks only to the neurosurfer gateway's `/v1/*` API.

## Develop

```bash
cd studio
npm install
# point at your running gateway (default http://localhost:8000)
export NEUROSURFER_GATEWAY=http://localhost:8000
npm run dev            # http://localhost:5273  (proxies /v1 -> gateway)
```

Start the gateway separately (from the repo root) so there are workflows to see.
If the gateway's bearer-token auth is enabled, set `VITE_NEUROSURFER_TOKEN`
(see `.env.example`).

## Layout

| Path | Purpose |
|---|---|
| `src/api/` | Typed Phase-2 client (`client.ts`) + JSON shapes (`types.ts`). |
| `src/graph/adapter.ts` | Graph JSON → React Flow nodes/edges (layout + edge derivation). |
| `src/graph/nodeKinds.ts` | Per-kind visual metadata (accent colors, glyphs). |
| `src/components/` | `Canvas`, `WorkflowNode` (the node card), `Sidebar`. |

## Scripts

- `npm run dev` — dev server with HMR + gateway proxy.
- `npm run build` — typecheck + production build to `dist/`.
- `npm run typecheck` — types only.
