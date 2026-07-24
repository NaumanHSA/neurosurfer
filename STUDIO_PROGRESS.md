# Neurosurfer Studio — Progress Log

> What each studio phase actually shipped. Newest at the bottom. Roadmap +
> TODOs in [STUDIO_PLAN.md](STUDIO_PLAN.md).

---

## S0 — Foundations ✅

React 18 + Vite 6 + TypeScript + React Flow app in `studio/`, talking only to the
gateway `/v1/*` Phase-2 API.

- Typed API client (`src/api/client.ts`) + exact JSON types (`src/api/types.ts`)
  covering workflows, runs, events (read paths wired; run/resume/SSE ready).
- Graph adapter (`src/graph/adapter.ts`): graph JSON → React Flow nodes/edges;
  longest-path layering; edges derived from `depends_on` / router
  `routes`+`cases`+`default` / `on_error`, control edges deduped over data edges.
- Per-kind visual metadata (`src/graph/nodeKinds.ts`) — accent colors + glyphs.
- ComfyUI-style dark canvas: custom `WorkflowNode` cards (kind header, prompt,
  tool/flow chips, typed sockets), dotted background, controls, minimap; sidebar
  workflow list.
- Verified vs a live gateway with 3 demo workflows (branching/linear/loop);
  adapter smoke-tested on the branching graph.

---

## S1 — Canvas UX & interaction ✅ (core)

- **Draggable nodes** with position tracking.
- **Layout persistence** — positions saved to `localStorage` per
  `workflow@version` (`src/graph/layout.ts`); restored on open.
- **Auto-layout** — dagre (`@dagrejs/dagre`) left→right layered layout; "Auto
  arrange" default when no saved layout exists; smoke-tested (clean layered
  positions, no overlap on the branching graph).
- **Toolbar** (React Flow panel): Auto arrange · Reset layout · Fit · node search
  (Enter to jump+select by id/goal).
- **Minimap fix** — colored by kind, stroke/border tuned, pannable/zoomable.
- **Selection + edge highlight** — selecting a node accents its connected edges
  and dims the rest; centers the node.
- **Keyboard** — `f` fit view, `Esc` → graph overview (ignored while typing).

**Carried forward (S1):** nested loop/map/subgraph body rendering (still a chip
count), box multi-select, grid snap.

---

## S2 — Node inspector (read-only) ✅

Right-hand resizable/closable inspector (`src/components/Inspector.tsx`) driven by
a shared selection (`src/selection.ts`).

- **Node view** — glyph/id/kind header, blurb; **Config** tab with goal/purpose/
  expected_result prompt blocks, config rows (kind, writes, model, mode, rag,
  when, on_error, tools), and **kind-specific sections** (router routes/cases/
  default/repair; loop until/break_when/max_iterations/accumulate/body; map
  over/item_var/concurrency/body).
- **Relationships** — upstream (`depends_on`) and computed downstream, as
  clickable chips that select + center the target (`src/graph/relations.ts`).
- **Edge view** — from/to (clickable), kind (data/route/error), label.
- **Graph view** (click empty canvas) — name/description/version, declared
  inputs, outputs (clickable), node count, kind breakdown, fail_fast/strict.
- **Copy** — node id / full node JSON; per-prompt copy.
- **Tabs scaffold** — Config (live) · Last Run · Trace (placeholders for S3).

**Verification:** `npm run build` clean (tsc + vite); dagre layout smoke-tested.

---

## S3 — Run & Trace (live execution) ✅ (core)

Runs are now drivable and observable from the studio, over the Phase-2 REST+SSE
API — **no backend changes needed** for the core.

- **Run launcher** (`RunLauncher.tsx`) — modal inputs form generated from the
  workflow's declared `inputs`; typed widgets (string/number/bool/object-JSON),
  required validation; `POST /v1/workflows/{name}/runs`.
- **Live run** (`run/useRun.ts` + `run/types.ts`) — subscribes to the run's SSE
  stream; maps `start/ok/error/skipped` node events → live statuses; on `[DONE]`
  fetches the full run record for per-node outputs.
- **Canvas animation** — nodes show a status ring + pulsing dot
  (running/succeeded/failed), skipped nodes dim (`WorkflowNode` + `Canvas`).
- **Run status bar** (`RunStatusBar.tsx`) — status, elapsed timer, per-node
  counts, cancel; Resume button when `awaiting_input`.
- **Runs history** — sidebar Runs tab (`Sidebar.tsx`) lists `GET /v1/runs`;
  click loads a past run onto the canvas (statuses + results) for replay.
- **Inspector Last Run tab** — status, duration, output, error, skip reason from
  the run record. (Trace tab — tokens/tool calls — awaits a trace-detail **[BE]**.)
- **Resume / Cancel** — `POST …/resume` (reuses the launcher form) and
  `DELETE …/runs/{id}`.

**Verification:** `npm run build` clean (249 modules). Live end-to-end vs the
gateway: started a run on `article_summarizer`, streamed SSE
(`run running → summarize start → summarize error → title skipped → run failed →
[DONE]`), and confirmed the run record populated per-node `status`/`error`/
`duration_ms`/`skip_reason`. (The node erred on `Connection error.` — this box
can't reach api.openai.com — which exercised the failure→skip propagation path;
the success path is the same code with `status: ok` + populated `output`.)

**Carried forward (S3):** full per-node **Trace** tab (tokens, tool calls,
sub-spans) + **cost** — need a trace-detail API endpoint **[BE]**; loop/map
per-iteration progress **[BE]**; scrub/step replay.

**Next:** either the S3 **[BE]** trace-detail endpoint (unlocks the Trace tab +
cost), S1 nested loop/map body rendering, or **S4 — Authoring** (needs the
write/validate **[BE]** endpoints).

---

## S4 — Authoring (edit + create) ✅

The studio is now an editor: mutate a working draft, validate against the engine,
and save back — no YAML.

**Backend [BE]** (`app/server/api/routes_workflows.py`):
- `POST /v1/workflows/validate` — dry-run validate a graph JSON →
  `{ok, errors, gaps, warnings}` (structural parse errors surface as an error
  issue; orphan/wiring-floor as warnings).
- `POST /v1/workflows` — create (201; 409 on dup).
- `PUT /v1/workflows/{name}` — update graph + meta (404 missing; 422 invalid,
  with the validation report in `detail`; preserves version + created_by).
- `DELETE /v1/workflows/{name}` — delete (404 missing).
- Tests: `tests/test_workflow_authoring_api.py` (6) — validate ok/error/warning,
  full CRUD lifecycle incl. 409/404/422. Full workflow API suite: **17 passed**.

**Frontend:**
- Edit-mode toggle with a working **draft** graph, **dirty** indicator, and an
  unsaved-changes guard (beforeunload + confirm on workflow/run switch).
- Immutable edit helpers (`graph/edit.ts`) — add / delete / duplicate node with
  full reference cleanup (depends_on / routes / on_error / outputs), route + deps
  editors.
- **Editable inspector** (`Inspector.tsx`) — goal/purpose/expected_result,
  writes/model/when/tools, on_error + depends_on wiring, router routes + default,
  loop until/break_when/max_iter/accumulate, map over/concurrency; duplicate /
  delete actions.
- **Add node** menu (all 10 kinds) with sensible defaults.
- **Validate** + **Save** in the top bar; **Problems panel** (`ProblemsPanel.tsx`)
  listing errors/gaps/warnings (click → select node); per-node error/warning
  **badges** on the canvas; success/error **toasts**.
- Canvas keeps node positions stable across inspector edits (rebuilds layout only
  on structural add/remove; refreshes card content in place).

**S4 completion (second pass) — full editing:**
- **Drag-to-wire** — connect node handles in edit mode to create `depends_on`;
  from a router source it prompts for a route label (and adds the required dep);
  remove a wire from the edge inspector (`graph/edit.ts` connect/disconnect).
- **Inputs / outputs editors** — add/rename/type/required inputs; pick outputs by
  node (graph inspector, edit mode).
- **New-from-blank** — sidebar “＋ New workflow” creates a 1-node package and
  opens it in edit mode.
- **Undo/redo** — 60-deep history (toolbar ↶/↷ and ⌘Z / ⇧⌘Z; skips text fields
  so native text-undo still works).
- **Import / export** — export the graph as YAML; import YAML/JSON as a draft
  (js-yaml).
- **Template-var awareness** — available-`{var}` chips under prompt fields +
  an unknown-`{var}` warning (`availableVars` / `unresolvedVars`).

**Verification:** `npm run build` clean (252 modules). Live: edit→save round-trip
(validate 200, PUT 200, persisted) and new-from-blank create (201, valid) +
delete, both through the studio proxy; registry restored to the 3 demos. Backend
suites 17/17.

**Carried forward (minor):** node-search palette (double-click), inline `{var}`
autocomplete dropdown, version-bump prompt, per-field help text, layout in the
package, edit the nested loop/map body.

**Next:** S3 trace-detail **[BE]** (Trace tab + cost), S1 nested body rendering,
or **S5 — Architect-in-UI**.

---

## S5 — Architect-in-UI ✅ (core)

Describe an intent → watch the ReAct Architect design, test, and register a
workflow live → open it. The headline capability.

**Backend [BE]:**
- Exposed the live build session on `ArchitectAgent` (`self.session`) so an
  observer can snapshot the staged graph as it assembles.
- `app/server/architect_builds/` — `BuildRecord` (event log + graph snapshot +
  outcome) and `ArchitectManager`: runs `ArchitectAgent.build()` in a worker
  thread; the agent's `notify` callback appends a `log` event and snapshots
  `session.graph_dict()` as a `graph` event; terminal maps to succeeded /
  blocked (`WorkflowInfeasible`) / failed. Registers into the shared registry;
  tools auto-approved with a log line. `agent_factory` is injectable for tests.
- `routes_architect.py` — `POST /v1/architect/builds`, `GET …/builds`,
  `GET …/builds/{id}` (`?events=true`), `GET …/builds/{id}/events` (SSE, replay
  + tail, `[DONE]`). Mounted in `api/router.py`.
- Tests: `tests/test_architect_builds_api.py` (5) — fake-agent build success
  with log/graph events, registration visible via the workflow API, SSE
  replay+close, list, 422/404. Backend suites: **22 passed**.

**Frontend:**
- `architect/useArchitect.ts` — starts a build, streams SSE (`log` → step log,
  `graph` → live snapshot, `build` → status/outcome), fetches the final record.
- `components/ArchitectDock.tsx` — bottom dock: intent textarea + verify select +
  example chips; live streaming step log; terminal outcome (Open workflow /
  blocked / failed).
- App: **✨ Architect** top-bar button; while a build runs the canvas shows the
  staged graph assembling (build snapshots feed a synthetic detail); **Open
  workflow** loads the registered result (runnable/editable). Client:
  `startArchitectBuild` / `getArchitectBuild` / `streamArchitectBuild`.

**Verification:** `npm run build` clean (254 modules). Live: endpoints mounted
(422/200), and a real build ran end-to-end through the real
manager→agent→SSE path — it terminated `failed` "Connection error." (this box
can't reach api.openai.com), exercising the failure path the dock renders. With
LLM access the log + graph snapshots stream and the workflow registers.

**Carried forward (S5):** requirement-gathering (`ArchitectConversation`
questions + answers endpoint), interactive tool approval (show code/tests),
structured verification view (criteria + judge verdicts), reasoning/token stream,
iterate ("refine this"), build cancel.

**Next:** S3 trace-detail **[BE]** (Trace tab + cost), S1 nested body rendering,
S5 depth, or **S6 — tools/capabilities/observability browsers**.
