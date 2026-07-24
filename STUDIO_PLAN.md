# Neurosurfer Studio — Master Plan

> The visual studio for neurosurfer workflows. A graph-first web app to
> **design, inspect, run, trace, debug, and iterate** on agent workflows —
> ComfyUI for agents, with the Architect living inside the canvas.
>
> This is the Phase 7 deliverable from [ARCHITECT_V2_PLAN.md](ARCHITECT_V2_PLAN.md),
> expanded into its own roadmap. Progress log lives in
> [STUDIO_PROGRESS.md](STUDIO_PROGRESS.md) (created as phases land).
>
> **Phase legend:** `S0`…`S9`. Newest detail near the top of each phase.
> Check boxes as items ship. Backend-dependent items are tagged **[BE]**.

---

## 0. Vision & big picture

Neurosurfer already has a rich control-flow engine, a self-verifying Architect
agent, and an execution API. What it lacks is a **face** — a place a human can
*see* a workflow, understand it, run it, watch it think, find where it broke, and
fix it, without reading YAML or tailing logs.

The studio is that face. Three audiences, one app:

1. **The operator** — opens a registered workflow, runs it with real inputs,
   watches it execute node-by-node, inspects each node's trace, reruns/resumes.
2. **The builder** — edits a graph on the canvas (rewire, retune prompts, add
   nodes), validates, and registers — or starts from a blank canvas.
3. **The director** — describes an intent in plain English and watches the
   Architect agent design, test, self-repair, and register a workflow live,
   then takes over and refines it.

**North star:** from "I have an idea" to "a tested, registered, running
workflow" — entirely in the browser, with full visibility at every step.

### Design principles
- **Graph-first.** The canvas is the primary surface; panels serve it.
- **The graph is the source of truth.** The UI is a view/editor over the same
  Phase-1 IR the engine runs — never a parallel model that can drift.
- **Everything is inspectable.** Any node, edge, run, or agent step can be
  clicked to reveal what it is and what it did.
- **Read before write.** Every phase ships a useful read-only slice before its
  editing counterpart.
- **ComfyUI-grade polish, agent-native semantics.** Borrow the feel (dark
  canvas, typed sockets, node search, queue panel); model *our* concepts
  (agents, tools, routers, loops, verification) — not image-gen nodes.

---

## 1. Current state — S0 Foundations ✅ (shipped)

- `studio/` app: React 18 + Vite 6 + TypeScript + React Flow (`@xyflow/react`).
- Typed Phase-2 API client (`src/api/`) + exact JSON types.
- Graph-JSON → React Flow adapter: longest-path layering; edges derived from
  `depends_on` / router `routes`+`cases`+`default` / `on_error`, with control
  edges winning over duplicate data edges.
- ComfyUI-style dark canvas: custom `WorkflowNode` cards (kind-colored header,
  glyph, prompt, tool/flow chips, typed sockets), dotted background, controls,
  minimap.
- Sidebar workflow list; loads + renders any registered workflow read-only.
- Verified end-to-end against a live gateway with 3 demo workflows
  (branching / linear / loop).

**Known gaps carried into S1+** (from first live review):
- Nodes are static (not draggable); no manual/auto layout control.
- Clicking a node does nothing (no inspector).
- No editing of any kind.
- Minimap renders empty / not the real graph (dimensions/coloring bug).
- Loop/map bodies shown as a chip count, not as real nested nodes.

---

## 2. Architecture

```
┌────────────────────────────── studio (browser) ──────────────────────────────┐
│  React + React Flow                                                            │
│  ┌───────────┐  ┌──────────────────────────┐  ┌────────────────────────────┐ │
│  │  Sidebar  │  │        Canvas             │  │        Inspector           │ │
│  │ workflows │  │  nodes · edges · runs     │  │ node/edge/run/trace detail │ │
│  │  runs     │  │  live animation           │  │ config editor              │ │
│  │  tools    │  │                           │  │                            │ │
│  └───────────┘  └──────────────────────────┘  └────────────────────────────┘ │
│  ┌──────────────────────────── Architect dock ───────────────────────────────┐│
│  │  intent chat · live build steps · tool approvals · verification report     ││
│  └────────────────────────────────────────────────────────────────────────────┘│
│         │ typed client (REST + SSE)                                            │
└─────────┼──────────────────────────────────────────────────────────────────────┘
          ▼
   neurosurfer gateway  (FastAPI, /v1/*)  ── run manager · registry · architect · tools · MCP
```

- **State:** a client graph store (Zustand or React Flow state + context) holds
  the working graph, selection, run state, and dirty tracking. React Query (or a
  thin fetch layer) for server cache.
- **Layout metadata:** node positions are UI concern; persisted client-side
  first (localStorage per workflow), optionally to the package later.
- **Transport:** REST for CRUD; **SSE** for run streams and architect streams
  (EventSource; token via same-origin proxy in dev / reverse proxy in prod).
- **Source of truth:** the server's graph JSON. The editor mutates a local copy
  and round-trips through validate → save.

---

## 3. Backend API gaps (gate several phases) — **[BE]**

The current gateway is **read + run only**:
`GET /v1/workflows`, `GET /v1/workflows/{name}`, `POST /v1/workflows/{name}/runs`,
`GET /v1/runs…`, `GET /v1/runs/{id}/events` (SSE), `…/nodes/{id}`, `…/resume`,
`DELETE /v1/runs/{id}`.

To reach the full vision we need these **new** endpoints (each tagged where it's
consumed below):

- [x] **Validate (dry-run)** — `POST /v1/workflows/validate` `{graph}` →
      `{ok, errors, gaps, warnings}`. Powers inline validation before save. *(S4)*
- [x] **Create workflow** — `POST /v1/workflows` `{name, graph, meta}` → 201
      (409 on dup). *(S4)*
- [x] **Update workflow** — `PUT /v1/workflows/{name}` `{graph, meta, version}`
      (404 if missing, 422 on invalid). *(S4)*
- [x] **Delete workflow** — `DELETE /v1/workflows/{name}` (404 if missing). *(S4)*
- [ ] **Layout persistence** — either store positions in the package
      (`graph.ui` metadata) or `PUT /v1/workflows/{name}/layout`. *(S1/S4)*
- [ ] **Node trace detail** — enrich `…/nodes/{id}` (or a new
      `…/nodes/{id}/trace`) with tokens, tool calls, timing, sub-spans parsed
      from the persisted trace JSON. *(S3)*
- [ ] **Tools catalog** — `GET /v1/tools`, `GET /v1/tools/{name}` (full input
      JSON schema, workflow-usable flag). *(S6)*
- [ ] **Capabilities / node kinds** — `GET /v1/capabilities` exposing the
      `KnowledgeBase` manifest (kinds, fields, expression fns, version). *(S6)*
- [ ] **MCP servers** — `GET /v1/mcp/servers` (+ connect status). *(S6)*
- [x] **Architect build stream** — `POST /v1/architect/builds` `{intent, verify}`
      → build id; `GET /v1/architect/builds`, `…/{id}`, and `…/{id}/events`
      (SSE of `log` + `graph` snapshots + terminal `build`). *(S5)* *(carry:
      `…/{id}/answers` for requirement-gathering + interactive tool-approve
      endpoint; tools are auto-approved today.)*
- [ ] **Auth** — confirm/extend bearer-token; optional short-lived token for
      EventSource query param, or document reverse-proxy injection. *(S7)*

> Each backend item ships **with** its consuming frontend phase, with its own
> tests, and gets logged in the neurosurfer progress log too.

---

## 4. Phased roadmap

### Phase S1 — Canvas UX & interaction ✅ (core; nested-body carried forward)

**Goal.** Make the canvas feel alive and manipulable: move nodes, lay them out,
navigate large graphs, and render nested control flow properly.

- [x] **Draggable nodes** — enable node dragging; track positions in the store.
- [x] **Layout persistence** — save positions to `localStorage` keyed by
      workflow name+version; restore on open; "Reset layout" action.
      *(later: server-side via **[BE]** layout endpoint.)*
- [x] **Auto-layout** — integrate a layered layout (dagre or elkjs); "Auto
      arrange" button; used as the default when no saved layout exists.
      Left→right rank by `depends_on`, control edges respected.
- [x] **Fix the minimap** — render the real graph (ensure custom nodes report
      dimensions; color by kind); pannable/zoomable; correct viewport mask.
- [ ] **Nested loop/map/subgraph rendering** — render `body` sub-graphs inside
      an expandable container node (React Flow parent/child or subflow), with a
      collapse/expand toggle; iteration/concurrency badges on the container.
      *(carried forward — still a `body: N` chip.)*
- [x] **Canvas controls** — fit-view + zoom-to-selection (centering on select).
      *(carry: zoom %, grid snap toggle, background style toggle.)*
- [x] **Selection** — single select; highlight connected edges; dim the rest on
      focus. *(carry: multi/box select.)*
- [x] **Node search / find** — toolbar search, Enter to jump+center by id/goal.
- [x] **Edge polish** — distinct styling per edge kind (data / route / error),
      selected/highlight states, readable labels.
- [x] **Keyboard shortcuts** — `f` fit, `Esc` deselect (ignored while typing).
      *(carry: delete/duplicate/select-all in edit mode; cheatsheet overlay.)*
- [x] **Empty / loading / error states** — loading + error hints, empty-state
      copy. *(carry: skeletons, retry button, "open the Architect" CTA.)*

**Done when:** a user can freely arrange a workflow, the layout persists, the
minimap reflects it, and loops/maps show their inner nodes. *(Met except nested
body rendering — carried forward.)*

---

### Phase S2 — Node inspector (read-only detail) ✅

**Goal.** Click anything → understand it fully. No editing yet.

- [x] **Inspector panel** — right-hand, closable; graph overview as the empty
      state. *(carry: drag-resize.)*
- [x] **Common fields** — id, kind (icon/label), goal, purpose, expected_result,
      `writes`, `depends_on` (clickable chips), `when`, `on_error`, `tools`,
      model, mode, `rag`. *(carry: `output_schema` render.)*
- [x] **Kind-specific sections:**
  - router → routes/cases table (label → target, clickable), default, repair.
  - loop → until / break_when, max_iterations, accumulate, body summary.
  - map → over, concurrency, item_var, body summary.
  - *(carry: subgraph body summary; input prompt/required value.)*
- [x] **Relationships** — upstream (deps) and downstream (consumers) as clickable
      chips; select jumps + centers.
- [x] **Edge inspector** — click an edge → kind, source/target (clickable),
      label (route case / on_error).
- [x] **Graph inspector** — click empty canvas → name, description, version,
      declared inputs, outputs, fail_fast/strict_inputs, node count, kind mix.
- [x] **Copy affordances** — copy node id / prompt / full node JSON.
- [x] **Tabs scaffold** — Config (live) · Last Run (S3) · Trace (S3).

**Done when:** every node/edge/graph element reveals its full config on click,
with navigable relationships. ✅

---

### Phase S3 — Run & Trace (live execution) ✅ (core; trace-detail + iteration carried forward)

**Goal.** Run a workflow from the UI and watch it think; inspect every node's
real I/O afterward. Highest operator value.

- [x] **Run launcher** — modal form auto-generated from declared `inputs` (typed
      widgets: string/number/bool/object-JSON), required-field validation,
      "Run" → `POST …/runs`.
- [x] **Live run view** — subscribe to `…/events` SSE; animate node states
      (pending → running → succeeded/failed/skipped) with color + pulsing dot.
      *(carry: progress overlay polish.)*
- [x] **Branch visualization** — skipped nodes dim; skip reasons shown in the
      Last Run tab. *(carry: explicit taken-path emphasis.)*
- [ ] **Loop/map progress** — per-iteration counter / fan-out progress on the
      container node. *(needs nested-event bubbling — carried forward.)* **[BE]**
- [~] **Node result panel** — **Last Run** tab shows status, duration, output,
      error, skip reason from the run record (shipped). Full **Trace** tab
      (tokens, tool calls, sub-spans from the persisted trace) still needs a
      trace-detail endpoint. **[BE]**
- [x] **Run status bar** — status, elapsed timer, per-node counts, cancel.
- [x] **Runs list / history** — sidebar Runs tab: past runs with status dots,
      time, workflow; click to load.
- [x] **Replay** — load a finished run onto the canvas (statuses + per-node
      results). *(carry: scrub/step through events.)*
- [x] **Resume** — `awaiting_input` runs offer Resume; form → `POST …/resume`.
- [x] **Cancel** — `DELETE …/runs/{id}` with UI feedback.
- [ ] **Cost/tokens** — per-node + per-run token usage; Langfuse link. **[BE]**
- [x] **Error surfacing** — failed nodes go red; error in Last Run + run bar.
      *(verified live: failed→skipped propagation rendered correctly.)*

**Done when:** a user runs a workflow, watches it execute live, and inspects any
node's result — matching the Phase 7 "v1 done" bar. *(Met for the run-record
path; rich per-node trace + cost carried to the trace-detail **[BE]** endpoint.)*

---

### Phase S4 — Authoring (edit + create) ✅

**Goal.** Turn the viewer into an editor. Round-trips through validate → save.
Requires the **[BE]** write/validate endpoints (§3).

- [x] **[BE] Write + validate endpoints** — validate (dry-run), create, update,
      delete workflow packages. Tested (`tests/test_workflow_authoring_api.py`,
      6 tests). *(carry: positions in package UI metadata.)*
- [x] **Edit mode toggle** — view/edit switch; dirty-state dot; unsaved-changes
      guard (beforeunload + discard confirms on workflow/run switch).
- [x] **Editable inspector** — edit goal/purpose/expected_result, tools, model,
      `writes`, `when`, `on_error`, and kind-specific fields.
      *(carry: per-field help text from capabilities.)*
- [x] **Router editor** — add/remove routes (label → target picker), default.
      *(carry: reorder, repair toggle, cases builder.)*
- [x] **Loop/map editor** — until / break_when, max_iterations, accumulate;
      over / concurrency. *(carry: edit the nested body.)*
- [x] **Add node** — kind menu; insert with sensible defaults.
      *(carry: ComfyUI-style double-click node search.)*
- [x] **Delete / duplicate node** — with full reference cleanup (depends_on /
      routes / on_error / outputs).
- [x] **Wire edges by dragging sockets** — drag between node handles creates a
      `depends_on`; from a router source it prompts for the route label (and adds
      the required dep). Remove an edge from the edge inspector.
- [x] **Inputs & outputs editors** — add/rename/type/required graph inputs;
      select outputs by node (graph inspector, edit mode).
- [x] **Inline validation** — problems panel + per-node error/warning badges
      from the validate endpoint; click a problem → select the node.
- [x] **Save** — validate then update; refuse on hard errors (422 surfaces the
      report). *(carry: version-bump prompt.)*
- [x] **New workflow from blank canvas** — sidebar “＋ New workflow” → create +
      open in edit mode.
- [x] **Undo/redo** — history stack (toolbar ↶/↷ + ⌘Z / ⇧⌘Z).
- [x] **Import / export** — export graph as YAML; import YAML/JSON as a draft.
- [x] **Template var awareness** — available-`{var}` chips under prompts +
      unknown-`{var}` warning. *(carry: inline autocomplete dropdown.)*

**Done when:** a user can build a valid multi-node branching workflow from
scratch on the canvas, validate it inline, and register it — no YAML. ✅

**Done when:** a user can build a valid multi-node branching workflow from
scratch on the canvas, validate it inline, and register it — no YAML.

---

### Phase S5 — Architect-in-UI ✅ (core; requirement-gathering / interactive tool-approval / verification-view carried forward)

**Goal.** The differentiator: describe an intent, watch the agent build it live,
approve tools, see verification — then edit the result (S4).

- [ ] **[BE] Architect build stream** — start build, SSE step events, tool
      approval, requirement answers (§3).
- [x] **[BE] Architect build stream** — `POST /v1/architect/builds`,
      `GET …/builds`, `GET …/builds/{id}`, `GET …/builds/{id}/events` (SSE of
      log + graph snapshots + terminal). Tested (`test_architect_builds_api.py`,
      5). *(carry: cancel, requirement-answers, interactive tool-approve
      endpoints.)*
- [x] **Intent dock** — intent textarea + verify select + example chips.
      *(carry: model / max-turns options.)*
- [x] **Live build animation** — the step log streams in the dock and the canvas
      shows the staged graph assembling (from the build's graph snapshots).
- [ ] **Requirement gathering** — render `ArchitectConversation` clarifying
      questions (choices) and post answers back. *(carried forward — needs an
      answers endpoint.)*
- [~] **Tool approval** — authored tools are auto-approved with a log line.
      Interactive in-app approve/reject (show code + tests) carried forward.
- [ ] **Verification view** — acceptance criteria + per-criterion judge verdicts
      as structured UI. *(carried forward — verify runs; results show in the log
      today.)*
- [x] **Terminal outcome** — registered (**Open workflow** → loads it, runnable/
      editable), blocked (`WorkflowInfeasible` reason), or failed (error).
- [~] **Reasoning stream** — the notify step log streams live. *(carry:
      expandable tool-call trace + token/cost meter.)*
- [ ] **Iterate** — "refine this" follow-up re-entering the loop. *(carried
      forward.)*

**Done when:** a user types an intent and watches a tested workflow assemble on
the canvas, then opens it. *(Met: intent → live build (log + canvas) → open.
Requirement-gathering, interactive tool-approval, structured verification view,
and iterate carried forward.)*

---

### Phase S6 — Capabilities & Observability browsers 

**Goal.** The reference + operations surfaces around the canvas.

- [ ] **[BE] Tools / capabilities / MCP endpoints** (§3).
- [ ] **Tool catalog** — searchable list (native / generated / MCP) with
      descriptions, input schemas, workflow-usable flag; "insert as node" in
      edit mode.
- [ ] **Node-kind reference** — the engine's kinds + fields + expression
      language (from the manifest), with examples; capability version badge.
- [ ] **MCP panel** — configured servers, connection status, exposed tools.
- [ ] **Runs dashboard** — table across workflows (filter by workflow / status /
      date); success rate, avg duration, token/cost aggregates.
- [ ] **Trace explorer** — richer per-run span tree; deep links to Langfuse.
- [ ] **Author-tool UI** (standalone) — write/generate a tool, sandbox-run,
      approve — outside an architect build.

**Done when:** builders can discover every capability and operators can analyze
runs — without leaving the studio.

---

### Phase S7 — Auth, polish, deploy, testing, docs 

**Goal.** Make it shippable to more than a local dev.

- [ ] **Login screen** over the gateway bearer token; token stored securely;
      401 handling + re-auth. **[BE]** (confirm auth model, EventSource token).
- [ ] **Settings** — gateway URL, token, theme, layout prefs.
- [ ] **Design system pass** — tokens, spacing, typography, motion; light theme;
      consistent components (buttons, inputs, dialogs, toasts, tooltips).
- [ ] **Resizable / collapsible panels**; remembered layout.
- [ ] **Accessibility** — keyboard nav, focus states, ARIA, contrast.
- [ ] **Performance** — virtualize large graphs / long lists; memoized nodes;
      throttled SSE re-render; lazy routes.
- [ ] **Testing** — Vitest unit (adapter, store, client), component tests
      (inspector, run view), Playwright e2e against a live gateway.
- [ ] **Error handling** — global boundary, toasts, offline/gateway-down banner.
- [ ] **Build & deploy** — production build; serve behind the gateway (static
      mount) or standalone; env config; Docker.
- [ ] **Docs** — user guide (view/run/build/architect), screenshots/gifs,
      contributor setup; link from README.
- [ ] **Telemetry (opt-in)** — basic usage analytics for UX decisions.

**Done when:** a new user can log in, and the studio is documented, tested, and
deployable.

---

### Phase S8 — Stretch / future

- [ ] Collaboration: multi-user presence, comments on nodes, shared editing.
- [ ] Workspaces / projects; workflow tags, search, favorites.
- [ ] Versioning & diff: visual graph diff between versions/runs; rollback.
- [ ] Workflow composition: drag a registered workflow in as a `subgraph`.
- [ ] Scheduling / triggers: cron + webhook launch from the UI.
- [ ] A/B & eval harness UI (ReAct vs pipeline; intent suites) with charts.
- [ ] Templates gallery: start from example workflows (branching/loop/map/MCP).
- [ ] Prompt playground per node (try a prompt against the model in isolation).
- [ ] Mobile/tablet read-only run monitoring.

---

## 5. Cross-cutting concerns

- **State management:** introduce a graph store (Zustand) in S1 before editing
  complexity lands; keep server cache separate.
- **Type safety:** consider generating the API client from the gateway's
  `openapi.yaml` to prevent drift; keep hand-written types until then.
- **Design tokens:** centralize the dark theme now (done in `styles.css`); add
  light theme + a tokens file in S7.
- **Testing discipline:** every phase adds tests; adapter/store logic must stay
  unit-covered (S0's adapter already smoke-tested).
- **Backend co-evolution:** each **[BE]** item is a small neurosurfer change
  with its own pytest coverage and a progress-log entry — the studio never forks
  the model.
- **Trace fidelity:** reuse the existing Langfuse/trace plumbing; the node trace
  panel parses persisted traces rather than inventing a new store.

---

## 6. Open decisions (resolve as we reach them)

- **Layout storage:** localStorage-only vs. persisted in the package
  (`graph.ui`) — decide in S1; lean localStorage first, package later.
- **State lib:** Zustand vs. React Flow store + context — decide entering S1.
- **Editing model:** live-edit the server graph vs. staged draft + explicit
  save — lean **staged draft + validate + save** (safer, undo-friendly).
- **Architect transport:** reuse run SSE shape vs. a dedicated event schema —
  decide in S5 (lean dedicated, richer step semantics).
- **Auth for SSE:** query-param token vs. reverse-proxy injection — decide in
  S7.
- **Deploy target:** static mount behind gateway vs. standalone app — decide in
  S7.

---

## 7. Immediate next steps

Ordered to attack the user's stated pain points first:

1. ~~**S1 canvas interaction** — draggable nodes + layout persistence + minimap
   fix.~~ ✅
2. ~~**S2 read-only inspector** — click a node → see goal/config.~~ ✅
3. ~~**S3 run & trace** — the biggest operator payoff.~~ ✅ (core; trace-detail
   **[BE]** + iteration progress carried forward)
4. ~~**S4 authoring** — full editing: drag-wire, inputs/outputs, add/delete/
   duplicate, undo/redo, import/export, new-from-blank, validate + save.~~ ✅
5. ~~**S5 Architect-in-UI** — intent → live build (log + canvas) → open.~~ ✅
   (core; requirement-gathering / interactive tool-approval / verification-view
   carried forward)
6. **Next candidates:**
   - **S3 trace-detail [BE]** — node trace-detail endpoint → Trace tab + cost.
   - **S1 nested loop/map body rendering** — render `body` sub-graphs inline.
   - **S6 tools/capabilities browsers**, **S7 auth/polish/deploy**.
   - **S5 depth** — requirement-gathering, interactive tool approval,
     structured verification view, iterate.

S0–S5 cores are done. Remaining high-value items are the Trace tab ([BE]),
S5 depth, and the S6/S7 surfaces.
