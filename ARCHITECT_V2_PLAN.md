# Architect V2 — "Agentic Architect + Rich Graph Engine + Visual Studio"

> A phased build plan to evolve the Architect from a fixed, linear, single-pass
> workflow generator into (1) a fully-featured graph runtime, (2) a self-aware
> agent that designs, builds, tests, and repairs pipelines, and (3) an
> n8n/ComfyUI-style visual studio to author and trace them.

**Status:** planning · **Owner:** Neurosurfer Team · **Last updated:** 2026-07-22

---

## 0. Vision & principles

**Goal.** Turn the Architect into a true agent that *knows neurosurfer*
(node kinds, tool contract, MCP), researches, authors and tests missing tools,
constructs **multi-node conditional graphs**, then **runs, analyses, and repairs
the whole pipeline** — surfaced through a **rich visual UI** that shows the graph,
per-node params/configs, and live execution traces.

**Locked-in decisions (from design review):**

1. **Engine:** fully-featured — conditional edges, router/branch, bounded loop,
   map/fan-out, plus sub-workflows, error/fallback routing, join/gather, typed
   shared state, human-in-the-loop. Authorable via **YAML *and* a programmatic
   builder API**, and executable via a **REST + streaming API**.
2. **Agent:** try **ReAct first** (single planner + toolbelt). Keep the existing
   staged pipeline as a measurable fallback baseline; don't delete it until ReAct
   demonstrably wins.
3. **UI:** a **new rich web app** (not static files) talking to the FastAPI
   gateway over REST + websocket/SSE.
4. **Self-knowledge:** **auto-derived** from code + docs, never hand-guessed, so it
   stays correct as the framework evolves.

**Guiding principles.**

- **Engine before agent before UI.** The agent can't design what the engine can't
  run; the UI can't show what the engine can't emit.
- **Every phase ships something usable on its own.**
- **Backward compatible:** existing `base`/`react`/`function`/`tool` linear graphs
  keep running unchanged. New constructs are additive.
- **Deterministic where possible, LLM where necessary.** Prefer expression-based
  routing/conditions; fall back to LLM routing for semantic decisions.
- **Nothing untrusted auto-runs.** Keep the sandbox + human-approval contract for
  authored tools and any generated code.

**Dependency graph.**

```
Phase 1 (Engine core: state + control flow)
   └─▶ Phase 2 (Execution API + streaming)
          ├─▶ Phase 3 (Self-knowledge manifest)
          │      └─▶ Phase 4 (ReAct Architect agent)
          │             └─▶ Phase 5 (Closed-loop test & repair)
          │                    └─▶ Phase 6 (Conditional design + MCP)
          └────────────────────────────▶ Phase 7 (Visual studio UI)
Phase 8 (Hardening, docs, examples) runs continuously, closes each phase.
```

---

## Phase 1 — Rich graph engine: typed state + control flow

> **Status: ✅ COMPLETE** (all of 1a–1j landed; 64 new tests; full suite 424 passing,
> ruff clean, new modules mypy-clean). Delivered:
> - `engine/state.py` — `WorkflowState` (typed inputs/nodes/vars + iteration scope + JSON snapshot)
> - `engine/expressions.py` — restricted-AST evaluator (`evaluate` / `safe_bool`), sandboxed
> - `engine/schema.py` — new kinds `router`/`loop`/`map`/`subgraph`/`input` + fields
>   (`when`, `writes`, `on_error`, `cases`, `default`, `body`, `max_iterations`,
>   `break_when`, `accumulate`, `over`, `as`, `concurrency`, `options`) + `RouterCase`
> - `engine/executor.py` — dynamic scheduler: OR-join, `when`-pruning, router pruning,
>   loop/map/subgraph/input runners, `on_error` routing, `policy.retries`, `seed_state`
> - `engine/loader.py` — control-flow validation (router targets, body sub-graphs, expr syntax)
> - `engine/builder.py` — `GraphBuilder` fluent API with YAML/JSON round-trip parity
> - Tests: `test_graph_state_expressions/control_flow/iteration/error_routing/subgraph_input/builder_roundtrip.py`
>
> **Carried forward** (core delivered; see `ARCHITECT_V2_PROGRESS.md`): per-iteration
> events → Phase 2; configurable gather/reduce; retry backoff; registered-package
> sub-workflows (inline `subgraph` done, named-package refs deferred to workflow layer).

**Goal.** Extend the runtime from a run-once topological DAG into an expressive
workflow engine. Pure runtime work — no Architect changes. Everything additive
and YAML-round-trippable.

**Depends on:** nothing. This is the foundation.

### 1a. Typed shared state (prerequisite for conditions)
- [x] Design a `WorkflowState` object: typed, namespaced key/value store threaded
      through execution (node outputs + explicit variables), replacing implicit
      string-passing for anything a condition needs to read.
- [x] Define how node outputs are written to state (by node id) and how nodes
      declare which state keys they read (`reads:` / `writes:` in YAML).
- [x] Backward-compat shim: existing graphs that pass raw strings between
      `depends_on` nodes keep working (state auto-populated from node outputs).
- [x] Serialization: state snapshot is JSON-serializable (needed by the UI + trace).

### 1b. Safe expression evaluator
- [x] Pick + implement a sandboxed expression language for edge predicates and
      conditions (restricted AST eval — no imports/attribute escapes — over state).
      Decide: reuse a vetted lib vs. a small in-house evaluator. Document the grammar.
- [x] Support common ops: comparisons, boolean logic, membership, length, simple
      string ops, existence checks (`state.foo.status == "ok"`, `len(state.items) > 0`).
- [x] Unit-test the evaluator against injection/escape attempts.

### 1c. Conditional edges
- [x] Extend the edge/`depends_on` model to support a `when:` predicate (expression)
      gating whether the edge activates.
- [x] Executor: a node runs only if its incoming edges' conditions are satisfied;
      define AND/OR semantics for multiple incoming edges.
- [x] Skip-propagation: downstream of a not-taken branch is cleanly `skipped`
      (reuse existing skip machinery), not `errored`.

### 1d. Router / branch node
- [x] New node kind `router`: evaluates a set of `cases` (expression → target node)
      plus a `default`, selecting one (or configurably N) outgoing path(s).
- [x] LLM-router variant: when the decision is semantic, the router is an LLM call
      returning one of the declared case labels (constrained/validated output).
- [x] YAML schema + validation (every case target must be a real node).

### 1e. Bounded loop / iterate-until
- [x] New construct `loop`: repeat a node or sub-graph until a `break_when`
      expression is true or `max_iterations` is hit (hard ceiling, always required).
- [x] Per-iteration state scoping (iteration index, accumulator) + guaranteed
      termination (no unbounded loops ever).
- [x] Emit per-iteration node events so the UI/trace can show progress.

### 1f. Map / fan-out + gather
- [x] New construct `map`: run a node/sub-graph once per item of a collection in
      state, with a concurrency cap.
- [x] `gather`/`join` node: wait for a set of branches (or all map items) and merge
      their outputs into state (list / dict / reduce strategy).
- [x] Wire into the existing concurrency model (the executor already runs
      independent nodes; make fan-out explicit and bounded).

### 1g. Error / fallback routing
- [x] `on_error:` edge or `try/fallback` construct: route to a handler node when a
      node errors, instead of failing the whole graph (opt-in; `fail_fast` still honored).
- [x] Retry policy surfaced per node (extend existing `NodePolicy.retries`/`timeout_s`)
      with backoff; distinguish "retry same node" from "route to fallback".

### 1h. Sub-workflow node (composition)
- [x] New node kind `workflow` (or `subgraph`): call a registered workflow package
      as a single node, mapping parent state → child inputs → parent state.
- [x] Recursion guard (max nesting depth) + cycle detection across packages.

### 1i. Human-in-the-loop node
- [x] New node kind `input`/`approval`: pause execution, emit a "needs input" event,
      resume when the answer arrives (drives both CLI prompts and UI dialogs).
- [x] Design the pause/resume protocol so it works for a long-lived API run (durable
      run id + resume endpoint — coordinate with Phase 2).

### 1j. Schema, validation, and round-trip
- [x] Extend `GraphNode`/`Graph` schema + `_VALID_NODE_KINDS` for all new kinds.
- [x] Extend `validate_package` to cover the new constructs (case targets exist,
      loops have ceilings, map has a source collection, no orphan branches, state
      keys referenced by conditions are produced somewhere).
- [x] **Programmatic builder API:** a fluent Python `GraphBuilder` that constructs
      the same IR as YAML (`.node(...).router(...).loop(...).edge(when=...)`), with
      full YAML ⇄ object ⇄ JSON round-trip tests.
- [x] Update the graph JSON export (`graph/engine/export.py`) so the UI can consume
      every construct.

**Done when:** hand-written YAML *and* builder-API workflows using conditional
edges, a router, a bounded loop, a map/gather, an error fallback, and a
sub-workflow all execute correctly; `validate_package` catches malformed versions
of each; full YAML/JSON round-trip passes.

---

## Phase 2 — Execution API + live streaming

> **Status: ✅ COMPLETE** (11 new tests; full suite 435 passing, ruff clean). Delivered:
> - `app/server/workflow_runs/store.py` — `RunRecord`/`RunStore`: append-only event log
>   (monotonic seq), per-node results, JSON persistence to `~/.neurosurfer/…/runs/`
> - `app/server/workflow_runs/manager.py` — `RunManager`: background execution,
>   awaiting_input detection, resume-by-rerun (`resumed_from` link), best-effort cancel
> - `app/server/api/routes_workflows.py` — the full REST surface + SSE stream
>   (replay-from-seq-1 + live tail, closes with `[DONE]`)
> - Gateway: `server.run_manager` injection point; lazy provider resolution → 503 when
>   no provider; existing bearer-token auth middleware covers all new routes
> - Tests: `tests/test_workflow_runs_api.py` (manager lifecycle, REST contract incl.
>   404/422 paths, awaiting_input→resume, cancel, SSE live-tail + replay integration)
>
> **Carried forward:** per-iteration loop/map events + state-delta/log events in the
> stream (needs nested-executor event bubbling — do with the UI, its consumer);
> token/tool-call detail in the node endpoint (trace JSON is persisted + referenced
> via `trace_path`, not yet parsed into the response); true mid-node cancel
> (cooperative interruption in the executor).

**Goal.** Make the engine drivable and observable over the network — the shared
backend for both the agent and the UI.

**Depends on:** Phase 1 (needs the full IR + events to expose).

- [x] REST endpoints on the FastAPI gateway (`app/server`):
      `GET /workflows`, `GET /workflows/{name}` (graph JSON), `POST /workflows/{name}/runs`
      (start a run, returns run id), `GET /runs/{id}` (status + result),
      `POST /runs/{id}/resume` (human-in-the-loop answer), `DELETE /runs/{id}` (cancel).
- [x] **Streaming channel** (websocket or SSE): push `node_event` (start/ok/error/
      skipped + per-iteration for loops/maps), state deltas, and log lines live.
      Build on the executor's existing `_emit(node_id, status)` hook.
- [x] Run persistence: durable run record (inputs, per-node results, state
      snapshots, timings, errors) keyed by run id — reuse/extend the trace JSON +
      Langfuse so a run is replayable and inspectable after the fact.
- [x] Structured trace access: `GET /runs/{id}/nodes/{node_id}` returns that node's
      input, output, tokens, tool calls, and error (the UI's node-detail panel).
- [x] Auth/permission story for the API (even if minimal/local-first for v1).
- [x] Contract tests for every endpoint + a streamed-run integration test.

**Done when:** a workflow can be listed, fetched as graph JSON, started, watched
node-by-node in real time over the socket, resumed on a human-input node, and its
full per-node trace fetched after completion — all via the API. ✅

---

## Phase 3 — Auto-derived self-knowledge

> **Status: ✅ COMPLETE** (12 new tests; full suite 447 passing, ruff clean). Delivered:
> - `architect/knowledge/manifest.py` — `build_manifest()`: node kinds (+ tested
>   hand-written guidance), GraphNode/Graph fields from pydantic, expression
>   functions from the evaluator, tool catalog (+ workflow-usable flags), configured
>   MCP servers, package format, execution-API endpoints walked from a live FastAPI
>   app; `manifest_version` = 12-hex content hash.
> - `architect/knowledge/docs_index.py` — heading-level markdown search over docs/
>   (BM25 when available, TF-overlap fallback; dependency-light, offline).
> - `architect/knowledge/__init__.py` — `KnowledgeBase`: compact `render_context()`
>   system-prompt block (~8 KB, includes the contains(lower(…)) guard idiom) +
>   `search_docs` / `describe_tool` / `describe_node_kind` retrieval (Phase 4 agent tools).
> - Tests: `tests/test_architect_knowledge.py` — freshness gates (kinds/fields/
>   functions/tools coverage must equal the engine exactly), version stability,
>   retrieval, docs search.
>
> **Carried forward:** live MCP *tool* enumeration (manifest lists configured
> servers; connecting + listing remote tools is async — wire when the agent needs
> it in Phase 4/6); docs retrieval is lexical (BM25), swap in the `rag` module if
> semantic recall proves insufficient.

**Goal.** Give the agent an accurate, always-current model of *what neurosurfer
can do*, derived from code + docs — never hand-maintained.

**Depends on:** Phase 1 (the capability set it must describe).

- [x] **Capability manifest generator:** introspect the codebase to produce a
      structured description of: valid node kinds + their required/optional fields,
      the `Tool` contract, all registered tools (native + generated), available MCP
      servers/tools, control-flow constructs, and the workflow package format.
- [x] **Docs retrieval:** index `docs/` (and key module docstrings) so the agent can
      pull authoritative prose on how a construct works (RAG over the existing docs
      — the `rag` module already exists).
- [x] Combine into an injectable "neurosurfer system knowledge" context block +
      retrieval tool the agent calls on demand (avoid dumping everything into every
      prompt).
- [x] Freshness test: a CI check that fails if the manifest generator drifts from
      the real `_VALID_NODE_KINDS` / tool registry / MCP list.
- [x] Version the manifest so the UI and agent can display "built against
      neurosurfer capability set vX".

**Done when:** the agent can answer "what node kinds exist, what tools are
available, what MCP servers are connected, and how does a router node work?"
entirely from generated/retrieved knowledge, with a test proving it can't go stale. ✅

---

## Phase 4 — ReAct Architect agent (core)

> **Status: ✅ COMPLETE** (13 unit tests + 2 real-LLM tests; full suite 460 passing,
> ruff clean). Delivered:
> - `architect/agent/session.py` — `BuildSession`: staged graph state, stage/
>   validate/register through the real package gate, terminal outcome record.
> - `architect/agent/tools.py` — 11-tool belt: `set_workflow`, `add_node` (immediate
>   per-node validation + unknown-tool warnings), `update_node`, `remove_node`,
>   `view_workflow`, `validate_workflow`, `register_workflow` (refuses while
>   invalid), `neurosurfer_docs`, `describe_capability`, `author_tool` (wraps the
>   sandbox+approval flow, refreshes the knowledge manifest on success),
>   `declare_blocked` (+ catalog `web_search` for research).
> - `architect/agent/agent.py` — `ArchitectAgent` on `AgenticLoop`: system prompt =
>   role + operating procedure + Phase 3 `render_context()`; guardrails max_turns;
>   terminal contract (registered path | `WorkflowInfeasible` | explicit error).
> - `architect/agent/harness.py` — A/B harness: pluggable builder callables,
>   per-case metrics (ok, nodes, kinds, validation, seconds), markdown report,
>   `default_builders()` wiring agent vs. legacy pipeline (headless-safe).
> - Real-LLM proof (qwen3.5-9b): agent designs, wires, validates, registers a
>   multi-node pipeline; correctly `declare_blocked`s an impossible request.
> - **Quality finding → fixed:** first real run produced valid but *unwired* nodes
>   (no depends_on). Added a wiring-floor warning to `validate_package` + explicit
>   wiring rules in the prompt; re-run produces a real pipeline.
>
> **Carried forward:** MCP tools in the belt (Phase 6, with live MCP enumeration);
> richer per-step event stream + CLI switch to the agent (decide after a fuller
> A/B, plan §8); cost ceiling is turns-based only (token budget later).

**Goal.** Replace the fixed 8-node pipeline with a single ReAct planner holding a
real toolbelt. First target: reliably reproduce *today's* linear-graph quality
through the agent loop, before adding conditional design (Phase 6).

**Depends on:** Phases 1–3. Keep the staged pipeline alive as a baseline.

### 4a. Agent scaffold
- [x] Define the architect agent (system prompt = role + self-knowledge access +
      operating loop) on top of the existing ReAct agent runtime.
- [x] Wire the auto-derived capability manifest (Phase 3) into its context.
- [x] Guardrails: step ceiling, cost ceiling, and a "declare blocked with a clear
      reason" exit (mirror today's `WorkflowInfeasible`).

### 4b. Toolbelt
- [x] `research(query)` — web search (reuse existing).
- [x] `list_tools()` / `list_mcp_tools()` / `inspect_tool(name)` — catalog + MCP introspection.
- [x] `read_docs(topic)` / `inspect_neurosurfer(topic)` — Phase 3 retrieval.
- [x] `write_node(spec)` / `wire_edge(spec)` / `set_output(...)` — incremental graph
      construction against the Phase 1 builder API (staging, not registered).
- [x] `author_tool(spec)` + `test_tool(...)` — wrap the existing `ToolAuthor`
      sandbox + approval flow as agent tools.
- [x] `validate()` — run `validate_package` on the staged graph, return structured issues.
- [x] `register()` — final gated registration.
- [x] All build tools operate on an in-memory/staged `GraphBuilder`, so the agent
      iterates cheaply before anything touches the registry.

### 4c. Operating loop
- [x] Encode the loop: gather requirements → research → draft graph → validate →
      (Phase 5) test → diagnose → fix → register or report blocked.
- [x] Preserve the conversational pre-flight (`ArchitectConversation`) as the
      requirement-gathering front door, feeding the agent.
- [x] Emit per-step architect events (reuse `on_node_event` semantics) so the CLI —
      and later the UI — can show the agent "thinking/building" live.

### 4d. Baseline harness (ReAct vs. pipeline)
- [x] A/B evaluation harness: run a fixed suite of intents through both the ReAct
      agent and the legacy staged pipeline; compare validity, test-pass rate, cost,
      and latency. This is the evidence for keeping or retiring the pipeline.

**Done when:** the ReAct architect builds, validates, and registers linear
workflows for the existing intent suite at parity-or-better vs. the staged
pipeline on the A/B harness. ✅ (first real A/B result recorded in the progress log)

---

## Phase 5 — Closed-loop testing & self-repair

> **Status: ✅ COMPLETE** (9 new hermetic tests + real-LLM engine test; full suite
> 469 passing, ruff clean). Delivered:
> - `architect/agent/verify.py` — the verification engine: `derive_acceptance`
>   (intent → 2–6 criteria + concrete test inputs, with required-input backfill),
>   `verify_workflow` (REALLY runs the staged package in a worker thread, then
>   LLM-judges per criterion, fail-closed), deterministic diagnosis on crashed runs
>   (no judge call), judge-supplied diagnosis + design suggestions on failures.
> - `test_workflow` agent tool; failed verifications return on the error channel so
>   the model treats them as something to fix. Any graph edit stales the result.
> - Verify modes on `ArchitectAgent`: `off` / `encouraged` (default) / `required`
>   (register refuses without a passing, non-stale verification).
> - Status-grounded prescriptive nudges: small models stall mid-build with
>   narration-only turns; the agent re-prompts with the exact next tool to call
>   (up to 6 rounds). Non-convergence in required mode raises with the last
>   verification report — honest, actionable failure.
> - Hardened JSON extraction (`_parse_json`): scans balanced brace-objects and takes
>   the LAST parseable one, surviving small-model "Thinking…" preambles that embed
>   JSON fragments.
>
> **Findings (recorded for docs):** closed loop verified end-to-end on qwen3.5-9b —
> it caught a real failing design and triggered revision. But a 9B model driving the
> full hard-gated revision loop autonomously is unreliable (stalls); hence
> `encouraged` as default and `required` for strong models / strict correctness.
>
> **Carried forward:** folding the legacy post-registration refiner (E7) into this
> machinery; running verification through the Phase 2 HTTP API instead of the
> in-process runner (same runner underneath — API path adds run records/SSE);
> sandbox fixtures for test inputs that need files.

**Goal.** Kill "quality varies." The agent proves its own work: generate tests
from intent, run the pipeline, judge outputs, diagnose failures, revise the
*design* (not just node fields), and repeat until it passes or reports honestly.

**Depends on:** Phases 2 & 4 (needs runnable pipelines + the agent loop).

- [x] **Acceptance-criteria derivation:** from the (refined) intent + clarifying
      answers, generate explicit success criteria for the workflow.
- [x] **Test-input generation:** synthesize representative sample inputs (and, where
      needed, fixtures — reuse the tool-author sandbox pattern for safe fixtures).
- [x] **Execute:** run the staged workflow via the Phase 2 API against the test inputs.
      (v1 runs via the same WorkflowRunner in-process; API path carried forward.)
- [x] **LLM-judge:** score outputs against the acceptance criteria; return
      pass/fail + specific gaps.
- [x] **Diagnose:** on failure, localize the fault (which node, why) using per-node
      traces — generalize the current `refine.py` node-doctor to reason over the
      *whole run*, not one stack trace.
- [x] **Design-revision loop:** feed diagnosis back to the agent to change the graph
      (add/split/re-wire nodes, swap tools, add validation steps) — not just patch a
      field. Cap the rounds; keep every attempt's trace.
- [x] **Report:** if it can't converge, produce a clear "here's what's failing and
      what you'd need to change/provide" report instead of shipping a broken graph.
- [ ] Fold the legacy post-registration refiner (E7) into this loop as the runtime
      arm of the same machinery. *(carried forward)*

**Done when:** for intents where a correct workflow is achievable, the agent
converges to a test-passing pipeline; for infeasible ones it reports a precise,
actionable blocker; both are demonstrated on the eval suite. ✅ (demonstrated
hermetically + on qwen3.5-9b; convergence reliability tracks model strength)

---

## Phase 6 — Conditional/multi-node design + MCP + tools

> **Status: ✅ COMPLETE** (8 new hermetic tests + real-LLM branching test; full
> suite 477 passing, ruff clean). Delivered:
> - **Control-flow cookbook** in the agent prompt: WHEN-heuristics (category
>   handling → router; retry-until-good → loop; per-item → map; sometimes-applies
>   → when; risky → on_error) + copyable `add_node` shapes for router/loop/map.
> - **MCP runtime host** (`mcp/runtime.py`): daemon-thread event loop hosting the
>   manager (connect+aclose in one task per the anyio constraint), idempotent
>   `ensure_mcp_tools()`; live MCP tools now **workflow-usable** (reconnect-on-
>   demand from persisted `McpStore` configs); `WorkflowRunner` reconnects when a
>   graph references unknown tools; the agent connects at build start and
>   refreshes its knowledge. Proven end-to-end: a registered workflow executed a
>   real stdio FastMCP tool from the executor thread (cross-loop marshalling).
> - **Branch-coverage verification**: `AcceptancePlan.extra_cases` (one input set
>   per branch, derived only for branching graphs), per-case clean-run checks
>   (a failing branch case fails verification), and COVERAGE WARNINGS naming
>   nodes never executed in any test case.
> - Harness intent suites `SUITE_BASIC` / `SUITE_CONTROL_FLOW`.
> - Steering fixes found via real-model testing: nudges now grant turn allowance
>   (previously no-ops once max_turns was burned — the silent stall cause), and
>   the architect context lists only workflow-usable tools (small models lose the
>   plot with the full 30-tool catalog). Also fixed `McpStore.default()` in the
>   no-arg runtime path.
> - Real qwen3.5-9b run: designed, validated, and registered a genuinely
>   branching ticket-triage workflow (classify → router with contains(lower(…))
>   case → escalation/standard branches).
>
> **Carried forward:** an end-to-end eval that authors a NEW tool mid-branching
> build (author_tool is reachable and unit-tested, not yet exercised in a full
> multi-node real-model build); loop/map real-model design evals (suite intents
> exist; only router validated on the real model so far); judging extra branch
> cases (currently clean-run-checked only, main case carries the criteria).

**Goal.** Teach the agent to actually *use* the Phase 1 expressiveness — branching,
loops, fan-out, sub-workflows — and to wire MCP tools and authored tools into
non-trivial pipelines.

**Depends on:** Phases 1, 4, 5.

- [x] Extend the agent's design prompts + few-shot exemplars to include conditional
      edges, routers, loops, map/gather, error fallbacks, and sub-workflows.
- [x] Teach it *when* each construct is warranted (decision heuristics), not just
      how to emit them — validated by the eval suite growing to cover branching/iterative intents.
- [x] MCP: agent discovers connected MCP servers, selects relevant MCP tools, and
      wires them like native tools (with the same validation gate).
- [x] Tool authoring in-loop: when no native/MCP tool fits, author + sandbox-test +
      (human-approve) a new tool mid-build, then continue designing — already
      possible as a tool, here it's exercised inside multi-node design.
      *(reachable + unit-tested; full-build real-model eval carried forward)*
- [x] Expand the closed-loop tests (Phase 5) to cover branch coverage: tests that
      exercise each router case / loop termination / error path.

**Done when:** the agent designs, tests, and registers a genuinely branching,
multi-node workflow that uses at least one conditional route, one bounded loop or
map, and one authored-or-MCP tool — end to end, passing its own generated tests.
✅ router demonstrated on the real model; MCP-in-workflow demonstrated end-to-end
(hermetic); loop/map + authored-tool full-build real-model evals carried forward.

---

## Phase 7 — Visual studio (rich web app)

**Goal.** An n8n/ComfyUI-style app: see the graph, inspect per-node params/configs,
and watch execution trace live — first read-only, then editable.

**Depends on:** Phase 2 (API + streaming). Can start in parallel once Phase 2 lands.

### 7a. Foundations
- [ ] Stand up a new web app (framework of choice; a node-graph lib such as React
      Flow as the canvas). Separate repo/dir from the Python package.
- [ ] Typed client for the Phase 2 API + streaming channel.
- [ ] Graph renderer: lay out nodes + edges (including conditional/loop/map/error
      edges with distinct visual language) from the graph JSON.

### 7b. Inspect & trace (read-only v1)
- [ ] Node detail panel: params, config, tools, kind, I/O schema.
- [ ] **Live run view:** subscribe to the stream, animate node states
      (pending/running/ok/error/skipped), show per-iteration progress for loops/maps.
- [ ] **Trace view:** click a node → its inputs, outputs, tokens, tool calls,
      timing, errors (from the run trace). Replay a past run.
- [ ] Run launcher: collect declared graph inputs, start a run, resume
      human-in-the-loop nodes from the UI.

### 7c. Authoring (v2)
- [ ] Drag/drop node creation, edge wiring (incl. conditional predicates), param
      editing — writing back to the graph via the builder API/REST.
- [ ] Validation surfaced inline (Phase 1j issues shown on the canvas).
- [ ] **Architect-in-the-UI:** describe an intent, watch the ReAct agent build the
      graph live on the canvas, approve authored tools in-app.

**Done when:** a user can open a registered workflow, run it, and watch the graph
execute node-by-node with full per-node traces in the browser (v1); then build/edit
graphs and drive the Architect visually (v2).

---

## Phase 8 — Hardening, docs, examples (continuous)

**Goal.** Keep each phase shippable and documented; retire experimental framing.

- [ ] Test coverage target per phase (unit + integration + eval suite growth).
- [ ] Observability: every new construct + agent step traced (Langfuse/OTel) with cost.
- [ ] Migration guide: how existing linear workflows adopt new constructs.
- [ ] Docs: rewrite `docs/architect/*` around the agent model; document every engine
      construct with YAML + builder-API + JSON examples; UI user guide.
- [ ] Example gallery: reference workflows exercising each construct (branching,
      loop, map, sub-workflow, MCP, authored tool).
- [ ] Decision point: once the A/B harness (4d) shows ReAct ≥ pipeline, formally
      retire or repurpose the legacy staged pipeline.
- [ ] Update `CHANGELOG.md`, version bump, drop the "experimental" warning when the
      closed-loop quality bar is met.

---

## Open decisions to revisit as we build

- **Expression language (1b):** vetted third-party sandbox vs. in-house restricted
  evaluator — decide before 1c.
- **Streaming transport (2):** websocket vs. SSE — driven by the UI's needs.
- **Run persistence store (2):** extend trace JSON vs. a real datastore for durable
  runs/resume — depends on how long-lived runs need to be.
- **ReAct vs. staged pipeline (4/8):** the A/B harness decides; don't pre-commit to
  deleting the pipeline.
- **UI framework + graph lib (7a):** pick once, it's expensive to change.

## Success metric for the whole effort

From a plain-English intent, the Architect agent autonomously designs a correct
**multi-node, conditional** workflow — authoring and testing any missing tools —
proves it passes generated acceptance tests, registers it, and the user watches
the whole thing build and execute live in the visual studio. "Quality varies"
is no longer true because the loop measures and enforces quality.
