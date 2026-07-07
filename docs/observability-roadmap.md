# Observability Roadmap

Status of the pluggable trace-exporter work (Langfuse + OpenTelemetry). User-facing
usage lives in [guides/observability.md](guides/observability.md); this file tracks
what's built and what's left.

**Legend:** ✅ done · 🚧 in progress · ⬜ not started

---

## Design summary

Agent runs emit a typed event stream (`ToolStarted`, `TurnCompleted{usage}`, …).
A side-channel **observer** on that stream ([base._tap](../neurosurfer/agents/base.py))
translates it into backend-neutral `TraceExporter` calls: run → trace, LLM turn →
generation (with token cost), tool call → span. Auto-on from the environment
(`LANGFUSE_*` / `OTEL_EXPORTER_OTLP_ENDPOINT`); zero overhead when unconfigured; a bad
or unreachable exporter never breaks a run. Nesting is carried by an ambient
`TraceContext` published in a contextvar and inherited by runs that start inside
another run.

Code: `neurosurfer/observability/` (`context.py`, `exporters/`),
`neurosurfer/config/observability.py`.

---

## Phase 0 — Foundations ✅
- [x] `TraceContext` (trace/span/parent/session ids) reusing `new_run_id`
- [x] `ObservabilityConfig` env auto-detection (`detect_exporters_from_env`), off by default
- [x] `observability` extra in `pyproject.toml` (lazy imports)

## Phase 1 — Exporter interface ✅
- [x] `TraceExporter` protocol (`on_run_start/on_turn/on_tool_*/on_event/on_error/on_run_finish`)
- [x] `NullExporter` + `MemoryExporter`
- [x] Lazy, env-aware, fail-soft registry (unknown / missing SDK → warn + skip)

## Phase 2 — Event-stream observer ✅
- [x] `TraceStreamObserver` translates the event stream → exporter lifecycle
- [x] Wired into `base._tap()` beside `AgentTrace`; independent of `verbose`
- [x] Exporter exceptions swallowed — never break a run

## Phase 3 — Langfuse adapter ✅
- [x] run → trace, turn → generation (model + token usage → cost), tool → span
- [x] Env config (`LANGFUSE_*`); **verified live** in the Langfuse UI

## Phase 4 — OpenTelemetry adapter ✅
- [x] GenAI-semconv spans over OTLP; own `TracerProvider` (doesn't touch global)
- [x] Root → turn/tool child spans

## Phase 6 — Docs, tests, polish ✅
- [x] `docs/guides/observability.md` + mkdocs nav
- [x] Tests: config detection, event→lifecycle mapping, live agent-run wiring, fail-soft
- [x] CHANGELOG + `.env.example`

## Sub-agent nesting + sessions ✅
- [x] Ambient `TraceContext` in a contextvar; nested runs inherit trace/session, nest under parent span
- [x] Propagates across `await` and `asyncio.gather` (sequential + parallel sub-agents)
- [x] Langfuse + OTel exporters key state by `span_id`, parent under the enclosing run
- [x] `session_id` on agents; CLI sets one per conversation (reset on `/clear`)
- [x] Tests + **live Langfuse** nested-trace verification
- [x] `.env` loader hardening (strip trailing inline comments; keep quoted `#`)

---

## Phase 5 — Graph / workflow-executor nesting ✅  *(code complete; live check pending)*

Make a multi-node workflow render as one nested trace. Agent/sub-agent nesting was
already done; this extended it to the graph executor path in two levels — both now
built. Only a live multi-node Langfuse verification remains (needs credentials).

### Level 1 — Workflow = one trace, node agents nested under it ✅  *(~1 hr, low risk)*
- [x] Mint a root `TraceContext` for the graph run; `push_trace_context` around it
      in the workflow runner (`traced_run` in
      [observability/run.py](../neurosurfer/observability/run.py), wrapping
      `executor.run` in [graph/workflow/runner.py](../neurosurfer/graph/workflow/runner.py))
- [x] Node agents already flow through `base._tap()` → they nest automatically
- [x] Test ([tests/test_observability_workflow.py](../tests/test_observability_workflow.py):
      root span + node nesting + no-exporter no-op)
- [x] Thread propagation: parallel nodes (`parallelism>1`) and timeout nodes
      (`policy.timeout_s`) hop to `ThreadPoolExecutor` workers, and
      `run_coro_blocking` may spawn a thread — all now run inside a
      `contextvars.copy_context()` snapshot, so the ambient `TraceContext` crosses
      the thread boundary and those node agents nest too
      ([executor.py](../neurosurfer/graph/engine/executor.py),
      [node_runner.py](../neurosurfer/graph/engine/node_runner.py); tests cover both)
- [ ] One live Langfuse check
- Result: `workflow:<name>` (root) → each node's agent run → its tool spans, one trace.
- Note: nodes nest whether they run serially, in parallel, or under a timeout. The
  per-node span layer (Level 2) is now built on top of this.

### Level 2 — Full `graph-run → node span → agent → tool` hierarchy ✅  *(built)*
- [x] Per-node span via a lightweight node-level context — each node's execution in
      `GraphExecutor._execute_one` is wrapped in `traced_run("node:<id>", flush=False)`
      ([executor.py](../neurosurfer/graph/engine/executor.py)), publishing an ambient
      `TraceContext` the node's agent inherits. `traced_run` gained a `RunSpan` handle
      so a node that *returns* (not raises) an error marks its span errored
      ([observability/run.py](../neurosurfer/observability/run.py))
- [x] Nodes that bypass `agent.run` (function / tool nodes) are now **visible** — they
      get their own node span even with no agent underneath
- [x] Correct nesting for **parallel branches** — the `copy_context()` snapshot from
      Level 1 means each concurrent node grabs its own node span as parent; verified
      with a 2-node `parallelism=2` graph
- [x] Deeper nesting needs **no exporter change**: Langfuse + OTel both resolve the
      parent by `parent_span_id` and register every span, so `workflow → node → agent
      → tool` renders at arbitrary depth
- [x] Tests: workflow→node→agent hierarchy, function-node visibility, error marking,
      parallel + timeout thread nesting
      ([tests/test_observability_workflow.py](../tests/test_observability_workflow.py))
- [ ] One live multi-node Langfuse check (needs credentials — run locally)

**Out of scope of the exporter work** (separate subsystems, left as-is):
- The structured JSON `Tracer` and its dead `NodeExecutionResult.traces` /
  `GraphExecutionResult.traces` fields (`graph/engine/schema.py`) — a different
  tracing path from the pluggable exporters; not needed for backend nesting.
- Per-node token **usage aggregation** on `GraphExecutionResult` (react nodes drop it
  in `node_runner.py`). Usage is still captured *in the trace* per turn via each
  agent's `on_turn`; only the result-object rollup is missing.

**Recommendation:** ship Level 1 first (visible win, near-zero risk), then decide if
Level 2's per-node polish is worth the extra half-day.
