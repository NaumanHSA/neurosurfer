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

## Remaining — Phase 5: Graph / workflow-executor nesting

Make a multi-node workflow render as one nested trace. Agent/sub-agent nesting is
done; this extends it to the graph executor path. Split into two levels.

### Level 1 — Workflow = one trace, node agents nested under it ⬜  *(~1 hr, low risk)*
- [ ] Mint a root `TraceContext` for the graph run; `push_trace_context` around it
      in the workflow runner / `GraphExecutor`
- [ ] Node agents already flow through `base._tap()` → they nest automatically
- [ ] Test + one live Langfuse check
- Result: `WorkflowRun` (root) → each node's agent run → its tool spans, one trace.

### Level 2 — Full `graph-run → node span → agent → tool` hierarchy ⬜  *(~half a day, more risk)*
- [ ] Per-node span (not just the agent run) — wire structured `Tracer._record_step`
      into exporters, or a lightweight node-level context
- [ ] Handle nodes that bypass `agent.run` (native nodes) and usage dropped in
      `graph/engine/node_runner.py`
- [ ] Correct nesting for **parallel branches** (concurrent nodes + contextvars + gather)
- [ ] Revive the currently-dead `NodeExecutionResult.traces` / `GraphExecutionResult.traces`
      fields (`graph/engine/schema.py`)
- [ ] Tests + real multi-node workflow verified in Langfuse
- Risk driver: the graph executor path is less uniform than the agent path.

**Recommendation:** ship Level 1 first (visible win, near-zero risk), then decide if
Level 2's per-node polish is worth the extra half-day.
