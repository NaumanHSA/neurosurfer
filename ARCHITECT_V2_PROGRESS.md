# Architect V2 — Delivered Features Log

> Running log of what each phase actually shipped, written so it can be lifted
> straight into the README, `CHANGELOG.md`, a docs "What's New" page, or release
> notes later. Newest phase at the bottom. See [ARCHITECT_V2_PLAN.md](ARCHITECT_V2_PLAN.md)
> for the full roadmap and TODOs.

**⚠️ Docs still TODO** (batch these once a few phases land):
- [ ] README — mention the new control-flow engine (conditional/router/loop/map/etc.)
- [ ] `CHANGELOG.md` — add entries per phase from this log
- [ ] `docs/` — a "What's New in the Engine" page + reference for each construct
      (YAML + `GraphBuilder` examples) + example gallery
- [ ] Drop the "experimental" framing on the relevant docs once Phase 5 quality bar is met

---

## Phase 1 — Rich graph engine: typed state + control flow ✅

**Shipped:** the graph engine now supports conditional branching, iteration,
fan-out, composition, error handling, and human-in-the-loop — authorable in YAML
**or** via a fluent Python builder, and JSON-round-trippable for tooling/UI.
Fully backward compatible: existing linear `base`/`react`/`function`/`tool`
workflows run unchanged.

### New node kinds
| Kind | What it does |
|---|---|
| `router` | Selects one downstream branch. Expression router (first matching `when` wins → `default`) or LLM router (model picks a labelled route). Non-selected branches are pruned. |
| `loop` | Repeats a nested `body` sub-graph until `break_when` (expression) or `max_iterations` (a required hard ceiling). Optional `accumulate` collects each iteration's output. |
| `map` | Runs a nested `body` once per item of an `over` collection, with a `concurrency` cap. Output is the gathered list. |
| `subgraph` | Runs a nested `body` sub-graph once — workflow composition. |
| `input` | Human-in-the-loop: resolves from a pre-supplied value (resume path), an interactive IO ask, or reports "awaiting input". |

### New node fields (on any node)
- `when:` — conditional-edge guard; the node runs only if the expression is truthy.
  Pruned (not-taken) branches use **OR-join** semantics: a join node still runs if
  *any* incoming branch is live (distinct from an error-skip).
- `writes:` — store this node's output under `state.vars.<name>` for stable reference.
- `on_error:` — reroute to a fallback node on failure (error text exposed as
  `state.vars.<id>__error`) instead of failing the branch.
- `policy.retries` — re-run a flaky node up to N times before it counts as failed.

### Typed workflow state + expressions
- **`WorkflowState`** threads `inputs` / `nodes` / `vars` (+ per-iteration scope)
  through a run, with a JSON snapshot for tracing and the UI.
- **Safe expression evaluator** (`evaluate` / `safe_bool`) powers all predicates
  (`when`, router `cases`, `break_when`, `over`). Restricted AST — no `eval`, no
  imports, no dunder access, no arbitrary calls — so predicates can't execute code.
  Reads like: `nodes.classify.label == "urgent" and inputs.count > 5`.

### Programmatic builder
- **`GraphBuilder`** — fluent API (`.base()/.react()/.router()/.loop()/.map()/…`)
  that produces the **same validated IR** as YAML (proven by parity tests). Graphs
  serialize to JSON and reload unchanged — the contract the API (Phase 2) and UI
  (Phase 7) depend on.

### Example (YAML)
```yaml
nodes:
  - id: classify
    kind: base
    purpose: "Classify urgency of: {text}"
    writes: label
  - id: route
    kind: router
    depends_on: [classify]
    cases:
      - when: "nodes.classify == 'urgent'"
        to: page
    default: queue
  - id: page
    kind: base
    depends_on: [route]
  - id: queue
    kind: base
    depends_on: [route]
```

### Files
- Added: `engine/state.py`, `engine/expressions.py`, `engine/builder.py`
- Extended: `engine/schema.py`, `engine/executor.py`, `engine/loader.py`, `engine/__init__.py`, `graph/__init__.py`
- Tests: `test_graph_state_expressions.py`, `test_graph_control_flow.py`,
  `test_graph_iteration.py`, `test_graph_error_routing.py`,
  `test_graph_subgraph_input.py`, `test_graph_builder_roundtrip.py`

### Verification
64 new tests · full suite **424 passed / 0 failures** · ruff clean · new modules mypy-clean.

### Carried forward (delivered the core, these refinements deferred)
- **Per-iteration node events** (1e): loop/map bodies run in nested executors; their
  events don't yet bubble to the parent `node_event` callback. Wire this when Phase 2
  builds the streaming channel (that's the consumer that needs it).
- **Configurable gather/reduce** (1f): `map` returns the gathered list and branches use
  OR-join — the common cases. A dedicated `gather` node with list/dict/reduce strategies
  is not built yet.
- **Retry backoff** (1g): retries are immediate; no backoff delay yet.
- **Registered-package sub-workflows** (1h): `subgraph` runs an **inline** body. Calling a
  *registered* workflow package by name (+ cross-package cycle/recursion guard) belongs to
  the workflow layer and is deferred to a later increment.

---

## Phase 2 — Execution API + live streaming ✅

**Shipped:** workflows are now drivable and observable over HTTP — the shared
backend for the Architect agent (Phase 4+) and the visual studio (Phase 7). Runs
execute in the background, stream node-by-node over SSE, persist to disk for
replay, and support human-in-the-loop resume.

### REST + SSE surface (on the existing FastAPI gateway)
| Endpoint | What it does |
|---|---|
| `GET /v1/workflows` | List registered workflows (name, description, version, tags). |
| `GET /v1/workflows/{name}` | Full graph JSON — nodes, edges, control flow — for rendering/inspection. |
| `POST /v1/workflows/{name}/runs` | Start a run (`{"inputs": {…}}`) → `202` + run record with id. |
| `GET /v1/runs` / `GET /v1/runs/{id}` | Run summaries / full record (`?events=true` for the event log). |
| `GET /v1/runs/{id}/events` | **SSE live stream**: replays the event log from the start, then tails live events; closes with `[DONE]` at a terminal status. |
| `GET /v1/runs/{id}/nodes/{node_id}` | One node's status, output, error, skip reason, duration. |
| `POST /v1/runs/{id}/resume` | Human-in-the-loop: re-runs with original inputs + supplied `values` (new run linked via `resumed_from`). |
| `DELETE /v1/runs/{id}` | Best-effort cancel. |

### Semantics worth knowing
- **`awaiting_input` status**: a run that hits an `input` node with no value finishes
  as `awaiting_input` (not `failed`) with an `input_required` event naming the node —
  the client resumes with the value.
- **Durable records**: every run persists as JSON under `~/.neurosurfer/artifacts/runs/`
  (events, per-node results, final outputs, `trace_path` to the full trace).
- **Late subscribers miss nothing**: events carry a monotonic `seq`; the SSE stream
  always replays from seq 1 before tailing.
- **Auth**: the gateway's existing bearer-token middleware covers all new routes.
- **Injection point**: set `server.run_manager = RunManager(provider, registry=…, store=…)`
  to embed; otherwise one is built lazily from the configured provider (503 if none).

### Files
- Added: `app/server/workflow_runs/{store,manager,__init__}.py`,
  `app/server/api/routes_workflows.py`, `config/paths.py:runs_dir()`
- Extended: `app/server/api/router.py`, `app/server/gateway.py`
- Tests: `tests/test_workflow_runs_api.py`

### Verification
11 new tests (manager lifecycle, REST contract incl. 404/422, awaiting_input→resume,
cancel race-free via slow workflow, SSE live-tail + replay) · full suite
**435 passed / 0 failures** · ruff clean.

### Carried forward
- Per-iteration loop/map events + state-delta/log events in the SSE stream (needs
  nested-executor event bubbling; build alongside the UI that consumes it).
- Token/tool-call detail in the node endpoint (the trace JSON is persisted and
  referenced via `trace_path`, not yet parsed into the response).
- True mid-node cancel (cooperative interruption inside the executor).

---

## Real-LLM validation of Phases 1+2 ✅

**Shipped:** `tests/test_graph_llm_integration.py` — integration tests against a
real local model (LM Studio, `qwen/qwen3.5-9b` at `localhost:1234/v1`) covering
the paths where real model output drives control flow, which scripted-provider
unit tests can't prove. Auto-skips when LM Studio/the model isn't up, so CI stays
hermetic. Run explicitly with:
`conda run -n LLMs python -m pytest tests/test_graph_llm_integration.py -v`

Verified with the real model (7/7 passing, ~30 s):
- `when` guards evaluating real free-text LLM output — **both** branch directions
  (urgent → escalate, normal → archive).
- LLM router routing semantically in both directions, non-selected branch pruned.
- `map` body with a real `base` node (capitals → countries, order preserved).
- `loop` body with a real `base` node (2 iterations, accumulated).
- Phase 2 API end-to-end with the real provider: start → SSE live tail →
  succeeded + correct branch/skip events → per-node detail shows the route decision.

**Documented finding:** real model output arrives with whitespace/formatting noise
(e.g. `"\n\nurgent"`), so exact-equality guards (`nodes.classify == 'urgent'`) are
brittle — `contains(lower(nodes.classify), 'urgent')` is the robust pattern. The
LLM-router path strips/normalizes internally, so router label matching is not
affected. Docs should teach the `contains(lower(...))` idiom for guards over raw
LLM text (or structured outputs for exact matching).

---

## Phase 3 — Auto-derived self-knowledge ✅

**Shipped:** the Architect agent's "knows neurosurfer" layer — a versioned
capability manifest introspected from live code (never hand-maintained lists),
plus lightweight docs retrieval, behind one `KnowledgeBase` facade.

### What it derives (from code, at call time)
- **Node kinds** — from `_VALID_NODE_KINDS`, each with usage guidance (what it's
  for, key fields, requirements). Guidance coverage is *tested* against the engine:
  adding a kind without documenting it fails CI.
- **Node/Graph fields** — auto-extracted from the pydantic models (type, default,
  description, the `as` alias).
- **Expression language** — allowed functions pulled from the evaluator itself,
  namespaces, and the `contains(lower(…))` robustness idiom from the real-LLM tests.
- **Tool catalog** — every registered tool (native + generated) with description,
  input fields, and a workflow-usable flag; full input JSON schema via `describe_tool`.
- **MCP** — configured servers from `McpStore` (no connection needed).
- **Workflow package format** + **execution API** — endpoints enumerated by walking
  a live FastAPI app's routes (handles lazy `_IncludedRouter` nesting).

### The facade (`neurosurfer.architect.knowledge.KnowledgeBase`)
- `render_context()` — ~8 KB markdown block for the agent's system prompt.
- `search_docs(query)` — heading-level BM25 (or TF fallback) search over `docs/`.
- `describe_tool(name)` / `describe_node_kind(kind)` — on-demand detail
  (these become agent tools in Phase 4).
- `manifest` / `version` / `refresh()` — `manifest_version` is a 12-hex content
  hash: any capability change ⇒ new version ("built against capability set vX").

### Files
- Added: `architect/knowledge/{__init__,manifest,docs_index}.py`
- Tests: `tests/test_architect_knowledge.py` (freshness gates: kinds/fields/
  functions/tools must equal the engine exactly; version stability; retrieval)

### Verification
12 new tests · full suite **447 passed / 0 failures** · ruff clean.

### Carried forward
- Live MCP *tool* enumeration (needs an async connect; wire in Phase 4/6 when the
  agent consumes it).
- Docs retrieval is lexical (BM25) — swap in the `rag` module if semantic recall
  proves insufficient for the agent.

---

## Phase 4 — ReAct Architect agent ✅

**Shipped:** `ArchitectAgent` — a single ReAct planner with an 11-tool belt that
designs, builds, validates, and registers workflows, replacing the fixed 8-node
pipeline (which stays alive as the A/B baseline). Usable today:

```python
from neurosurfer.architect import ArchitectAgent
path = await ArchitectAgent(provider).build("Summarise an article and write a title")
```

### The toolbelt (all operating on one staged BuildSession)
`set_workflow` · `add_node` (immediate per-node validation + unknown-tool warnings)
· `update_node` · `remove_node` · `view_workflow` · `validate_workflow` (full
package gate) · `register_workflow` (refuses while invalid) · `neurosurfer_docs` ·
`describe_capability` · `author_tool` (sandbox + human approval, then refreshes the
capability manifest) · `declare_blocked` · plus catalog `web_search`.

### Terminal contract
Registered path returned, or `WorkflowInfeasible` raised on `declare_blocked`,
or an explicit error when the agent stalls — never a silent partial build.

### Real-LLM proof (qwen3.5-9b, LM Studio) — `tests/test_architect_agent_llm.py`
- Builds a wired multi-node pipeline for a summarise+title intent (validates clean),
  stable across repeated runs.
- Correctly `declare_blocked`s an impossible request (prod Oracle, no credentials).
- **Quality finding → fixed:** the first real run registered *valid but unwired*
  nodes (no `depends_on` — a parallel bag, not a pipeline). Two fixes: a
  wiring-floor warning in `validate_package` (multi-node graph with zero edges)
  and explicit "WIRING IS MANDATORY" rules in the agent prompt. Post-fix runs wire
  correctly. (Phase 5's closed loop is the systematic answer to this class of gap.)

### First real A/B (one intent, qwen3.5-9b — directional, not conclusive)
| builder | ok | nodes | valid | seconds |
|---|---|---|---|---|
| react_agent | ✅ | 2 | ✅ | 29.4 |
| legacy_pipeline | ✅ | 4 | ✅ | 106.1 |

Read: the agent is ~3.6× faster and leaner; the legacy pipeline decomposes more
deeply (adds input-validation/format steps its prompts force). Decision on
retiring the pipeline waits for a fuller suite (plan §8) — harness:
`neurosurfer.architect.agent.run_harness` / `render_report`.

### Files
- Added: `architect/agent/{__init__,session,tools,agent,harness}.py`
- Extended: `architect/__init__.py` (exports `ArchitectAgent`),
  `graph/workflow/validate.py` (wiring-floor warning)
- Tests: `tests/test_architect_agent.py` (13: toolbelt units, scripted-provider
  agent loop incl. blocked + stall paths, harness), `tests/test_architect_agent_llm.py`
  (2, real model, auto-skip)

### Verification
Full hermetic suite **460 passed / 0 failures** · ruff clean · real-LLM 2/2 twice.

### Carried forward
- MCP tools in the belt + live MCP enumeration (Phase 6).
- CLI still drives the legacy pipeline — switch after a fuller A/B (plan §8).
- Cost ceiling is turns-based; add a token budget later.
- Richer per-step architect event stream for the UI (currently notify strings).

---

## Phase 5 — Closed-loop testing & self-repair ✅

**Shipped:** the Architect now proves its own work. Before finishing, it can
derive acceptance criteria and realistic test inputs from the user's intent,
**actually run** the staged workflow on them, judge every criterion with an LLM
judge, and feed failures — with a diagnosis and suggested design changes — back
into its own graph-editing loop. "Quality varies" becomes "quality is measured".

### The engine (`architect/agent/verify.py`)
- `derive_acceptance(provider, intent, graph_yaml, declared_inputs=)` — one LLM
  call → 2–6 intent-specific criteria + concrete test inputs. Required inputs the
  model missed are backfilled with typed placeholders so a run can always start;
  garbage model output degrades to a single fulfils-the-intent criterion.
- `verify_workflow(...)` — loads the staged package, runs it in a worker thread
  (`asyncio.to_thread`), then:
  - **crashed run** → deterministic diagnosis from node errors, judge skipped;
  - **clean run** → per-criterion LLM verdicts, **fail-closed** (an unruled
    criterion counts as failed), plus diagnosis + design suggestions on failure.

### Agent integration
- New `test_workflow` tool: validates structure first, derives/caches the plan,
  runs + judges, and returns FAILED reports on the **error channel** so the model
  treats them as work to do. Every graph edit (add/update/remove/set_workflow)
  stales the verification.
- `ArchitectAgent(verify=...)`: `off` | `encouraged` (default) | `required` —
  in required mode `register_workflow` refuses without a passing, non-stale
  verification, and non-convergence raises with the full last report.
- **Prescriptive nudges:** small models stall mid-build with narration-only turns
  (the loop reads that as "done"). The agent now re-prompts the same conversation
  with current status + the exact next tool to call, up to 6 rounds.

### Real-model findings (qwen3.5-9b, recorded for docs)
- The closed loop works end-to-end: in a live run it **caught a failing design**
  (judge failed the blurb workflow) and the agent responded by adding a node —
  design revision, not field patching.
- A 9B model driving the *hard-gated* revision loop to convergence is unreliable
  (stalls after revisions) — hence `encouraged` as the default and `required`
  recommended for stronger models. This is the model-strength dependency the
  plan predicted, now quantified.
- Small models emit JSON answers after a "Thinking Process:" preamble that itself
  contains `{...}` fragments; naive regex extraction fails. `_parse_json` now
  scans balanced brace-objects (string-aware) and takes the last parseable one.

### Files
- Added: `architect/agent/verify.py`
- Extended: `architect/agent/{session,tools,agent,__init__}.py`
- Tests: `tests/test_architect_verify.py` (9: engine incl. fail-closed + crash
  paths, tool + staleness + register gating, scripted required-mode agent loop),
  `tests/test_architect_agent_llm.py` (+1 real-model engine test)

### Verification
Full hermetic suite **469 passed / 0 failures** · ruff clean · real-LLM 3/3.

### Carried forward
- Fold the legacy post-registration refiner (E7) into this machinery.
- Run verification through the Phase 2 HTTP API (adds run records/SSE visibility).
- Sandbox fixtures for test inputs that need files (tool-author pattern).
- A "must-test-but-needn't-pass" middle gate if it proves useful in practice.

---

## Phase 6 — Conditional/multi-node design + MCP + branch coverage ✅

**Shipped:** the agent now *uses* the Phase 1 expressiveness — and MCP servers are
first-class workflow citizens.

### Control-flow design (the cookbook)
The agent prompt gained WHEN-heuristics (different handling per category →
`router`; retry-until-good → `loop`; per-item work → `map`; sometimes-applies →
`when:`; risky step → `on_error:`; linear task → none of these) plus copyable
`add_node` shapes for router/loop/map. Validated on the real model: qwen3.5-9b
designed and registered a genuinely branching ticket-triage workflow
(classify → router with a `contains(lower(…))` case → escalation/standard
branches, both wired to the router).

### MCP tools in workflows (`mcp/runtime.py`)
- A **process-lifetime MCP host**: daemon thread + event loop running the manager;
  connect and close happen in one task (anyio requirement); `ensure_mcp_tools()`
  is sync + idempotent; `shutdown_mcp()` for tests/clean exit.
- **Policy change:** live MCP tools are now **workflow-usable** — durable because
  server configs persist in `McpStore` and the runtime reconnects on demand
  (`WorkflowRunner` auto-connects when a graph references unknown tools; the
  agent connects at build start and refreshes its capability context).
- Proven end-to-end in tests: a registered workflow with a tool-kind node backed
  by a real stdio FastMCP server executed correctly from the graph-executor
  thread (McpTool's cross-loop marshalling back to the host loop).

### Branch-coverage verification
- `AcceptancePlan.extra_cases` — for branching graphs the deriver produces one
  input set per branch (max 3); linear graphs get none.
- Each case must run cleanly (a failing branch case fails the verification);
  the main case still carries the judged criteria.
- **COVERAGE WARNING** lists nodes never executed in any test case — dead branch,
  wrong guard, or missing case — surfaced to the agent (advisory, not failing).

### Steering fixes from real-model testing
- **Nudge turn-budget bug:** once a flailing model burned `max_turns`, every nudge
  `run_collect` returned instantly (loop guard) — nudges were silent no-ops. Each
  nudge now grants a small turn allowance. This was the actual cause of the
  "stalls despite nudging" failures.
- **Focused context:** the architect's system prompt now lists only
  workflow-usable tools (first-line descriptions), not the full 30-tool catalog —
  measurably better small-model behaviour.
- `ensure_mcp_tools()` no-arg path used `McpStore()` (TypeError) → `McpStore.default()`.

### Files
- Added: `mcp/runtime.py`, `tests/test_architect_phase6.py`
- Extended: `tools/registry.py` (live tools workflow-usable),
  `graph/workflow/runner.py` (MCP reconnect), `architect/agent/agent.py`
  (cookbook, MCP at build start, nudge budget), `architect/agent/verify.py`
  (extra_cases, coverage gaps), `architect/knowledge/__init__.py`
  (`workflow_tools_only` context), `architect/agent/harness.py` (intent suites)

### Verification
8 new hermetic tests · full suite **477 passed / 0 failures** · ruff clean ·
real-LLM suite 4/4 (incl. the branching-design test).

### Carried forward
- Full-build real-model eval that authors a NEW tool mid-branching build.
- Loop/map design evals on the real model (suite intents exist).
- Judging extra branch cases (clean-run-checked only today).

---

## Fix — trace I/O propagation to node/workflow spans ✅

**Found while testing the Phase 1 tutorial in Langfuse:** clicking a `node:*` or
`workflow:*` span showed Input `null` / Output `undefined` — only the nested
`Agent.run` generation carried real I/O. The `traced_run` plumbing supported
input/output all along; the executor and runner simply never supplied them.

- `graph/engine/executor.py` — each `node:*` span now opens with
  `input={graph_inputs, dependencies}` and closes with `output=<raw_output>`
  (or `error(...)` on failure), JSON-safe via the state serializer.
- `graph/workflow/runner.py` — the `workflow:*` root span opens with the run's
  inputs and closes with the final outputs (or the joined node errors).
- `observability/exporters/base.py` — MemoryExporter's `run_finish` now records
  `span_id` so tests can match finishes to spans.
- Test: `test_workflow_node_agent_hierarchy` extended to assert workflow + node
  spans carry real I/O. Verified live: `workflow:trace_fix_demo` in Langfuse
  shows populated Input/Output at every level.

---

## Redesign — `routes` router: the router IS the classifier ✅

**User feedback while testing the tutorial:** the `cases` router read as binary
(if/else), the expression mini-language (`contains(lower(...))`) is
programmer-facing plumbing, and a separate classify node + router was redundant
for the common case.

**New primary form** (the classify node disappears entirely):
```yaml
- id: route
  kind: router
  goal: "Route this support ticket by its content: {ticket}"
  routes: {urgent: escalate, billing: finance, routine: reply}   # N-way, one node
  repair: true      # invalid model answer → one corrective retry, then default
  default: reply
```
- The router itself classifies via ONE LLM call (instructed by purpose/goal,
  interpolated with graph inputs; upstream outputs included as evidence when it
  has parents), picks a label, prunes every other target.
- `repair` (default on) retries an unmappable answer with explicit corrective
  feedback; then `default`; no default → honest error (previously the LLM-router
  fell back **silently**).
- `cases` ([{when, to}]) remains as the deterministic variant (no LLM call,
  reproducible — for routing on structured/function-node state). `routes` and
  `cases` are mutually exclusive (validated); same targets-must-depend rule.

**Touched:** engine schema/executor/loader, `GraphBuilder.router(routes=…)`,
knowledge guidance, agent cookbook (routes-first), tutorial §10/§12 rewritten
(3-way triage: escalate/finance/reply). 7 new tests (N-way, prose-tolerant match,
repair retry, default fallback, no-default error, mutual exclusion, target
wiring) — full suite **484 passed**, ruff clean.

---

## Redesign — `until` loops: plain-English stop condition + feedback ✅

**User feedback (same instinct as the routes router):** `break_when` expressions
over raw LLM text are brittle and programmer-facing; the review node shouldn't
need an output contract; the loop should decide stop/continue itself.

**New primary form:**
```yaml
- id: refine
  kind: loop
  max_iterations: 3          # ceiling stays mandatory
  until: "the review approves the slogan"    # ← plain English
  body: [...]
```
- After each iteration a **hidden exit judge** (one small LLM call) reads the
  body's outputs and answers STOP or CONTINUE with a reason.
- **CONTINUE reasons become the next iteration's `{feedback}`** (template var +
  expression scope) — directed refinement instead of blind retry.
- Unparseable judge answers get one corrective retry (reuses the router's
  `repair` flag), then fail SAFE to continue — `max_iterations` always bounds.
- Judge verdicts recorded in `structured_output["judge"]`; a `routing`-style log
  line per iteration.
- `break_when` remains the deterministic sibling (budgets, cursors, index
  checks — no LLM call). `until`/`break_when` mutually exclusive (validated).

**Touched:** engine schema/executor/loader, `GraphBuilder.loop(until=…)`,
knowledge guidance + agent cookbook (until-first), tutorial §11 (judge feedback
visibly steering redrafts). 5 new tests (stop+feedback threading, repair retry,
judge-failure fail-safe, mutual exclusion, empty-until) — full suite
**489 passed**, ruff clean.

---

## Session 2026-07-23 — Tutorial 06 (Architect) + provider-owned sampling + toolbelt fixes ✅

**Shipped this session:**

1. **New tutorial `tutorials/06_architect.ipynb`** — a compact Architect walkthrough
   (Phases 3–6): self-knowledge (`KnowledgeBase`), ONE linear build + run, closed-loop
   verification *reusing that build* (`derive_acceptance` + `verify_workflow`), and ONE
   branching build. Heavier flows (requirement-gathering `ArchitectConversation`,
   `verify="required"` self-repair, the A/B `run_harness`) are documented as copy-paste
   "Go further" snippets to keep runtime short (2 builds). Trimmed down from a fuller
   5-build draft after it proved too slow/costly on local models.
   - **Committed with outputs cleared** — see TODO: needs one clean end-to-end run to
     embed live outputs. Was validated through §4 (build + verification) on `gpt-5-mini`.

2. **Provider-owned sampling params** (`llm/types.py` + providers). `GenerationConfig`'s
   `max_tokens` / `temperature` / `effort` now default to **`None`**; the provider
   resolves them from its own capabilities at request time. Agents no longer hardcode
   these (stripped from `architect/conversation`, `architect/refine`,
   `architect/tool_author`, `architect/agent/verify`, `agents/base`,
   `agents/context/manager`, `rag/agent` router call). Rationale: gpt-5 / o-series reject
   any non-default temperature and require `max_completion_tokens` — agents shouldn't
   know per-model constraints; the Provider owns them.
   - `OpenAIProvider`: `_send_temperature=False` for `gpt-5`/`o1`/`o3`/`o4` (omit
     temperature entirely); `max_tokens` → `capabilities.max_output_tokens` when unset.
   - `AnthropicProvider`: `max_tokens` resolved from caps; `temperature`/`effort` sent
     only when explicitly set.
   - `OpenAIProvider` (official api.openai.com) now actually drives current gpt-5 models.

3. **Architect toolbelt fixes** — found by autopsying a Langfuse trace of one branching
   build (`ArchitectAgent.build 6b980132b9e3`: **266s, 37 LLM calls, 5 errors, $0.066**):
   - `GraphNode.accumulate` now coerces a **bool** → `False`→`None`, `True`→`"accumulated"`
     (models pass `accumulate: false` where a var-name string is expected). *(fixed the
     `add_node` loop crash.)*
   - New **`set_outputs`** tool + a redirect hint on `update_node` — the model had
     hallucinated a `set_workflow_outputs` *node* to declare graph outputs. Prompt step 7
     now says declare outputs with `set_outputs`. *(fixed 2 of the 5 trace errors.)*

**Verification:** full hermetic suite **489 passed**, ruff clean (real-LLM tests pinned to
`qwen/qwen3.5-9b` are excluded — that model crashes on load on this box).

**TODO — left for the next machine:**
- [x] **Run `tutorials/06_architect.ipynb` end-to-end** to embed live outputs — done on
      `gpt-4o-mini` (see next section).
- [x] **Over-engineering steer** (agent prompt) — done + a structural orphan-node warning.
- [x] **Verification cost** — done (stop extra-case re-runs at full node coverage + dedup).
- [x] Real-LLM architect tests — parametrized via `NEUROSURFER_TEST_*` (hosted path added).
- [x] README / `docs/` "What's New in the Engine" + drop "experimental" framing — done
      (example gallery deferred: redundant with tutorial 06 + the new control-flow guide).

---

## Session 2026-07-23 (continued) — trace-driven hardening + close-out of the above ✅

Machine had OpenAI `gpt-4o-mini` + a local Langfuse (docker) wired via `.env`. Every
workflow run below was traced to Langfuse and the traces were inspected for issues — the
approach that has repeatedly surfaced silent bugs.

**Bug fixes (trace/warning-driven):**

1. **Node templates now interpolate upstream `writes` vars + dep outputs** (`graph/engine/
   executor.py`). A node whose `goal`/`purpose` referenced an upstream output by name
   (`"…based on the summary: {summary}"`) silently failed — `_build_system_prompt`
   interpolated over **graph inputs only**, logged a warning, and leaked the literal
   `{summary}` into the prompt (the node only worked because deps are *also* dumped in the
   user prompt). Now interpolates over `{**graph_inputs, **dependency_results, **state.vars}`,
   consistent with loop `{feedback}`. Found via the per-run WARNING; confirmed the warning
   is gone and `{summary}` resolves.

2. **Over-engineering steer + orphan-node validation warning.** The depth-floor validation
   warning literally told the agent to *"add intermediate steps (validation, transformation,
   output formatting)"* whenever a graph had <3 LLM nodes — the mechanical cause of bloat (a
   traced branching build came out **6 nodes with an orphan `validate_ticket`**). Removed that
   warning; added an **orphan-node warning** (a node whose output is neither in `outputs` nor
   any `depends_on`, accounting for `nodes.*`/`vars.*` expression refs). Agent prompt gained
   minimalism ("fewest nodes, 2–4"), a NO-ORPHAN rule, and a "fix the smallest thing first"
   steer for failed `test_workflow`. Re-ran the same branching intent traced → **4 nodes,
   router, no orphan, zero warnings** (was 6 + orphan).

3. **Verification cost lever** (`architect/agent/verify.py`). Branch-coverage extra cases
   re-run the whole graph; now they **stop as soon as every node has been exercised** and
   **skip input sets that duplicate an already-run case** — no coverage loss.

**Tests + tooling:**

4. **Real-LLM tests parametrized** via `NEUROSURFER_TEST_MODEL` / `_BASE_URL` / `_API_KEY`
   (shared `tests/_llm_test_provider.py`; `.env` fallback). Default is unchanged (LM Studio
   qwen, auto-skip); a hosted path runs the same suites on OpenAI. Verified: **11/11 real-LLM
   tests pass on `gpt-4o-mini`** (7 graph control-flow + 4 architect builds).

5. **Tutorial 06 run end-to-end on `gpt-4o-mini`**, outputs embedded. Made the notebook
   **model-agnostic** (reads `MODEL` from `.env`, no longer hard-pins `gpt-5-mini`) and fixed
   a **tracing bug**: the Langfuse cell didn't strip quotes from `.env` values, so
   `LANGFUSE_HOST` kept its quotes and tracing silently read "disabled". Now **Tracing:
   ENABLED**, 12/12 cells clean, §4 verification PASSED, §5 produced a real router; all
   `tutorial:*` + `ArchitectAgent.build` traces landed in Langfuse.

**Docs:** README "What's new" + "in the box" now cover the control-flow engine / self-verifying
Architect; `CHANGELOG.md [Unreleased]` gained engine/API/architect/verification entries; the
[Graph & Workflows guide](../docs/guides/graph-workflows.md) gained a **Control flow** reference
(router / loop / map / `when` / `on_error` / `writes` / expressions); the "experimental" framing
on the architect docs was softened to "maturing — quality tracks the model" with a pointer to the
ReAct `ArchitectAgent`.

**Verification:** full hermetic suite **489 passed / 11 skipped**, ruff clean, real-LLM **11/11**
on `gpt-4o-mini`.

**Token/cost surfacing (done in a follow-up):**

6. **Gateway returns token usage.** The OpenAI-compatible `/v1/chat/completions` response now
   carries a spec-standard `usage {prompt_tokens, completion_tokens, total_tokens}`, and the
   streaming path emits a final usage-only chunk when the request sets
   `stream_options.include_usage` (matching OpenAI). Tokens come from the agent's `Usage`
   (real when the provider reports them, tiktoken/char-estimated for local servers), computed
   as a per-request delta on the possibly-shared agent. *Tokens only — the gateway never emits
   dollars.* (`schemas/openai.py`, `streaming/openai_chunks.py`, `backends/agent.py`; +2 tests.)

7. **Trace fidelity — system prompt captured** (fix ②). Langfuse generations now prepend the
   agent/node system prompt to the traced `input` (`input[0].role == "system"`), so
   prompt-templating issues (like fix #1) are visible in the trace, not just stdout. (`observability/
   exporters/stream.py`, `agents/base.py`.)

8. **Langfuse token→cost handoff** (fix ③). Root cause was **not** ours: the Langfuse instance's
   *managed* model rows ship with an empty `prices: {}` map (only the legacy scalar
   `inputPrice`/`outputPrice` are seeded, which the v3 cost calc ignores) — so cost stayed $0 for
   any usage format. Fixed by (a) modernising the exporter to send **both** `usage` (v2 compat)
   and `usage_details` (v3-native; verified no double-count), and (b) adding a properly-priced
   `gpt-4o-mini` model to the local Langfuse via `POST /api/public/models`
   (`inputPrice`/`outputPrice`). Verified end-to-end: `usageDetails` + `costDetails` now populate
   (`calculatedTotalCost` non-zero). Neurosurfer still computes **no dollars** — cost is the
   backend's job; we only ship tokens. *Other models need the same one-time Langfuse price entry.*

**Verification (follow-up):** full suite **491 passed / 11 skipped**, ruff clean; fixes ②/③
confirmed against live Langfuse; gateway usage covered by 2 new tests.

- Env note: dev runs use conda env **LLMs** (`langfuse<3`, `nbclient`/`nbconvert`/`ipykernel`
  installed there); `.env` `OPENAI_BASE_URL` was commented out to use real OpenAI.

---

## Remaining TODO — assessment after trace-driven hardening

The build → validate → verify core is solid and fast; closed-loop verification is the standout
(it caught a real "4 sentences, not 3" defect live). The recurring weakness is **model strength**:
on `gpt-4o-mini` the same branching intent variously came out as a clean 4-node router, a 6-node
bag-with-orphan (pre-steer), and a fully linear graph — nondeterministic. Ranked next steps:

1. **[High] Validate template-var resolution at build time.** The `{summary}` bug was silent
   until a stdout WARNING. `validate_package` checks tools/edges/outputs/orphans but NOT whether
   `{vars}` in a node's `purpose`/`goal`/`expected_result` resolve to a known graph input /
   upstream `writes` / dependency output. Add that check → catches a whole class of silent
   failures *before* a run and hands the agent an error to fix. Best-bounded robustness win (the
   resolution scope already exists in `executor._run_node`: `{**graph_inputs, **dependency_results,
   **state.vars}`).
2. **[High] Verification cost.** Even with the early-exit + dedup, a build calling `test_workflow`
   several times re-runs the whole graph many times (~half the LLM calls in a traced build). Next
   levers: judge on a single run, cache verdicts across unchanged edits, or a cheap "smoke" tier
   before the full judged run.
3. **[Med] Collapse the classify→router redundancy.** Even good branching builds add a separate
   `classify_ticket` base node feeding a router that re-classifies the same input (4 nodes where 3
   suffice). Tighten the cookbook, or detect structurally (a base node whose only consumer is a
   router re-classifying its input) and warn.
4. **[Med] Model-tier strategy.** Since quality tracks model strength, adapt automatically:
   `verify="encouraged"` + structured-output parsing on weaker models, `verify="required"` on
   strong ones — instead of the user having to know.
5. **[Low] Langfuse model prices are per-instance.** Only `gpt-4o-mini` was priced on the local
   Langfuse; other models (gpt-4o, claude, …) need the same one-time `POST /api/public/models`
   entry, or a small idempotent `scripts/` seeder. (Not a neurosurfer code concern.)
6. **[Low] Deferred observability polish (by request):** none outstanding — ②/③ done.
7. **[Docs] Architect pages still describe the legacy `ArchitectBuilder` pipeline**, not the ReAct
   `ArchitectAgent`. A full V2 rewrite of `docs/architect/*` (+ an example gallery) was deferred as
   larger than this pass; the pages now carry a pointer to the agent + tutorial 06.

**Not exercised this session** (confidence caveats): requirement-gathering `ArchitectConversation`,
mid-build `author_tool`, `verify="required"` on a *strong* model, and the A/B `run_harness`.
