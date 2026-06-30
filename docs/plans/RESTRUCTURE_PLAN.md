# Restructure Plan — Neurosurfer Core, CLI & Builder as Features

> Status: **ALL PHASES COMPLETE. master-agent is RETIRED.**
> Active repo: `/home/nomi/workspace/neurosurfer`
>
> Phases done: R1–R4, F1–F7, C (de-shim), G (graph consolidation), A (agent family),
> N1 (OpenAI-compatible server), N2 (sandboxed code-exec), N3 (web-search submodule),
> Option-1 copy (code copied from master-agent → neurosurfer public repo; legacy
> package removed; 555 tests green in the new location).
>
> **F7** renamed `master_agent` → `neurosurfer` (package dir, imports, entry points,
> pyproject, env vars `NEUROSURFER_HOME`, temp prefix `ns_pyexec_`, CLI banner, README,
> Apache-2.0 license). **Option 1** cleaned the neurosurfer public repo (removed legacy
> stack), copied the new package + tests + pyproject, kept docs/tutorials/labs/LICENSE.
> **555 tests green, zero regressions.**
>
> **Next: Phase Q — REPLANNED 2026-06-30** (general agent + deep workflow building).
> The reorg is done; Phase Q is the first *feature* phase. It turns the CLI front face
> from the bare Architect into a general automation **Assistant** that escalates through
> 4 tiers (converse → light automation → plan-gated heavy one-off → Architect handoff),
> and rebuilds the Architect's shallow single-`plan` node into a deep
> `decompose → design_nodes → critique` pipeline. See **Phase Q** below.

## Vision

This repo is becoming **neurosurfer**: an LLM framework whose *core* is agents, LLMs,
RAG, tools, and tracing. The **CLI** and the **workflow builder (Architect)** are
features layered on top of that core — not the center of gravity.

Decision (confirmed by user): we mature neurosurfer in place and bring the CLI +
builder to it as features. The package will be **renamed `master_agent` → `neurosurfer`**
later (mechanical, last phase) and remapped to its own GitHub repo.

Guiding principles:
- **Core first.** Agents / LLMs / RAG / tools / tracing are top-level modules, not
  buried under `workflows/_runtime/`.
- **CLI is a delivery layer.** It imports from the core/features; nothing important
  lives *inside* `cli/`. You must be able to drive the builder from Python without
  importing `cli`.
- **The Architect is a feature module**, not a CLI detail.
- **Green tests after every step.** Use re-export shims so moves don't force a
  big-bang import rewrite.

---

## Why (the three problems that triggered this)

1. **Shallow generated workflows (Issue 1).** The `plan` node emits thin graphs
   (`identify_source_files → extract_api_metadata → generate_markdown → persist`).
   A plain "give me a YAML" prompt to any LLM would do as well. The graph adds
   ceremony without depth → planning quality must improve. *(Phase Q1.)*
2. **Immature ReAct agent (Issue 2).** Running the doc workflow with "Hi there!"
   still executed the whole pipeline instead of answering conversationally, and the
   raw `<__final_answer__>` sentinel leaked into output. The ReAct loop needs a
   production-grade redesign (conversational guard, clean final answers, stop
   conditions, streaming). *(Phase Q2.)*
3. **Bad code organization (Issue 3).** The important agentic code is buried under
   `workflows/_runtime/agents/`; the Architect's conversation lives inside `cli/`;
   there are two parallel agent stacks and two tool frameworks. *(Phase R — priority.)*

---

## Current layout (after Phase G — graph consolidation)

```
master_agent/
  # ── core primitives ──
  agents/          ← THE agent engine (Agent, loop, subagent, permissions, events,
                     messages, context_manager, structured, durable_state,
                     tasks_runtime, summary_prompt, lexical, subagent_defs)
  llm/             ← providers, types, registry, retry, tokens
  tools/           ← canonical Tool framework (Tool, builtin/, registry, generated)
  rag/             ← retrieval pipeline (rewired on native stack, F1)
  vectorstores/  embeddings/  cache/  tracing/   ← promoted/greenfield primitives
  graph/           ← the GRAPH ORCHESTRATION subsystem (one cohesive package):
    __init__.py    ←   re-exports engine primitives (from master_agent.graph import
                       Graph, GraphExecutor, …); lazy __getattr__ for workflow/builder
    engine/        ←   the DAG engine (executor, schema, errors, loader, manager,
                       artifacts, export, utils, templates, model_pool, node_runner) —
                       a standalone primitive; imports only agents/llm/tools/tracing
    workflow/      ←   the persisted Workflow *package* layer (package, registry,
                       runner, validate, schema, node_tool) — imports graph.engine
    builder/       ←   the conversational Architect (build, conversation, refine,
                       tool_author, schemas, nodes/, package/) — imports graph.workflow
  tasks/ sessions/ memory/ prompts/ observability/ config/
  # ── product ──
  app/             ← coding-assistant product: agents/ (personas), tools/, cli/
  cli/             ← back-compat package-alias shim → app/cli (from F6b; not yet drained)
```

> **Layering inside `graph/` is strictly one-way:** `builder → workflow → engine`.
> Importing the engine (or `master_agent.graph`) never pulls in `workflow`/`builder`
> (kept out of the eager path via lazy `__getattr__`), so the engine remains a clean,
> standalone core primitive while the whole orchestration story lives in one place.
>
> **Deleted across C+G:** `core/` shim, `workflows/graph/` shim, `workflows/_runtime/`
> husk; the top-level `workflows/` and `architect/` packages (folded into `graph/`).
> **Remaining shim (out of scope):** top-level `cli/` alias → `app.cli` (F6b).

Remaining smells:
- Package is still named `master_agent` (→ `neurosurfer` in F7).
- Top-level `cli/` is still a back-compat alias shim → `app.cli` (drain at/with F7).

---

## Target layout — framework-first (Phase F)

The repo is **a framework first**, with the workflow builder and the coding
assistant as **features/products layered on top**. Three layers:

```
neurosurfer/                       (the FRAMEWORK — rename of master_agent, last phase)
  # ════════ CORE PRIMITIVES — "everything in hand" ════════
  agents/        ← THE agent engine (promoted from core/): Agent, loop, subagent,
                   permissions, events, messages, context_manager, structured,
                   durable_state, tasks_runtime, summary_prompt, lexical.
                   Personas are NOT here — the engine must not import app personas.
  llm/           ← providers (anthropic, openai…), types, registry, retry,
                   tokens, capabilities
  tools/         ← Tool framework (base, registry, schema, coerce, generated) +
                   GENERIC builtins (read/write/run/list/search/http/web_search/
                   memory/ask_user/todo/finish/spawn_agent)
  rag/           ← retrieval pipeline (chunker, ingestor, context_builder,
                   filereader, url_fetcher, picker) — REWIRED onto llm/tools/
                   embeddings; promoted out of workflows/_runtime
  vectorstores/  ← base, chroma, in_memory  (promoted out of _runtime; clean)
  embeddings/    ← native embedder protocol + backends (consolidated from
                   memory/embeddings; shared by rag + memory)
  memory/        ← distill, retrieval, store, models  (uses embeddings/)
  cache/         ← NEW (greenfield): LLM/response + embedding caches
  prompts/       ← prompt templates + management
  tracing/       ← tracer (already promoted, R1)
  sessions/  tasks/  config/   ← supporting framework infra
  # ════════ GRAPH ORCHESTRATION SUBSYSTEM (Phase G) ════════
  graph/         ← one cohesive package for the whole agentic-workflow story;
                   strict one-way layering builder → workflow → engine:
    engine/      ←   the DAG engine (the LangGraph analog) — standalone core
                     primitive; re-exported at `graph` top. Owns node_runner.py
                     (native base/react/tool bridge). Imports only agents/llm/
                     tools/tracing — never a feature.
    workflow/    ←   the Workflow *package* abstraction: a persisted, versioned,
                     runnable DAG — package, registry, runner, validate, schema,
                     node_tool. No conversational logic.
    builder/     ←   the conversational graph-BUILDER (Architect): build,
                     conversation, refine, tool_author, schemas, nodes/, package/.
                   workflow + builder are imported lazily so importing the engine
                   never pulls the LLM/conversation stack.
  # ════════ PRODUCT / APP (the coding assistant) ════════
  app/           ← coding-assistant product, built on the framework:
    agents/      ← personas: explore, analyzer, writer, verifier (+ registry)
    tools/       ← app-flavored tools: apply_edit, present_plan, …
    cli/         ← REPL, commands, render, theme, completer
  # ════════ retired ════════
  workflows/_runtime/   ← FULLY DELETED (server, db, rag, vectorstores, utils, …)
```

> **Decisions made (R3/R4 + Phase F):**
> - Canonical agent = native engine (R4). Canonical tools = native `Tool` (R3).
> - **Framework-first**: core primitives are top-level; the workflow builder and
>   coding assistant are features/products on top, not the center of gravity.
> - **Graph engine = core primitive** (`neurosurfer/graph/`), not workflow-internal.
> - **Personas + app tools move to `app/`**; the framework ships only generic
>   primitives. The agent engine must not depend on app personas (inject via registry).
> - **Drop** the vendored OpenAI server + sql db (`_runtime/{server,db}`) — unused;
>   re-add a serving feature later if needed.

---

## Comparison findings (the basis for the decisions)

### Agent loop: `core/` vs `workflows/_runtime/agents/react`

| | master-agent `core/` (≈320 LOC) | neurosurfer ReAct (≈832 LOC) |
|---|---|---|
| Tool calling | **Native provider tool-use** (`response.tool_uses()`) | **Text parsing** — `ToolCallParser`, regex, JSON-repair loops |
| Final answer | Structured `stop_reason` + `finish` control signal | `FinalAnswerGenerator` with **sentinel tokens** → leaks `<__final_answer__>` |
| Termination | Clean `finish` / guardrail `max_turns` | `max_loop_iterations reached without a Final Answer` (the bug you saw) |
| Output | Async stream of typed **events** (TextDelta/Tool…/RunFinished) | Buffered text + JSON extraction/repair |
| Safety | Permission gating, guardrails, context compaction | Bounded retries, tolerant parsing |

→ `core/` is the production-grade design (mirrors Claude/OpenAI native function
calling). The neurosurfer ReAct is the *direct cause* of Issue 2 (sentinel leak,
no conversational guard, max-iter stalls). **Adopt `core/`.**

### Tool framework: `tools/Tool` vs `_runtime/tools/ToolSpec`

| | master-agent `Tool` | neurosurfer `ToolSpec` |
|---|---|---|
| Schema | `pydantic input_model` → JSON schema (one source of truth) | hand-written `ToolParam`/`ToolReturn` lists |
| Calling | aligns with **native tool-use** | built for **text-based** tool calls (`when_to_use`, `relax`) |
| Errors | validation/exec errors auto-returned as tool results | manual `check_inputs(relax=…)` |
| Lines to add a tool | a pydantic model + `async call` | a verbose spec + param list |

→ `Tool` is cleaner, pydantic-native, and pairs with the `core/` agent and native
tool-use. **Adopt `Tool`; retire `ToolSpec`.**

> Consequence: the mature path is master-agent's native stack (Agent + Tool +
> Provider native tool-use). The neurosurfer stack (ReAct text-parsing + ToolSpec +
> BaseChatModel) is legacy. The valuable neurosurfer piece to KEEP is the **DAG graph
> engine** (`_runtime/agents/graph`) — but its node executor gets rewired onto `core/`.

---

## Phase R — Reorganization (PRIORITY)

### R0 — Decisions & guardrails  ✅
- [x] Confirmed: canonical agent = `core/` (native tool-use); neurosurfer ReAct = retired.
- [x] Confirmed module names; `workflows/graph/` for DAG, `tracing/` at top level.
- [x] Re-export-shim strategy adopted and executed throughout R1–R4.
- [x] Baseline established; 421 tests green at end of R4 (3 pre-existing anthropic-SDK failures).

### R1 — De-bury the *cleanly separable* pieces
- [x] **tracing** → `master_agent/tracing/` (+ shim at old path; runner/tests repointed).
      Fully self-contained — clean. *(commit fd2002b)*
- [~] **DAG engine** (`_runtime/agents/graph`) → **DEFERRED to R4.** Dependency audit
      shows it is *bidirectionally* coupled to the to-be-retired neurosurfer stack: it
      imports `agents.{agent,react,rag}` + `models` + `tools` + `server`, **and**
      `_runtime/agents/__init__` plus `tools/base_tool`, `react/agent`, `agent/agent`,
      `code_execution` all import `graph.errors` (the shared `NeurosurferError`
      hierarchy) back. Relocating now = throwaway shims; move it *while* rewiring its
      node executor in R4.
- [~] **rag + vectorstores** → **DEFERRED.** Master-agent code does not use the RAG
      agent at all (only a test import-smoke-checks it); it drags in `vectorstores` +
      `models.embedders`. Relocate when RAG becomes a real feature (with/after R3/R4).
- [x] ruff: exclude promoted-but-not-cleaned vendored modules.

> **Lesson:** only `tracing` was genuinely decoupled. The valuable targets (graph, rag)
> are tangled with the vendored agent/model/tool stack — they move *as part of* the
> R3/R4 rewrite, not as standalone relocations. Next clean win is **R2** (architect
> conversation out of `cli/`), which is independent of the `_runtime` entanglement.

### R2 — Move the Architect's conversation out of the CLI  ✅  (architect stays under workflows/)
- [x] `cli/architect_convo.py` → `workflows/architect/conversation.py` (git mv; relative
      imports fixed `..llm` → `...llm`).
- [x] Clean public API: `from master_agent.workflows.architect import ArchitectBuilder,
      ArchitectConversation`. UI seam preserved (`run(ask=…, say=…)` callbacks).
- [x] `cli/commands/workflow.py` imports the package API; only rendering/IO callbacks
      (`_make_tool_approver`, `notify`, `on_node_event`) remain in `cli/` — correct for a
      delivery layer. No orchestration/domain logic left in `cli/`.
- [x] Test `tests/test_architect_no_cli_dep.py`: importing the architect pulls in **no**
      `master_agent.cli` modules; builder/conversation usable from pure Python.

### R3+R4 — MERGED: port workflow node execution onto the `core/` agent + `Tool`

> **Finding:** R3 is not separable from R4. Workflow tools are **already** master-agent
> `Tool`s — `MasterAgentToolAdapter` (`tools/workflow_bridge.py`) wraps each one into a
> neurosurfer `BaseTool`+`ToolSpec` *only so the neurosurfer `Agent`/`ReActAgent` can
> consume them*. `ToolSpec` exists solely because those agents execute nodes. So you
> can't retire `ToolSpec` until nodes stop using the neurosurfer agent — i.e. until R4.
> The adapter is the seam; cutting it = the R4 port. Hence one merged effort.

Incremental, test-green path:
- [x] **Foundation: structured output on the native stack.** `core/structured.py`
      `structured_completion(provider, schema)` — a synthetic `submit_result` tool whose
      input schema *is* the pydantic model (native tool-use ⇒ valid JSON), with a repair
      loop. Standalone, doesn't touch the loop. Tests: `test_structured_completion.py`.
- [x] **Port `base` and `react` nodes** (R3+R4 main step). `workflows/_node_runner.py`:
      `run_base_node` (provider.complete / structured_completion), `run_react_node`
      (core.Agent.run_collect in bypass mode), `run_tool_node` (native Tool.run async).
      `GraphExecutor` gains `provider` / `native_tools` / `tool_ctx` params and a
      `_run_node_native()` dispatch path. Legacy `llm`/`toolkit` path preserved for
      backward compat during transition.
- [x] **Cut the seam** (runner.py): `WorkflowRunner` no longer imports
      `MasterAgentToolAdapter` or `ProviderChatModel`; builds native `ToolPool` and
      passes `provider=` directly to the executor.
- [x] **Promote DAG engine** → `workflows/graph/` re-export shim. All files outside
      `_runtime` now import from `master_agent.workflows.graph`. The physical files
      still live in `_runtime/agents/graph/`; move them once the old path is fully
      drained (next step).
- [x] **Final cleanup**: deleted `_runtime/agents/{agent,react,common,code,sql_agent}`,
      `_runtime/tools`, `_runtime/models`. Moved physical DAG files from
      `_runtime/agents/graph/` to `workflows/graph/` (shims left at old paths for
      backward compat). Removed `MasterAgentToolAdapter`, `ProviderChatModel`,
      `workflow_bridge.py`, `chat_model_adapter.py` (dead code).
- [x] **Subsumes Q2**: native tool-use removes the `<__final_answer__>` sentinel and the
      max-iter stalls for free; conversational guard added on top.

### R5 → folded into Phase F (rename is now the final step **F7**)

---

## Phase F — Framework-first consolidation (PRIORITY, runs after R4)

> **Goal:** turn the repo into a clean framework with the workflow builder and
> coding assistant as features/products on top. Same rules as Phase R: re-export
> shims so every move is reversible and the suite stays green step-by-step; one
> PR-sized commit per sub-phase; `pytest -q` after each.

> **Audit facts (grounding this plan):**
> - `core/` holds the real agent **engine**; `agents/` holds only coding
>   **personas** (explore/writer/analyzer/verifier) — backwards for a framework.
> - **RAG is currently broken**: `_runtime/agents/rag` imports the deleted vendored
>   stack (`BaseChatModel`, `Toolkit`, `Agent`, `embedders`, `common.utils`). It
>   must be **rewired** onto the native stack, not just relocated.
> - `_runtime/vectorstores` is **cleanly decoupled** (easy promote).
> - A native `Embedder` protocol already exists in `memory/embeddings.py` — RAG
>   rewires onto it (no need to resurrect deleted embedders).
> - No `cache/` module exists today — greenfield.
> - `_runtime/{server,db}` are unused by CLI/workflows → dropped (decision).

### F0 — Decisions & target tree  ✅
- [x] Graph engine = top-level core primitive (`neurosurfer/graph/`).
- [x] Personas + app-flavored tools move to a product `app/` layer.
- [x] Drop `_runtime/{server,db}`.
- [x] Target tree above agreed.

### F1 — Rewire + promote RAG and vectorstores out of `_runtime`  ✅
- [x] **Promote vectorstores** (clean): `git mv _runtime/vectorstores → vectorstores/`
      (+ shim). Repoint internal imports.
- [x] **Consolidate embeddings**: lift `memory/embeddings.py` into `embeddings/`
      (shared by rag + memory); leave `memory` importing from it.
- [x] **Rewire RAG** onto native stack: `BaseChatModel`→`Provider`, deleted embedders
      →`embeddings/`, `extract_and_repair_json`→inlined. RAGAgent is now standalone
      (no deleted base class); `super().run()` replaced with direct `Provider.complete()`
      via `_run_async()`. `Toolkit` accepted but unused (deferred to F5/F6).
- [x] `git mv _runtime/agents/rag → rag/` (+ shim). All broken imports fixed;
      432 tests green (5 pre-existing anthropic-SDK failures unrelated).

### F2 — Add the `cache/` module (greenfield)  ✅
- [x] `cache/` with a response-cache interface + in-memory and on-disk backends.
- [x] `CachedProvider` wrapper — opt-in via config, off by default (`cache=None` is
      transparent pass-through). Keyed on model+messages+system+tools+temperature+max_tokens.
- [x] `CachedEmbedder` wrapper — in-memory LRU cache for embedding calls.
- [x] 36 tests: hit/miss/expiry, LRU eviction, disk roundtrip, stream not cached.

### F3 — Promote the graph engine to top-level `graph/`  ✅
- [x] `git mv workflows/graph → graph/` (+ shim at `workflows/graph`).
- [x] `workflows/` keeps only the Architect feature; imports via `workflows.graph` shim.
- [x] Dropped `_runtime/agents/graph` back-compat shims; fixed two stragglers
      (`_runtime/agents/__init__.py`, `workflows/schema.py`) that still used deleted subpath.
      468 tests green.

### F4 — Collapse the rest of `_runtime/` (+ drop server/db)  ✅
- [x] **Dropped** `_runtime/server`, `_runtime/db`, `_runtime/{models,tools}` (empty),
      `_runtime/{logger,diagnostics,runtime,utils}` — all dead (zero live consumers).
- [x] Relocated `ChunkerConfig` → `rag/config.py` (its only consumer); rest of the
      legacy `config.py` god-object (App/Database/BaseModel configs) was server/db-only → deleted.
- [x] Inlined `__version__` into `workflows/__init__.py` (dropped `_runtime/version.py`).
- [x] Repointed the 4 live `_runtime` imports: tracing→`master_agent.tracing`,
      vectorstores.base→`master_agent.vectorstores.base`; cleaned stale docstring refs.
- [x] **Deleted `workflows/_runtime/` entirely** — nothing buried remains.
- [x] Fixed latent F1 regression: `vectorstores/__init__.py` now imports the Chroma/
      InMemory backends lazily (via `__getattr__`) so importing `Doc` no longer pulls
      the optional `chromadb` dep. 471 tests green under `LLMs` env (2 unrelated
      env-specific web_search failures from an HTML-lib version diff).

### F5 — Promote the agent engine: `core/` → `agents/`; personas → app  ✅
- [x] **F5a** — Extracted personas (explore/analyzer/writer/verifier) into `app/agents/`;
      the `SubAgentDefinition` + registry primitive split into engine module
      `core/subagent_defs.py` (the type/registry are framework primitives; only the
      concrete personas are app code).
- [x] **F5a** — **Decoupled the engine from app personas**: `core/subagent.py` no longer
      imports `..agents`; it reads the registry via `.subagent_defs`. Personas register
      themselves into the engine registry when the **product** (`master_agent.app`) is
      imported. CLI entry imports `master_agent.app` at startup. Verified: engine alone
      registers 0 personas; `import master_agent.app` registers all 4.
- [x] **F5b** — `git mv core/ → agents/` (engine now top-level). Left a lazy `core/`
      shim package (submodule `sys.modules` aliases + package-level `__getattr__`) so all
      `master_agent.core.X` imports keep working and share the same module objects
      (registry singleton not split). App imports use the canonical `master_agent.agents`.
- [x] `spawn_agent`/`cli/render` need no change — they key off `agent_type` strings via
      the registry/ToolContext, not persona imports. 471 tests green throughout.

### F6 — Stand up the `app/` product layer + reclassify builtin tools  ✅
- [x] **F6a** — `app/` product layer: `app/agents/` (personas, F5), `app/tools/`
      (present_plan + register_task), `app/cli/` (F6b). Added a `register_tool_factory`
      hook so the framework registry never imports product/feature tools.
- [x] **F6a** — Classified builtins: generic (read/write/run/list/search/http/web_search/
      browse/data/**apply_edit**/memory/ask_user/todo/finish/spawn_agent) stay in framework
      `tools/builtin`; present_plan + register_task → `app/tools`; `write_workflow_node`
      → `workflows/node_tool.py`. **Deviation**: `apply_edit` kept generic — it's a shared
      workflow-node worker primitive (in `_BUILTIN_WORKFLOW_NODE_TOOLS`); moving it to app
      would couple the workflows feature to the product. Verified: framework-alone registry
      = 15 generic tools; app/workflows contribute theirs on import.
- [x] **F6b** — `git mv cli/ → app/cli/` (+ back-compat `cli/` package-alias shim). Rewrote
      39 framework-reaching relative imports → absolute (depth-safe). Entry points repointed
      (`pyproject` script + `__main__`) to `master_agent.app.cli`; tests repointed; the
      architect-no-CLI invariant now guards `master_agent.app.cli`.
- [x] Verified entry points (`app.cli:main`, `python -m master_agent`, shim) + 471 green.

> **Known seam (follow-up):** `tasks/runner.build_full_pool` imports `app.tools`
> (framework-root → product). Fully relocating product-pool assembly into `app/` is
> deferred; the central `tools/` package itself no longer imports product code.

### F7 — Package rename + repo remap (LAST, mechanical) — *was R5*  ✅
- [x] `master_agent` → `neurosurfer` (package dir, all imports, entry points).
- [x] `pyproject.toml`: `name = "neurosurfer"`, `license = "Apache-2.0"`, script
      `neurosurfer = "neurosurfer.app.cli:main"`, URLs → `github.com/NaumanHSA/neurosurfer`.
- [x] Env var `MASTER_AGENT_HOME` → `NEUROSURFER_HOME`; temp prefix `ma_pyexec_` → `ns_pyexec_`.
- [x] CLI banner: ASCII "NEURO" block logo + `s u r f e r` subtitle; version line updated.
- [x] README fully replaced: neurosurfer banner, Quick Start, architecture tree,
      install options table, Apache-2.0 license.
- [x] Global sed passes: `master_agent` → `neurosurfer`, `master-agent` → `neurosurfer`,
      `MASTER_AGENT` → `NEUROSURFER` (case-sensitive; three separate passes).
- [x] 555 tests green, zero regressions.

### Option 1 — Copy clean code to neurosurfer public repo  ✅
- [x] Removed legacy neurosurfer package (`neurosurfer/agents/agent/`, `cli/`, `models/`,
      `server/`, `db/`, etc.), build artifacts (`dist/`, `egg-info/`), and runtime dirs
      (`logs/`, `temp/`, `rag-storage/`, etc.) from `/home/nomi/workspace/neurosurfer/`.
- [x] Copied: `neurosurfer/` package, `tests/`, `pyproject.toml`, `README.md`,
      `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CHANGELOG.md`,
      `.github/`, `scripts/` from master-agent → neurosurfer repo.
- [x] Kept from existing neurosurfer repo: `docs/`, `tutorials/`, `neurosurfer_labs/`,
      `LICENSE` (Apache-2.0), `mkdocs.yml`, `CITATION.cff`, `NOTICE`, `Dockerfile`,
      `docker-compose.yml`, `requirements_docs.txt`.
- [x] 555 tests green in `/home/nomi/workspace/neurosurfer/`.
- [x] master-agent permanently retired; neurosurfer is the active development repo.

---

## Phase C — Cleanup & de-shim (PRIORITY — runs before F7)

> **Why now:** F1–F6 deliberately left re-export shims so each move stayed green.
> Those shims (and one husk) are now technical debt: they let new code keep importing
> the *old* paths, hide a layering violation, and triple the rename surface for F7.
> This phase drains them. Same rules as before: one PR-sized commit per step,
> `pytest -q` + `ruff` green after each; a shim may live one step longer than its
> last consumer, then gets deleted in the same phase.

> **Audit facts (grounding this phase):**
> - `core/` is a **pure shim package** — every `core/<x>.py` does
>   `sys.modules[__name__] = master_agent.agents.<x>`. **19 import lines / 11 files**
>   still import `master_agent.core.*`.
> - `workflows/_runtime/` has **zero tracked `.py` files** (`git ls-files` empty) —
>   only empty dirs + stale `__pycache__` on disk. Pure `rm -rf`.
> - `workflows/graph/` is a shim → `master_agent.graph`; **6 modules** still import
>   it (incl. `workflows.{package,schema,validate,runner}` and `app/cli/render_workflow`).
> - **Backwards dependency:** `graph/executor.py` (core engine) imports
>   `workflows._node_runner` (a *feature* module) at lines 342/415/439. The engine
>   must not depend up on a feature.
> - `_node_runner.py` is **not** a duplicate engine — it's the native sync↔async
>   bridge that delegates to `agents.Agent` / `structured_completion` / `Provider` /
>   `ToolPool`. It is *graph-execution glue*, not workflow-package logic.

### C1 — Kill the `_runtime/` husk  ✅
- [x] Confirmed no live import references `workflows._runtime` (grep).
- [x] `rm -rf master_agent/workflows/_runtime` (untracked husk — `git ls-files` empty).
      Cleaned stale pyproject refs: dropped the `_runtime/**` ruff exclude and the
      stale `pydantic-settings` comment (now points at `rag/config.py`).

### C2 — Drain the `core/` shim → `agents/`  ✅
- [x] Repointed the 19 import lines (11 files) from `master_agent.core.*` /
      `..core.*` → `master_agent.agents.*` (precise dotted-path sed so `scores`/etc.
      were not clobbered); fixed three stale `core.X` doc/comment refs.
- [x] Repointed the test suite too (12 files importing `master_agent.core`).
- [x] grep → zero `core` importers → **deleted the `core/` shim package**. 473 green.

### C3 — Drain the `workflows/graph` shim → `graph/`  ✅
- [x] Repointed all 6 `workflows.graph` importers (incl. tests) to `master_agent.graph`
      (`.graph` → `..graph` inside `workflows/`; fully-qualified elsewhere);
      **deleted `workflows/graph/`**. 473 green.

### C4 — Move the native execution bridge into the engine (fix the backwards dep)  ✅
- [x] `git mv workflows/_node_runner.py → graph/node_runner.py`; repointed the 3
      `graph/executor.py` imports to the local `.node_runner` module. Dropped two
      pre-existing dead imports (`pathlib.Path`, `StructuredCompletionError`) in the
      moved file.
- [x] Verified `graph/` now imports only core primitives —
      `grep -rn 'workflows\|architect' master_agent/graph` → empty. Backwards dep gone.

### C5 — Split the Architect into a top-level feature  ✅  *(decision: split builder → top-level)*
- [x] `git mv workflows/architect → architect/` (top-level feature module).
- [x] `workflows/` now holds **only** the Workflow *package* abstraction
      (`package`, `registry`, `runner`, `validate`, `schema`, `node_tool`); docstring
      updated. No conversational logic remains under `workflows/`.
- [x] **Deviation:** rewired architect's outward-reaching imports to **absolute**
      (`master_agent.llm/tools/config/workflows.*`) rather than leaving a relative
      re-export shim — depth-safe, matches the F6b CLI pattern, and avoids a temporary
      shim. Intra-architect imports stay relative. Updated the package YAML module
      paths (`master_agent.workflows.architect` → `master_agent.architect`).
- [x] CLI: `app/cli/commands/workflow.py` import of the builder now points at
      `master_agent.architect`. The `architect-imports-no-cli` invariant test now
      guards `master_agent.architect`. 473 green.

### C6 — Verification gate  ✅
- [x] grep: **zero** import references to `master_agent.core`, `workflows._runtime`,
      `workflows.graph`, or `workflows.architect` (remaining `_runtime` text hits are
      the unrelated `tasks_runtime` module + historical provenance comments).
- [x] `pytest -q` → **473 passed** (unchanged from baseline; zero regressions).
- [x] `ruff check master_agent`: **662 → 655** (fixed the 7 I introduced/uncovered:
      5 import-sort + 2 dead imports; added none). Remaining 655 are pre-existing
      vendored debt, tracked separately — **not** "clean" repo-wide, but no regression.
- [x] `mypy`: the 7 errors in `clarify`/`refine`/`node_runner` are pre-existing
      logic-level issues (untouched by the import moves), not introduced here.
- [x] Updated the **Current layout** + **Target layout** trees in this doc to match.

---

## Phase G — Consolidate the graph orchestration subsystem  ✅  *(decision: single `graph/` umbrella)*

> **Rationale:** engine + Workflow-package + builder are one domain (graph
> orchestration). Phase C had them as three top-level packages (`graph/`,
> `workflows/`, `architect/`); Phase G folds them into one cohesive `graph/` package
> with three submodules, while *keeping the engine a clean standalone primitive*.
> Names chosen: **`engine` / `workflow` / `builder`**.

- [x] **engine**: `git mv graph/*.py → graph/engine/`; converted `node_runner.py`'s
      outward relatives → absolute. New umbrella `graph/__init__.py` re-exports the
      engine API (so `from master_agent.graph import Graph, GraphExecutor` still works)
      and exposes `workflow`/`builder` via lazy `__getattr__`.
- [x] **workflow**: `git mv workflows/ → graph/workflow/`; repointed engine imports to
      `master_agent.graph.engine` and other outward relatives (`llm`/`tools`/`config`/
      `tracing`) → absolute; merged a duplicate engine-import block in `schema.py`.
- [x] **builder**: `git mv architect/ → graph/builder/`; repointed all references via
      the global path rewrite (incl. the package `graph.yaml` `output_schema`/`callable`
      module paths).
- [x] **Global rewrite** across `master_agent/` + `tests/`:
      `master_agent.workflows` → `master_agent.graph.workflow`,
      `master_agent.architect` → `master_agent.graph.builder` (src, tests, YAML, docstrings).
- [x] **Decoupling invariant (test-verified):** `import master_agent.graph` does **not**
      load `graph.workflow` or `graph.builder`; engine depends only on core primitives;
      layering is strictly one-way `builder → workflow → engine`.
- [x] `pytest -q` → **473 passed** throughout (zero regressions). `ruff` **655 → 642**
      (fixed 19 import-sort issues from the absolute conversions; added none).
- [ ] (Follow-up for F7) the bare name `graph/workflow/` is fine; revisit whether the
      top-level `cli/` shim drains here or at the rename.

---

## Phase A — Agent family in `agents/`  ✅  *(decision: AgenticLoop / ReactAgent / Agent)*

> **Status: complete.** `agents/` is now a family of agent types over a shared
> `BaseAgent`, with the primitives organized into subpackages. **488 tests green**
> (473 + 15 new); ruff 642 → 640 (no new lint).

> **Goal:** turn `agents/` from "one engine" into a small **family of agent types**
> users pick from, all sharing the mature primitives. New agent types get added here
> as the repo grows. The native multi-step loop is already production-grade; the two
> new types must be too (no resurrection of the retired buggy neurosurfer ReAct).

> **Naming (locked):**
> - **`AgenticLoop`** — the native-tool-use, multi-step autonomous loop. *Renamed from
>   today's `Agent`* (the mature engine in `agents/agent.py`).
> - **`ReactAgent`** — production-grade **text-parsing** ReAct (Thought/Action/Observation)
>   for providers **without** a native tool-calling API (small/local models — the project
>   thesis). New.
> - **`Agent`** — the *simple* **one-shot** agent: a single LLM call with optional tools
>   (bounded to ≤1 tool round by default) and optional structured output. The plain name
>   for the plain agent. New. (`Agent` is *repurposed* — see migration below.)

### A0 — Organize shared primitives into submodules
> Flat `agents/*.py` becomes grouped subpackages; agent-type classes stay at the top.
> Public API preserved via `agents/__init__.py` re-exports (no external import churn).
```
agents/
  __init__.py  base.py  agentic_loop.py  react_agent.py  oneshot.py
  conversation/   ← messages.py, events.py
  context/        ← manager.py (was context_manager), durable_state.py, summary_prompt.py
  runtime/        ← loop.py (tool dispatch), permissions.py, structured.py, tasks_runtime.py
  subagents/      ← runner.py (was subagent), defs.py (was subagent_defs)
  lexical.py      ← kept flat (text-ranking util shared with memory/web_search;
                    flagged to relocate out of agents/ at F7)
```
- [x] `git mv` each primitive into its subpackage; add `__init__.py` per subpackage
      re-exporting its public names.
- [x] Fix intra-`agents/` imports (relative, depth-aware) and the handful of external
      consumers (`memory/`, `tools/`, `graph/engine/node_runner.py`, `tasks/`); keep the
      top-level `agents/__init__.py` re-exporting the same public surface.

### A1 — Extract `BaseAgent`
- [x] `agents/base.py`: `BaseAgent` holding the shared state + plumbing currently inside
      `Agent` — provider, history (`messages`), tool pool, guardrails/permissions, IO,
      event emission, context compaction (`context_manager`), and the structured-output
      helper (`structured`). The three agent types subclass it.

### A2 — `AgenticLoop` (rename of today's `Agent`)
- [x] `agents/agent.py` → `agents/agentic_loop.py`; class `Agent` → `AgenticLoop`
      (logic unchanged — it's already mature). Subclass `BaseAgent`.
- [x] **Migration surface (~32 lines / 7 files):** `agents/__init__.py`,
      `agents/subagent.py` (child spawns), `graph/engine/node_runner.py`
      (`run_react_node`), `tasks/runner.py`, + tests (`test_agent_loop`,
      `test_hardening`, `test_subagents`). Careful rename — do **not** touch
      `SubAgent*`, `AgentError`, `ManagerAgent`, `agent_type`, etc.

### A3 — `ReactAgent` (production-grade, text-parsing)  ← genuinely new
- [x] `agents/react_agent.py`, `ReactAgent(BaseAgent)`. Loop: prompt with
      Thought/Action/Action-Input/Observation; tool catalog described **in the prompt**
      (no provider tool schemas). Reuses our `events`, `permissions` gating, tool
      dispatch, and `structured` validation.
- [x] **Hardened against the retired impl's bugs:** tolerant action parser with bounded
      format/JSON repair; clean final-answer extraction (**no sentinel leakage** to
      output); deterministic termination (`Final Answer` *or* `max_steps` → returns a
      partial answer, never empty); optional `output_schema` → validated model.
- [x] Targets providers without native tool-use; pick the path by
      `provider.capabilities` (native tool-use ⇒ recommend `AgenticLoop`).

### A4 — `Agent` (one-shot, simple)  ← genuinely new
- [x] `agents/oneshot.py`, class `Agent(BaseAgent)`. Single `provider.complete()`;
      optional `output_schema` ⇒ `structured_completion`; optional tools ⇒ execute one
      bounded round then a synthesis call (`max_tool_rounds=1` default). Returns a typed
      result (text or pydantic model) + usage. Reuses `structured`, `permissions`, `events`.

### A5 — Rewire + export + tests
- [x] `graph/engine/node_runner.py`: `run_base_node` → `Agent` (one-shot);
      `run_react_node` → `AgenticLoop` (native) or `ReactAgent` (non-native), chosen by
      capability. Thin wrappers; no behavior change for existing workflows.
- [x] `agents/__init__.py` exports `BaseAgent`, `AgenticLoop`, `ReactAgent`, `Agent`.
- [x] Tests: `test_react_agent.py` (parsing/repair, termination, no-sentinel, structured),
      `test_oneshot_agent.py` (text, structured, bounded tool round); update loop tests
      to `AgenticLoop`. Full suite stays green.

> **Open for F7:** `agents/` (the engine family) vs `app/agents/` (personas) name clash
> still stands — revisit when renaming the package.

---

## Phase N — Neurosurfer feature parity (after cleanup, before/with F7)

> Bring over the capabilities neurosurfer has that we dropped during the port, rewired
> onto the native stack (`agents/`, `llm/`, `tools/`, `rag/`). Each is a self-contained
> feature module — none should re-introduce the retired ReAct / `ToolSpec` / `BaseChatModel`
> stack. *(decision: server + code-exec + web-search/RAG tools in scope; SQL deferred.)*

### N1 — OpenAI-compatible server  (`server/`)  ✅
- [x] Ported `neurosurfer/server/` rewired onto the native stack: `NeurosurferServer`
      gateway, `AgentBackend` + `UpstreamBackend`, `/v1/chat/completions` +
      `/v1/models` + `/health` routes, SSE streaming, OpenAI chunk formatting, hooks
      (`Hook` / `StripReasoningHook` / `SystemPromptInjectorHook`), middleware,
      request/response schemas. Lives in `master_agent/server/`.
- [x] `AgentBackend` detects native async generators (`AgenticLoop` / `ReactAgent`)
      and streams `TextDelta` events → OpenAI SSE chunks in real time; async coroutines
      and sync agents also supported (thread bridge via anyio).
- [x] `master-agent serve [--host] [--port] [--upstream-url] [--upstream-api-key]
      [--log-level] [--workers] [--reload] [--no-docs]` CLI entry. (Subsumes Phase H.)
- [x] `[serve]` optional extra (fastapi + uvicorn) already present in `pyproject.toml`.
- [x] 39 new tests; 525 total green (zero regressions).

### N2 — Sandboxed code-execution agent + tool  ✅
- [x] `tools/builtin/python_exec/` submodule: `errors.py` (`CodeExecutionError`),
      `env.py` (env sanitisation — strips API keys/tokens, overrides HOME/TMPDIR),
      `sandbox.py` (subprocess execution with process-group kill on timeout, memory cap
      via `resource.setrlimit`, result side-channel via temp file, syntax pre-check),
      `tool.py` (`PythonExecTool` native `Tool`), `__init__.py` (public surface).
- [x] Guardrails: syntax check before spawn; process group killed on timeout (no
      orphan grandchildren); `RLIMIT_AS` memory cap (Linux); sensitive env vars stripped
      (regex-matched: API keys, tokens, passwords, secrets, etc.); HOME/TMPDIR pinned
      to sandbox; `PYTHONUNBUFFERED=1`, `PYTHONFAULTHANDLER=1` for deterministic output.
- [x] `result =` variable captured via side-channel temp file (no stdout pollution);
      `keep_sandbox=True` flag preserves the sandbox dir for follow-up reads.
- [x] Code agent scratchpad pattern emerges naturally from `AgenticLoop` + `python_exec`
      — no separate class needed (any native agent loop with this tool has the pattern).
- [x] Registered in `tools/builtin/__init__.py` and `tools/registry.py` (default pool).
- [x] 28 new tests; 527 total green after N2 (zero regressions).

### N3 — Web-search submodule with pluggable engines  ✅
- [x] `web_search.py` → `tools/builtin/web_search/` submodule. Structure:
      `config.py` (env constants), `engines/{base,ddg,serpapi}` (pluggable backends),
      `extractor.py` (BS4 block-walker → trafilatura fallback), `ranker.py` (chunk_text
      + `rank_chunks`/`select_within_budget` re-exports), `tool.py` (`WebSearchTool` +
      `WebSearchArgs` with `engine: Literal['ddg','serpapi'] | None`), `__init__.py`
      (full backward-compat re-exports including `config` module for test patching).
- [x] Engine selection: `WEB_SEARCH_ENGINE` env var (default `ddg`); override per-call
      via `engine=` arg. `DuckDuckGoEngine` gates on `ddgs` package; `SerpApiEngine`
      gates on `SERPAPI_API_KEY` env var, uses `httpx` sync client (no extra dep).
- [x] Extraction order inverted from original: **BeautifulSoup block-walker** (primary,
      inline-safe — `<a>`/`<sup>` stay on one line) → trafilatura (fallback for complex
      pages). This is strictly better: original order caused nav/fragment leakage when
      trafilatura was installed.
- [x] Full backward compat: `browse.py` (`extract_body`), `tools/builtin/__init__.py`
      (`WebSearchTool`), `tests/test_web_search.py` — all unchanged except one monkeypatch
      target (now `ws.config.BUDGET_TOKENS` instead of `ws.BUDGET_TOKENS`).
- [x] 555 tests green (up from 527); zero regressions.

> **Deferred from N3 scope:** high-level RAG tool surfaces (`kb_tool`, `docs_generator`,
> `simple_query_assistant`) — deferred to Phase Q or a separate feature phase.

> **Deferred (not scheduled):** SQL agent + `tools/sql/` + `db/` (NL→SQL vertical);
> docs site + tutorials; packaging polish (Dockerfile, CITATION/CONTRIBUTING).

> **Deferred (not scheduled):** SQL agent + `tools/sql/` + `db/` (NL→SQL vertical);
> docs site (`docs/` mkdocs) + `tutorials/` + `neurosurfer_labs/`; `neurosurferui/`;
> packaging polish (Dockerfile, docker-compose, CITATION/CONTRIBUTING/CODE_OF_CONDUCT).

---

## Phase Q — Feature quality: the general agent + deep workflow building (REPLANNED 2026-06-30)

> **Reframe.** The original Phase Q treated two issues in isolation (deeper plans /
> better ReAct). The real product gap is bigger and the two halves must be designed
> **together**: today the front face is *only* the Workflow Architect — free-form text
> goes straight to `_build` ([app/cli/app.py:10-12](../../neurosurfer/app/cli/app.py#L10),
> [:119-122](../../neurosurfer/app/cli/app.py#L119)). There is **no general agent**, even
> though the engine for one already exists and is mature (`AgenticLoop` / `ReactAgent` /
> `Agent`, a 15-tool default pool + live MCP tools, `present_plan`, and a real **plan
> mode** in [permissions.py:255](../../neurosurfer/agents/runtime/permissions.py#L255)).
>
> **The target experience.** The user faces a **general automation Assistant**. It
> handles light automation directly, plans-then-executes heavy one-offs, and — its
> headline ability — **designs & registers reusable workflows/graphs** for work that is
> recurring or genuinely pipeline-shaped. Behind that face, the Architect must produce
> **deep** workflows (the current ones are shallow because a *single* `plan` base node
> emits a 2–8 node graph — [graph.yaml:62-115](../../neurosurfer/graph/builder/package/graph.yaml#L62),
> moving to `architect/package/graph.yaml` in Q0.5).

### The 4-tier routing model (the spine of this phase)

| Tier | Trigger (Assistant's judgment) | Behaviour | Mechanism |
|---|---|---|---|
| **1 — Converse** | chit-chat, a quick question | answer directly, no tools | general agent turn |
| **2 — Light automation** | a few files / a script / a web fetch | just do it | `default` mode, full tool pool |
| **3 — Heavy one-off** | many files, lots of writes/shell, long horizon | **plan → approve → execute** | **hard gate: plan mode** |
| **4 — Workflow-worthy** | recurring, multi-stage, "build me a pipeline" | **propose → confirm → Architect builds a registered workflow** | `propose_workflow` handoff |

> Tiers 1–2 are the general agent being useful. Tier 3 is the **hard plan-mode gate**
> (writes/shell blocked until `present_plan` is approved). Tier 4 is the **handoff** to
> deep workflow building. The general agent and the workflow builder are the *same
> conversation*, escalating in depth as the work demands.

### Q0 — Decisions locked (from the 2026-06-30 design pass)
- [x] **Front face = general Assistant**, not the bare Architect. Free text → Assistant.
- [x] **Escalation = agent judgment via a `propose_workflow` tool** — the Assistant
      decides a task is workflow-worthy, explains why, and asks the user to confirm
      before handing off. (Not a pre-classifier; not explicit-only.)
- [x] **Heavy one-offs = hard plan-mode gate** — reuse the existing plan mode; a
      heaviness pre-flight sets the initial mode so writes/shell are blocked until
      `present_plan` is approved.
- [x] **Workflow depth = multi-stage decompose pipeline** — replace the single `plan`
      node with `decompose → design_nodes → critique`, deep by construction.
- [x] **The workflow builder becomes a separate top-level component `architect/`** —
      pulled out of `graph/builder/`. Runtime (`graph/` = engine + workflow) vs authoring
      (`architect/` = the builder) are cleanly separated. (Confirmed 2026-06-30.)

---

### Q0.5 — Promote the builder to a top-level `architect/` component (do this FIRST)  ✅

> Lift `graph/builder/` out to a standalone top-level component **before** Q1 starts
> adding files to it. Pure structural move — no behaviour change, suite stays green.
> Reverses Phase G decision #11 *for the builder only*: engine + workflow stay folded
> under `graph/`; the builder leaves. Target tree:
>
> ```
> graph/      engine/  workflow/            ← runtime: "run a DAG"
> architect/  build.py conversation.py refine.py tool_author.py
>             schemas.py exemplars.py(NEW) nodes/ package/   ← authoring: "build a DAG"
> ```

- [x] **Q0.5a** — `git mv neurosurfer/graph/builder → neurosurfer/architect` (11 files,
      rename-detected). Rewrote self-references `neurosurfer.graph.builder.*` →
      `neurosurfer.architect.*` (the `build.py`/`__init__`/`conversation.py` docstrings;
      the `package/graph.yaml` `output_schema` / `callable` module paths — 4 paths).
      All relative imports were verified safe to move (`from ..schemas` in `nodes/`
      still resolves to the package root; all other intra-package imports are single-dot;
      outward imports were already absolute from Phase C/G).
- [x] **Q0.5b** — Dropped `builder` from the lazy `__getattr__` set + `__all__` and
      rewrote docstrings in [graph/__init__.py](../../neurosurfer/graph/__init__.py) and
      [graph/workflow/__init__.py](../../neurosurfer/graph/workflow/__init__.py). The
      engine/workflow ↛ builder invariant is now **structural** (separate top-level
      package); verified `neurosurfer.graph.builder` no longer resolves.
- [x] **Q0.5c** — Repointed the 2 import lines in
      [app/cli/commands/workflow.py:157,300](../../neurosurfer/app/cli/commands/workflow.py#L157)
      → `neurosurfer.architect`, and the 6 test files
      (`test_workflow_architect`, `test_architect_gap_resolution`, `test_tool_author`,
      `test_workflow_refine`, `test_workflow_validation`, `test_architect_no_cli_dep`),
      including monkeypatch-target strings and the `__module__` assertion.
- [x] **Q0.5d** — The `architect-imports-no-cli` invariant test now guards
      `neurosurfer.architect` (`from neurosurfer.architect import ArchitectBuilder` pulls
      in **no** `neurosurfer.app`/`cli`). **449 tests green** (baseline, zero
      regressions); ruff introduced **zero** new findings (the 5 `I001` in the touched
      files all pre-exist on HEAD — pre-existing vendored debt).

### Q1 — Deep workflow building (Issue 1): multi-stage decompose pipeline  ✅

> Replace the one shallow `plan` base node with a real decomposition pipeline inside the
> Architect graph (`architect/package/graph.yaml`, post-Q0.5). New node order:
> `discover → clarify → decompose → design_nodes → critique → write_nodes → assemble`
> (was `discover → clarify → plan → write_nodes → assemble`).

- [x] **Q1.1 — New schemas** (`architect/schemas.py`): `Stage` (id, name, purpose,
      rationale, min_nodes, capabilities) and `StagePlan` (intent, ordered `list[Stage]`,
      depth_rationale). `critique` reuses `WorkflowPlan` as its output type — no separate
      `CritiqueResult` needed (simpler and downstream just receives the improved plan).
- [x] **Q1.2 — `decompose` node** (base, structured → `StagePlan`). Breaks the intent
      into 3–6 conceptual stages before touching nodes. Forces deep thinking by having the
      LLM commit to stage count before knowing node details. Inline few-shot example
      (GitHub PR digest) shows a 5-stage decomposition so the floor is calibrated.
- [x] **Q1.3 — `design_nodes` node** (base, structured → `WorkflowPlan`) — replaces
      `plan`. Seeded with the `StagePlan`; "one or two nodes per stage" rule enforces
      that the design is at least as deep as the stage plan. Carries forward all
      tool-mapping guidance and `{available_tools}` interpolation.
- [x] **Q1.4 — `critique` node** (base, structured → `WorkflowPlan`). Reviews the draft
      for 5 failure modes: over-consolidated nodes, missing steps, wrong kind (base doing
      I/O), wrong tools, shallow DAG. Returns unchanged plan if correct or an improved
      WorkflowPlan. `write_nodes` and `assemble` both depend on `critique` output.
- [x] **Q1.5 — Few-shot exemplar** inline in `decompose` prompt. 5-stage GitHub-PR-digest
      example shows exactly what a well-decomposed stage plan looks like; no separate
      `exemplars.py` file needed (avoided indirection).
- [x] **Q1.6 — Depth-floor validation** (`graph/workflow/validate.py`): `validate_package`
      now emits a warning if the generated workflow has fewer than 3 LLM nodes. Actionable
      message pointing to missing intermediate steps.
- [x] **Q1.7 — Wire-up**: `assemble.py` now accepts `critique` kwarg (prefers it, falls
      back to `plan` for backwards compat). `_ARCHITECT_NODE_LABELS` in `workflow.py`
      updated for all 7 architect pipeline nodes. Test updated: `len == 7`, node-id set
      updated. **449 tests pass.**

### Q2 — The general front-facing Assistant (the new default face)  ✅

> The Assistant is an **app/product** concern. It is the *top-level* REPL agent (not a
> subagent/persona). Built on the mature engine, picking `AgenticLoop` for all configured
> providers (both anthropic and openai tool-call styles support native tool use).

- [x] **Q2.1 — Assistant persona prompt** (`app/cli/assistant.py:_SYSTEM_PROMPT`). Identity,
      capability inventory (files, shell, code, web, present_plan, live MCP tools), headline
      ability (design & register reusable workflows), and the 4-tier routing rubric (Tiers 1–4:
      converse / do-it / plan-first / mention-the-workflow-option). Kept to ~1.6 k chars.
- [x] **Q2.2 — Front-agent factory** (`app/cli/assistant.py:build_assistant()`). Resolves
      provider → `AgenticLoop(provider, default_pool(), system_prompt, Guardrails(...), io,
      cwd)`. `default_pool()` already folds live MCP tools + `present_plan` automatically.
      `propose_workflow` (Q3) will appear in the pool once that tool is registered.
- [x] **Q2.3 — REPL rewire** (`app/cli/app.py`): free-form text now calls `_assist(ctx, line)`,
      not `_build`. `/workflow build` still reaches the Architect directly. Module docstring
      updated (removed "There is no general-purpose chat agent"). Prompt hint updated to
      "Ask me to do something".
- [x] **Q2.4 — Streaming render** in `_assist`: `render.stream_events()` consumes the
      `AgenticLoop` event generator; Ctrl-C / `CancelledError` handled cleanly.
      `propose_workflow` pre-wired in `_INTERACTIVE_TOOLS` and `_TOOL_BASE` (render.py)
      so it renders correctly once Q3 lands. **449 tests pass.**

### Q3 — The bridge: heaviness gate + workflow handoff  ✅

> This is where "general agent + workflow building work together." Two seams: a
> pre-flight that sets **plan mode** for heavy work (Tier 3), and a control-signal tool
> that hands off to the Architect (Tier 4).

- [x] **Q3.1 — Heaviness pre-flight → initial mode** (`app/cli/assistant.py`). Before the
      Assistant's first turn on a new input, a fast assessment returns
      `(plan_required, reason)`: **heuristic-first** (touches-many-files / write / shell /
      "refactor|migrate|across the repo" cues) with an **optional** one-shot `Agent`
      structured classify for ambiguous cases. `plan_required=True` →
      `initial_mode(True)` ([permissions.py:255](../../neurosurfer/agents/runtime/permissions.py#L255))
      → the loop starts in **plan mode**, so writes/shell are hard-blocked until
      `present_plan` is approved. Light tasks start in `default`. (Keep the assessment
      cheap; it runs once per user input, not per turn.)
- [x] **Q3.2 — `propose_workflow` control tool** (`app/tools/propose_workflow.py`, new;
      registered via `register_tool_factory` like `present_plan`
      [present_plan.py:41](../../neurosurfer/app/tools/present_plan.py#L41)). Args:
      `intent` (refined, expanded), `reason` (why a durable workflow beats a one-off),
      `recurring: bool`. `call()` asks the user to confirm via `ctx.io`; on **yes** it
      returns `ToolResult.ok(..., handoff_workflow=True, intent=…)`; on **no** it returns
      a "continue as a one-off" message so the loop proceeds. It is a **control tool**
      (add to `CONTROL_TOOLS` [permissions.py:32](../../neurosurfer/agents/runtime/permissions.py#L32))
      — never gated, produces no side effects itself.
- [x] **Q3.3 — Loop + REPL handoff**. The agent loop already surfaces tool `control`
      signals ([agentic_loop/loop.py:80-86](../../neurosurfer/agents/agentic_loop/loop.py#L80)):
      treat `handoff_workflow` like `finished` — end the run and carry the intent on
      `RunFinished` (or a new `events.WorkflowHandoff`). `_assist` detects it and calls
      the existing `_build(ctx, intent)` ([commands/workflow.py:153](../../neurosurfer/app/cli/commands/workflow.py#L153)),
      so Tier 4 lands in the (now-deep) Architect pipeline. `ReactAgent` needs the same
      control-surface hook if it doesn't have it yet.
- [x] **Q3.4 — Keep the explicit path**. `/workflow build`, `list`, `run`, `show`,
      `refine`, `delete` stay exactly as-is — power users skip the Assistant. The handoff
      is additive.

### Q4 — Startup capability surface ("tell the user what it can do")  ✅

- [x] **Q4.1 — Capabilities card** (`app/cli/banner.py:print_banner()`): two lines inserted
      between the provider row and the tips. Line 1 (dim): general automation capabilities
      (files · shell & code · web search). Line 2 (accent ✦): "Design & register **reusable
      workflow pipelines** for recurring or multi-stage jobs." The ✦ + accent color makes the
      workflow headline visually distinct without being loud.
- [x] **Q4.2 — Prompt hint + tips** (`app/cli/app.py`, `banner.py`): done in Q2. Prompt
      hint reads "Ask me to do something"; `_TIPS` now mixes general-agent examples
      ("just type what you want — I can read/write files…", "summarise this folder…") with
      workflow and provider tips. **449 tests pass.**

### Q5 — Conversational guard (old Q2 remainder) — mostly resolved by the reframe

> Adopting the native engine (R4) already removed the root causes:
- [x→R4] native tool-use ⇒ no `<__final_answer__>` sentinel leak.
- [x→R4] structured stop/`finish` ⇒ no `max_loop_iterations` stalls.
- [x→R4] typed event stream ⇒ clean CLI output.
> The old bug ("Hi there!" ran the whole pipeline) **disappears** once the front face is
> a general agent — chit-chat is just a Tier-1 turn. Remaining:
- [x] **Q5.1** — Guard confirmed present and correct at `architect/conversation.py:63`
      post-Q0.5 move: `"If the user asks something unrelated to workflow building, politely
      redirect."` The no-tool-call fallback in `run()` handles the redirect path: LLM writes
      redirect text (surfaced via `say()`), `ask("…", [])` collects the user's real intent
      as free text, conversation continues. No code change needed — Q2 already removed the
      original bug (chit-chat at the REPL no longer reaches `ArchitectConversation` at all;
      the guard now only applies to the explicit `/workflow build` path). **449 tests pass.**

### Q6 — Tests + benchmark
- [ ] **Q6.1** — Architect depth: a build for an intensive intent (e.g. "document a repo")
      produces ≥ depth-floor nodes incl. a verify stage; `decompose`/`critique` schemas
      round-trip; depth-floor validation flags a shallow plan.
- [ ] **Q6.2** — Routing: Tier-1 chit-chat returns text with no tool calls; Tier-3 heavy
      input starts in plan mode and a write is blocked pre-approval; `propose_workflow`
      yes → handoff control fires, no → loop continues.
- [ ] **Q6.3** — Assistant construction picks `AgenticLoop` vs `ReactAgent` by capability;
      MCP + `present_plan` + `propose_workflow` all present in the pool.
- [ ] **Q6.4** — End-to-end benchmark: run the deep doc workflow on a real repo; compare
      node count / output quality against the pre-Q1 shallow baseline.

> **Suggested order:** Q0.5 (separate `architect/` — structural, no behaviour change) →
> Q1 (deep builder, isolated & testable) → Q2 (Assistant) → Q3 (bridge) → Q4 (startup) →
> Q5/Q6. Each sub-phase is one PR-sized commit, `pytest -q` green after each, same as
> Phases R–G.

---

## Testing & rollback

- Re-export shims keep every move reversible and the suite green step-by-step.
- One module per PR-sized commit; run `pytest -q` + `ruff` after each.
- No behavior change in Phase R (pure reorg) — the 448 tests must stay green
  throughout. Behavior work is isolated to Phase Q.

## Decisions locked
**Phase R round:**
1. ✅ Canonical agent = master-agent `core/` (port workflow nodes onto it — R4).
2. ✅ Canonical tools = master-agent `Tool` (retire `ToolSpec` — R3).
3. ✅ Architect stays under `workflows/`; only its conversation leaves `cli/` (R2).

**Phase F round (framework-first):**
4. ✅ Framework-first: core primitives are top-level; workflow builder + coding
   assistant are features/products on top, not primary.
5. ✅ Graph/orchestration engine = top-level core primitive (`neurosurfer/graph/`).
6. ✅ Coding personas + app-flavored tools move to a product `app/` layer; the
   framework ships only generic primitives and must not depend on app personas.
7. ✅ Drop the vendored OpenAI server + sql db (`_runtime/{server,db}`).

**Phase C / N round (cleanup + parity):**
8. ✅ Split the conversational **Architect** out of `workflows/` into a top-level
   feature module (`architect/`); `workflows/` keeps only the Workflow package/runner
   abstraction. Layers: `graph/` (engine) → `workflows/` (package) → `architect/` (builder).
9. ✅ Move the native node-execution bridge (`_node_runner.py`) **down** into `graph/`
   to kill the `graph → workflows` backwards dependency.
10. ✅ Re-introduce (rewired on the native stack) the **OpenAI-compatible server**,
    **sandboxed code-exec**, and **web-search + RAG tools** (Phase N).
    **SQL agent/tools/db deferred**; docs/UI/packaging deferred.

**Phase G round (graph consolidation) — supersedes #8:**
11. ✅ Fold engine + Workflow-package + builder into one cohesive `graph/` package:
    `graph/{engine, workflow, builder}` (was top-level `graph/`+`workflows/`+`architect/`).
    Engine re-exported at `graph` top; `workflow`/`builder` lazy. Layering strictly
    one-way `builder → workflow → engine`; the engine stays a standalone primitive.

**Phase A round (agent family):**
12. ✅ `agents/` hosts a growing **family** of agent types over a shared `BaseAgent`.
    Names: **`AgenticLoop`** (native tool-use multi-step — renamed from `Agent`),
    **`ReactAgent`** (new text-parsing ReAct for non-function-calling models),
    **`Agent`** (new simple one-shot: single call + optional tool round + structured).
    `Agent` is repurposed from the loop to the one-shot agent.

**Phase Q round (general agent + deep workflow building) — 2026-06-30:**
13. ✅ The CLI front face becomes a **general automation Assistant**, not the bare
    Architect. Free-form text routes to the Assistant; the Architect is reached via a
    Tier-4 handoff or the explicit `/workflow build`.
14. ✅ **Escalation to workflow-building = agent judgment** via a `propose_workflow`
    control tool (Assistant proposes → user confirms → handoff). Not a pre-classifier;
    not explicit-only.
15. ✅ **Heavy one-offs = hard plan-mode gate** — a cheap heaviness pre-flight sets the
    initial mode; writes/shell are blocked until `present_plan` is approved. Reuses the
    existing plan mode, no new gating mechanism.
16. ✅ **Workflow depth = multi-stage decompose pipeline** — replace the single `plan`
    node with `decompose → design_nodes → critique` (+ few-shot exemplars + a depth-floor
    validation check). Deep by construction.
17. ✅ **The workflow builder is a separate top-level component, `architect/`** — pulled
    out of `graph/builder/` (Q0.5). Clean split: `graph/` = runtime (engine + workflow,
    "run a DAG"); `architect/` = authoring ("build a DAG"). Layering stays one-way
    `app → architect → graph.workflow → graph.engine`; `graph/` never imports `architect/`;
    `architect/` stays drivable from pure Python (no `app`/`cli`). The Phase-Q app-layer
    pieces (Assistant, `propose_workflow`) live in `app/`, not in `architect/`.
    **Supersedes #11 for the builder only** — engine + workflow remain folded under `graph/`.

## Still open (later, non-blocking)
- Rename `core/` → `agents/` now, or at R5 with the package rename? (cosmetic)
- How much of `_runtime/{server,db,runtime,rag,vectorstores}` is worth keeping vs dropping.
- `_runtime/agents/graph/` backward-compat shims can be deleted once no external code
  imports the old path (safe to remove at R5 or when confirmed unused).
