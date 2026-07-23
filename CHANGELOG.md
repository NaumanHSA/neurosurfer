# Changelog

All notable changes to neurosurfer are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

- **Graph engine: control flow.** Workflows are no longer linear-only. New node kinds —
  `router` (the router *is* the classifier: one LLM call picks a labelled route, others are
  pruned; or deterministic `cases`), `loop` (iterate a nested `body` until a plain-English
  `until` judge says stop, with the reason threaded back as `{feedback}`; or deterministic
  `break_when`), `map` (fan out a `body` over a collection with a `concurrency` cap),
  `subgraph`, and `input` (human-in-the-loop). Any node can also carry a `when:` guard
  (conditional edge, OR-join at merges), `on_error:` fallback, `writes:` (name an output for
  downstream `{templates}` and expressions), and `policy.retries`. Predicates run through a
  **safe expression evaluator** (no `eval`/imports/attribute access). Authorable in YAML or via
  the fluent **`GraphBuilder`**, and JSON round-trippable. Fully backward compatible. See the
  [Graph & Workflows guide](guides/graph-workflows.md).
- **Workflow execution API + live streaming.** Drive and observe workflows over HTTP on the
  FastAPI gateway: list/inspect workflows, start runs (`POST /v1/workflows/{name}/runs`), tail
  them node-by-node over **SSE** (`GET /v1/runs/{id}/events`, replays from seq 1 then live-tails),
  fetch per-node detail, resume human-in-the-loop `input` nodes, and cancel. Runs persist to disk
  for replay.
- **The Architect (ReAct agent).** `ArchitectAgent(provider).build(intent)` turns plain English
  into a runnable, validated, registered workflow package via an 11-tool belt on one staged
  session — never registering an invalid graph. It **knows neurosurfer from live code** (a
  versioned capability manifest — node kinds, tools, expressions, MCP — that a freshness test
  keeps from drifting), reaches for control flow only when the intent warrants it, and can author
  new tools (sandboxed + approved) mid-build. MCP servers are first-class workflow citizens.
- **Closed-loop self-verification.** The Architect proves its own work: `derive_acceptance`
  turns an intent into testable criteria + concrete inputs, `verify_workflow` **actually runs**
  the staged workflow (incl. one case per branch for coverage) and LLM-judges each criterion
  (fail-closed), feeding failures back into design revision. Gate it with
  `ArchitectAgent(verify="required")`.
- **Observability: pluggable trace exporters.** Agent runs can now be shipped to an
  external monitoring backend — **Langfuse** (traces, token cost, sessions) and
  **OpenTelemetry** (GenAI-semconv spans over OTLP → Phoenix / Grafana / Datadog /
  Langfuse). A side-channel observer on the agent event stream maps runs → traces,
  LLM turns → generations (with token usage), and tool calls → spans. Auto-on from
  the environment (`LANGFUSE_*` / `OTEL_EXPORTER_OTLP_ENDPOINT`), or configure in code
  via `neurosurfer.observability.exporters`. Zero overhead when unconfigured; a bad or
  unreachable exporter never breaks a run. Install with
  `pip install "neurosurfer[observability]"`. See the
  [Observability guide](guides/observability.md).
- **Trace nesting + sessions.** A run started inside another run (a spawned
  sub-agent) now nests under it as a child span in the same trace, via an ambient
  trace context propagated across `await` and `asyncio.gather`. Agents accept a
  `session_id` so all their runs group into one Langfuse session — the CLI sets one
  per conversation (reset on `/clear`).

### Fixed

- **`.env` loader** now strips trailing inline comments on unquoted values
  (`KEY=val   # note` → `val`), while preserving `#` inside quoted values. Prevents
  a comment leaking into a value (e.g. a malformed `LANGFUSE_HOST`).

### Removed

- **`local-models` extra** (`torch`, `transformers`, `accelerate`, `sentencepiece`)
  — Neurosurfer never loads a model in-process; LLMs are reached over the API
  (Anthropic, OpenAI, and OpenAI-compatible servers over HTTP). The extra pulled a
  multi-GB torch stack for a capability that doesn't exist. `pydantic-settings`
  also moved out of the core install into the `rag`/`serve` extras that import it.
- **Docker artifacts** — the stale GPU `Dockerfile` and `docker-compose.yml` (built
  on the removed local-inference stack) are gone. Containerize the gateway with a
  minimal `python:3.12-slim` + `pip install "neurosurfer[serve]"` image — see
  [Deployment](server/deployment.md#containerized-deployment).

---

## [1.0.0] — 2026-07-01

First stable release. Neurosurfer is now a full framework for building
intelligent apps that blend LLM reasoning, tools, and retrieval — with a
ready-to-run OpenAI-compatible gateway, a graph/workflow runtime, an MCP client,
and an interactive CLI. The public API surface (`neurosurfer.agents`,
`neurosurfer.llm`, `neurosurfer.tools`, `neurosurfer.rag`, `neurosurfer.graph`,
`neurosurfer.architect`, `neurosurfer.mcp`, `neurosurfer.app.server`) is
considered stable under semantic versioning from this release onward.

### Added

- **Agent family** — `AgenticLoop` (native multi-step tool-use), `ReactAgent`
  (text-parsing ReAct for providers without a native tool-calling API), and
  `Agent` (a single bounded call with optional tools / structured output), all
  re-exported from `neurosurfer.agents`. Streaming is event-based
  (`TextDelta`, `ThinkingDelta`, `ToolStarted`/`ToolFinished`, `TurnCompleted`,
  `RunFinished`, …). Sub-agents (`SubAgentRunner`), permissions/guardrails
  (`Permissions`, `PermissionMode`, `Guardrails`), and context management
  (`ContextManager`, `DurableState`, auto-compaction) ship as shared primitives.
- **Provider layer** — `Provider` protocol with native Anthropic and OpenAI
  providers plus any OpenAI-compatible server (Ollama, LM Studio, vLLM,
  llama.cpp); `build_provider`, capability introspection, unified retry, token
  math, and canonical message/content/event types under `neurosurfer.llm`.
- **Vision support** — image content blocks flow through the canonical types and
  capable providers.
- **Graph & Workflow runtime** — `neurosurfer.graph.engine` (`Graph`,
  `GraphExecutor`, `GraphNode`, loader, errors) as a standalone DAG primitive,
  and `neurosurfer.graph.workflow` for persisted multi-file Workflow packages
  (load / validate / register / run).
- **Architect** — `ArchitectBuilder` / `ArchitectConversation`: describe a
  workflow in plain English and the Architect designs and builds a Workflow
  package (including a deep tool-design pipeline for capability-aware tools).
- **MCP client** — `neurosurfer.mcp` connects to Model Context Protocol servers
  and exposes their tools to agents; managed via `/mcp` in the REPL.
- **OpenAI-compatible gateway** — `NeurosurferServer` exposes `/v1/models`,
  `/v1/chat/completions` (SSE streaming), and `/health`; register upstream
  backends (`UpstreamBackend`) or native agents (`AgentBackend`); request/response
  `Hook`s and a `ModelRouter`. Started with `neurosurfer serve` or embedded in
  Python. Requires the `[serve]` extra.
- **Interactive CLI** — a `prompt_toolkit` REPL with persistent chat, slash
  commands, session reset, provider profiles (`~/.neurosurfer/providers.json`,
  mode 0600), per-task provider pinning, and a live status line; plus
  `neurosurfer serve`, `neurosurfer provider`, and `neurosurfer doctor`.
- **Built-in tools** — 15+ tools including web search (DuckDuckGo/SerpAPI),
  sandboxed Python execution, file ops, HTTP, a headless browser, and memory,
  discoverable via `neurosurfer.tools` (`default_pool`).

### Changed

- **Built-in task lineup** — the user-facing built-ins are now exactly **`code`**
  (interactive software-engineering agent operating in the working directory)
  and **`general`** (research, writing, data, light automation; redirects to
  `code` for substantial coding work). Both are `readonly` (protected);
  `task_builder` remains hidden (`system`).
- **Owner identity** standardized to **Neurosurfer Team** across `pyproject.toml`,
  `CITATION.cff`, and the README citation.

### Fixed

- **Usage-line Rich markup** — `/task` and `/provider`'s fallback usage messages
  embedded literal `[...]` inside `[style]...[/style]` markup, causing Rich to
  silently swallow everything after the first `[`. Brackets are now escaped.
- **A model that answers with plain text instead of calling `ask_user`** used
  to silently end the run (no tool call ⇒ the engine treats the turn as
  finished) with no indication anything went wrong. The base system prompt
  now states this constraint explicitly.

### Removed

- **`doc_gen` and `code_understanding` built-in tasks** — superseded by `code`.
- **Cost estimation / budget rail** — the hardcoded per-model USD price table,
  `estimate_cost_usd`/`budget_exceeded`, `Guardrails.budget_usd`, and
  `PolicyCeiling.allow_budget`/`max_budget_usd` are gone. Neurosurfer never
  estimates or caps API spend; use your provider's own billing/usage dashboard.
- **Legacy automation package** — the old `neurosurfer/automation/` package,
  `automation_builder`, `register_automation`, and the `neurosurfer automation`
  subcommand are removed. Workflow serving is now handled by the Graph/Workflow
  runtime and the OpenAI-compatible gateway (`neurosurfer serve`).

---

## [0.2.0] — 2026-06-13

### Added

- **Interactive REPL** — full prompt_toolkit shell with slash-command
  suggestions (`/`), arrow-key menus, history, and a live provider/task status
  line (green = connected).
- **Provider profiles** — named, switchable provider configurations stored at
  `~/.neurosurfer/providers.json` (mode 0600, secrets masked on display);
  managed via `/provider` in the REPL or `neurosurfer provider` subcommands.
- **Task Builder meta-agent** — converse to define and register a new Task;
  10-question interview covers goal, tools, guardrails, inputs, sub-agents, and
  plan gate; Task is validated against the policy ceiling before registration.
- **Interrupt & resume** — Ctrl-C cleanly persists the approved plan, todos,
  and decisions; `neurosurfer resume` / `/resume <run_id>` continues any
  interrupted run (plan gate skipped when already approved).
- **Budget rail** — per-Task `budget_usd` guardrail stops the run when
  estimated Anthropic API spend reaches the ceiling; partial result preserved.
- **Docker support** — multi-stage `Dockerfile` (slim, non-root), `.dockerignore`,
  `docker-compose.yml` with TTY passthrough and named state volume.
- **`/local` optional dep group** — `pip install "neurosurfer[local]"` adds
  `tiktoken` for accurate token counting with OpenAI-compatible local models.
- **`docs/DOCKER.md`** — full Docker and Compose reference.
- **`docs/TASKS.md`** — full Task system user guide with YAML reference.
- **174 tests** — unit, provider-parity, integration, and e2e; CI on every PR.

### Changed

- CLI restructured as a package (`neurosurfer/cli/`) with separate modules for
  theme, banner, rendering, IO handling, and doctor.
- System prompt assembly unified: `TaskRunner` calls
  `prompts.system.build_system_prompt`; base sections (identity, tone,
  tool-discipline, planning, guardrails, env) wrap every Task's body.
- `pyproject.toml`: added PyPI classifiers, project URLs, author email, `local`
  optional dep group; version bumped to 0.2.0.

### Fixed

- Loose small-model output for `todo` and `register_task` tools now tolerated
  (off-spec but recoverable JSON structures are repaired before validation).

---

## [0.1.0] — 2026-06-01

### Added

- Initial release.
- **Provider layer** — `AnthropicProvider` and `OpenAICompatProvider` behind a
  single `Provider` protocol; canonical message/content/event types; unified
  retry (429/500/529/timeout with backoff); tool-call schema validation and
  repair for weak local models.
- **Tool system** — `Tool` ABC, `ToolPool`, `ToolResult`; 12 built-in tools:
  `read_file`, `list_dir`, `search`, `run_command`, `write_file`, `apply_edit`,
  `ask_user`, `present_plan`, `todo`, `spawn_agent`, `finish`, `register_task`.
- **Agent loop** — manual async generator loop; plan gate, shell gate, and
  guardrail enforcement; parallel concurrency-safe tool dispatch.
- **Context management** — auto-compaction (threshold: effective window −
  13k buffer), reactive compaction on overflow, 8-section summary prompt;
  durable state (plan/todos/decisions) pinned outside the compactable history.
- **Sub-agent orchestration** — `spawn_agent` tool; built-in roles: `explore`,
  `analyzer`, `writer`, `verifier`; parallel via `asyncio.gather`; depth and
  concurrency caps enforced.
- **Task layer** — `TaskDefinition` (YAML), `TaskRegistry`, `TaskRunner`,
  `PolicyCeiling`; built-in Tasks: `doc_gen` and `task_builder`.
- **`neurosurfer doctor`** — provider reachability check.
- **`docs/PROVIDERS.md`** — LM Studio / vLLM / Ollama / llama.cpp / LiteLLM
  setup guide with recommended tool-calling models.
- CI: ruff + mypy + pytest on every pull request.

[1.0.0]: https://github.com/NaumanHSA/neurosurfer/compare/v0.2.0...v1.0.0
[0.2.0]: https://github.com/NaumanHSA/neurosurfer/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/NaumanHSA/neurosurfer/releases/tag/v0.1.0
