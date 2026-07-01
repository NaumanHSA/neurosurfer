# Changelog

All notable changes to neurosurfer are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

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
