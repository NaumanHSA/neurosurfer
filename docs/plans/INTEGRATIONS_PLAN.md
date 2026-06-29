# Neurosurfer Integrations Plan — MCP, Vision, OpenTelemetry

> Status: **Phases 1 & 3 COMPLETE** (2026-06-25). P2 deferred; P4 not started.
> Scope of this plan: (1) MCP **client** ✅, (2) MCP **server** (workflows only),
> (3) **multimodal / vision** content support ✅, (4) **OpenTelemetry** tracing export.
>
> Sequencing recommendation: **P1 (MCP client) → P3 (Vision) → P2 (MCP server) → P4 (OTel)**.
> Each phase is independently shippable and additive. Keep tests green after every step.
>
> **P1 landed:** `neurosurfer/config/mcp.py` (`McpStore`/`McpServerConfig`),
> `neurosurfer/mcp/` (`McpTool`, `McpManager`), live-tool registry hook in
> `tools/registry.py`, `is_mcp` flag on `Tool`, MCP permission gate
> (`mcp_policy` on `Guardrails`), `/mcp` CLI command + REPL lifecycle + doctor section,
> `mcp` optional extra. Tests: `tests/test_mcp_client.py` (13).
>
> **P3 landed:** `ImageBlock` canonical type + `Message.user_with_images`
> (`llm/types.py`), `ToolResult.images` (`tools/base.py`) carried into history by the
> agent loop, Anthropic + OpenAI projection with `supports_vision` gating
> (`llm/providers/*`, `llm/capabilities.py`), per-image token heuristic
> (`llm/tokens.py`), image-returning `read_file` + `browse --screenshot`. No new deps
> (base64 passthrough). Tests: `tests/test_vision.py` (10). Full suite: 449 green.

---

## Conventions used in this doc

- `[ ]` = not started, `[~]` = in progress, `[x]` = done.
- `TODO(Pn.k)` markers are grep-able anchors; mirror them in code comments where the
  work lands so `grep -rn "TODO(P1" ` ties code back to this plan.
- "New file" / "Edit" tags state whether a task creates or modifies a file.
- File references point at the integration seams discovered during analysis.

---

## Architectural anchors (why these seams)

- **Tool contract**: `Tool` ABC derives its JSON schema from a pydantic `input_model`
  ([neurosurfer/tools/base.py](../../neurosurfer/tools/base.py)); `ToolSchema` already
  accepts a **raw** `input_schema: dict` ([neurosurfer/llm/types.py:89](../../neurosurfer/llm/types.py#L89)).
  → MCP tools (raw JSON Schema) fit by subclassing `Tool` and overriding `schema`/`parse_args`.
- **Tool registry**: static built-ins + `register_tool_factory` plugin hook + on-disk
  generated tools ([neurosurfer/tools/registry.py](../../neurosurfer/tools/registry.py)).
  MCP tools are *session-scoped & async-discovered*, so they attach to the live `ToolPool`
  at session start rather than via the static factory list.
- **Permissions** already anticipate MCP
  ([neurosurfer/agents/runtime/permissions.py:128](../../neurosurfer/agents/runtime/permissions.py#L128)):
  "Unknown / MCP tools: allow read-only-looking, otherwise allow (loop guards)." This plan
  replaces that line with a real gate.
- **Content blocks** are text/thinking/tool_use/tool_result only
  ([neurosurfer/llm/types.py:52](../../neurosurfer/llm/types.py#L52)); no image block →
  vision work adds one + provider projection.
- **Config stores** follow the `ProviderStore` pattern: a JSON file under `~/.neurosurfer/`,
  pydantic model, 0600 perms ([neurosurfer/config/profiles.py](../../neurosurfer/config/profiles.py)).
  MCP server config mirrors this.
- **Workflow registry** ([neurosurfer/graph/workflow/registry.py](../../neurosurfer/graph/workflow/registry.py))
  + `WorkflowRunner` ([neurosurfer/graph/workflow/runner.py](../../neurosurfer/graph/workflow/runner.py))
  are the entry points the MCP **server** exposes.
- **Capabilities** gate provider features ([neurosurfer/llm/capabilities.py](../../neurosurfer/llm/capabilities.py));
  vision adds a `supports_vision` flag here.
- **Gateway** is a FastAPI app with a pluggable router
  ([neurosurfer/app/server/gateway.py](../../neurosurfer/app/server/gateway.py),
  [neurosurfer/app/server/api/router.py](../../neurosurfer/app/server/api/router.py)) — MCP
  server endpoints mount alongside the OpenAI routes (or run standalone).

---

## Dependencies to add (pyproject optional-extras)

Keep the core install lean; gate each integration behind an extra.

```toml
# TODO(DEPS): add to [project.optional-dependencies] in pyproject.toml
mcp = ["mcp>=1.2.0"]                          # official MCP Python SDK (client + server)
otel = [
  "opentelemetry-api>=1.27",
  "opentelemetry-sdk>=1.27",
  "opentelemetry-exporter-otlp-proto-http>=1.27",
]
# vision needs no new deps (base64/URL passthrough); Pillow optional for local-file encoding:
vision = ["pillow>=10.0"]                     # optional: only for local image path → base64
```

- [ ] `TODO(DEPS)` add the three extras above and document them in README "What's in the box".

---

# Phase 1 — MCP Client (consume external MCP servers)

**Goal:** any configured MCP server's tools appear in the agent's `ToolPool` and are
callable/gated exactly like built-ins. stdio + streamable-HTTP/SSE transports.

**New module:** `neurosurfer/mcp/` (client side).

### P1.1 — Config store

- [x] `TODO(P1.1)` New file `neurosurfer/config/mcp.py` — `McpServerConfig` + `McpStore`
      mirroring `ProviderStore` ([profiles.py](../../neurosurfer/config/profiles.py)).
  - `McpServerConfig`: `name`, `transport: Literal["stdio","http"]`, `command`/`args`/`env`
    (stdio) or `url`/`headers` (http), `enabled: bool = True`, `tool_prefix: str | None`
    (namespace its tools to avoid collisions, e.g. `github__create_issue`).
  - Persist to `~/.neurosurfer/mcp.json`, 0600. Reuse `mask_secret` for header/token display.
  - Support `${ENV_VAR}` expansion in `env`/`headers` so secrets aren't stored raw.

### P1.2 — MCP tool adapter

- [x] `TODO(P1.2)` New file `neurosurfer/mcp/tool.py` — `McpTool(Tool)`.
  - Holds: `session` (live MCP `ClientSession`), server name, raw `input_schema`,
    `description`, and MCP `annotations`.
  - Override `schema` to return `ToolSchema(name, description, input_schema=<raw>)` — **no**
    pydantic round-trip (the server's JSON Schema is authoritative).
  - Override `parse_args` to pass the dict through (validate minimally; MCP server is the
    real validator). Keep `run()`'s error-return contract intact.
  - Map MCP tool annotations → behaviour flags:
    `readOnlyHint → is_read_only`, `destructiveHint → is_destructive`,
    `is_concurrency_safe = readOnlyHint and not openWorldHint`.
  - `call()`: invoke `session.call_tool(name, args)`; flatten returned content blocks
    (text; describe non-text parts) into `ToolResult.content`; `result.isError → is_error`.
    Never raise — return `ToolResult.error(...)` on transport failure.
  - `progress_message`: `"<server>: <tool>…"`.

### P1.3 — Connection manager / lifecycle

- [x] `TODO(P1.3)` New file `neurosurfer/mcp/manager.py` — `McpManager`.
  - `async connect_all()`: open a session per enabled server (stdio subprocess or HTTP),
    `initialize()`, `list_tools()`, wrap each as `McpTool`. Collect failures per-server
    (one bad server must not break the others); log + skip.
  - `tools() -> list[Tool]`: all wrapped tools across connected servers (prefixed).
  - `async aclose()`: tear down all sessions/subprocesses. Must be idempotent.
  - Async-context-manager friendly (`async with McpManager(...) as m:`).
  - Health: expose `status()` → per-server connected/tool-count/error for `/mcp` CLI + doctor.

### P1.4 — Wire into the agent / session

- [x] `TODO(P1.4a)` **Implemented differently (cleaner):** instead of `ToolPool.extend`, the
      manager publishes discovered tools to a process-global live registry
      (`set_live_tools`/`clear_live_tools`/`live_tools` in
      [tools/registry.py](../../neurosurfer/tools/registry.py)) which `all_tools()` folds in
      (built-in/generated win name clashes). This means *every* pool built from `all_tools()`
      / `default_pool()` sees MCP tools with zero per-call-site wiring. Deliberately **not**
      added to `workflow_node_tools()` (Architect must not compose graphs from ephemeral
      connections — see Phase 2).
- [x] `TODO(P1.4b)` Edit `CLIContext.create` ([app/cli/context.py](../../neurosurfer/app/cli/context.py))
      — construct an `McpManager` from `McpStore`, `connect_all()` at startup, attach tools to
      the default pool, and register `aclose()` for shutdown. Make startup non-fatal if a
      server is down.
- [x] `TODO(P1.4c)` Edit subagent runner ([agents/subagents/runner.py](../../neurosurfer/agents/subagents/runner.py))
      — MCP tools flow through `full_pool` automatically; verify allow/deny resolution
      (`defn.resolve_tools`) handles prefixed names. Add to subagent defs if needed.

### P1.5 — Permissions gate (replace the TODO at permissions.py:128)

- [x] `TODO(P1.5)` Edit [agents/runtime/permissions.py](../../neurosurfer/agents/runtime/permissions.py)
  - Add `MCP_TOOLS` detection (tools whose instance is `McpTool`, surfaced via a flag on the
    tool or a name registry passed to `Permissions`).
  - Add a `mcp_policy: Literal["gated","open","denied"] = "gated"` to `Guardrails`.
  - In `check()`: for MCP tools → if `denied` refuse; if `open` allow; if `gated` ask via
    `io.request_shell_approval(label, "call an MCP tool")` **unless** the tool's
    `readOnlyHint` is true (then allow). Reuse the existing approval channel so headless IO
    denies consistently.

### P1.6 — CLI surface

- [x] `TODO(P1.6)` New file `neurosurfer/app/cli/commands/mcp.py` — `/mcp` command:
      `list` (configured + live status), `add` (interactive), `remove`, `enable/disable`,
      `tools <server>` (introspect), `test <server>` (connect + list). Register in
      `build_registry` ([commands/__init__.py](../../neurosurfer/app/cli/commands/__init__.py)).
- [x] `TODO(P1.6b)` Edit doctor ([app/cli/doctor.py](../../neurosurfer/app/cli/doctor.py)) — add an
      MCP servers section (reachable? tool count?).

### P1.7 — Tests

- [x] `TODO(P1.7)` New file `tests/test_mcp_client.py`:
  - Fake in-process MCP server (the `mcp` SDK ships an in-memory transport) exposing 1–2 tools.
  - Assert: discovery wraps tools, schema passes through raw, `call()` maps content + errors,
    annotations map to flags, gate behaviour (gated/open/denied), prefix namespacing,
    one-server-down doesn't break others, `aclose()` tears down cleanly.

**P1 Definition of done:** `neurosurfer` REPL with a configured filesystem/github MCP server
lists its tools, an agent calls one, the call is gated per policy, and disconnect is clean.

---

# Phase 2 — MCP Server (serve workflows only)

**Goal:** expose **registered workflows** (and only those) as MCP **tools** so any MCP
client (Claude Desktop, Cursor, another neurosurfer) can invoke a neurosurfer-built
workflow natively. **Out of scope for now:** exposing built-in tools or RAG as MCP
resources (revisit later).

**New module:** `neurosurfer/mcp/server.py` (+ CLI serve command).

### P2.1 — Workflow → MCP tool mapping

- [ ] `TODO(P2.1)` New file `neurosurfer/mcp/server.py` — build an MCP server (FastMCP from the
      `mcp` SDK) that, on startup, reads `WorkflowRegistry`
      ([graph/workflow/registry.py](../../neurosurfer/graph/workflow/registry.py)) and registers
      one MCP tool per workflow:
  - Tool **name** = workflow name; **description** = package manifest description.
  - Tool **input schema** = derived from the workflow's declared inputs
    (`workflow.yaml` manifest / `Graph` inputs — see `WorkflowRunner` input validation,
    `InputValidationError`). Surface required inputs as JSON Schema properties.
  - Tool **handler** = run via `run_workflow` / `WorkflowRunner`
    ([graph/workflow/runner.py](../../neurosurfer/graph/workflow/runner.py)) with the active
    provider, returning the final artifact/text as MCP tool result.

### P2.2 — Provider + execution wiring

- [ ] `TODO(P2.2)` The server needs a `Provider` for `agent`-kind nodes. Resolve it from the
      active provider profile ([config/profiles.py](../../neurosurfer/config/profiles.py)) /
      `.env`, same resolution the CLI uses. Fail fast with a clear message if none configured.
- [ ] `TODO(P2.2b)` Map `WorkflowRunner` progress/`NodeEventCallback` → MCP progress
      notifications (optional, nice-to-have; gate behind a flag).
- [ ] `TODO(P2.2c)` Decide on refresh: re-scan the registry per `list_tools` call (cheap, always
      fresh) vs. cache at startup. Default to per-call scan so newly built workflows appear
      without a restart.

### P2.3 — Transport + serve command

- [ ] `TODO(P2.3)` New CLI: extend `serve` ([app/cli/commands/serve.py](../../neurosurfer/app/cli/commands/serve.py))
      with `neurosurfer serve --mcp` (stdio for Claude Desktop) and/or
      `neurosurfer serve --mcp-http --port N` (streamable HTTP). stdio is the priority
      (that's what desktop clients use).
- [ ] `TODO(P2.3b)` Optional: mount the HTTP MCP app alongside the OpenAI gateway in
      [gateway.py](../../neurosurfer/app/server/gateway.py) so one process serves both. Keep
      this optional — stdio standalone is the MVP.

### P2.4 — Safety

- [ ] `TODO(P2.4)` Workflows run real tools (shell, write, network). When served over MCP the
      caller is remote → run the workflow under a **restrictive default Guardrails**
      (`shell_policy="denied"` or `"gated"` with a non-interactive deny, `network_policy`,
      `write_scope` limited). Make the served-workflow guardrails explicit and configurable;
      document that serving workflows executes code.

### P2.5 — Tests

- [ ] `TODO(P2.5)` New file `tests/test_mcp_server.py`:
  - Register a trivial workflow, start the MCP server in-memory, connect a client,
    assert `list_tools` returns the workflow, `call_tool` runs it and returns the result,
    input-schema validation rejects bad args, restrictive guardrails are applied.

**P2 Definition of done:** `neurosurfer serve --mcp` exposes registered workflows; a generic
MCP client lists and successfully invokes one with the configured provider.

---

# Phase 3 — Multimodal / Vision support

**Goal:** pass images (screenshots, diagrams, PDF-page renders) into vision-capable models;
RAG/browse/verify agents can reason over pixels.

### P3.1 — Canonical content block

- [x] `TODO(P3.1)` Edit [llm/types.py](../../neurosurfer/llm/types.py) — add `ImageBlock`:
  - Fields: `type: Literal["image"]`, `source: Literal["base64","url"]`, `media_type`
    (e.g. `image/png`), `data` (base64) or `url`.
  - Add to the `ContentBlock` union (discriminated on `type`).
  - Update `Message.text()` / helpers to ignore image blocks gracefully, and add a
    `Message.user_with_images(text, images)` constructor.

### P3.2 — Provider projection

- [x] `TODO(P3.2a)` Edit [llm/providers/anthropic.py](../../neurosurfer/llm/providers/anthropic.py)
      `_block_to_param` — render `ImageBlock` → Anthropic `{"type":"image","source":{...}}`
      (base64 or url source).
- [x] `TODO(P3.2b)` Edit [llm/providers/openai.py](../../neurosurfer/llm/providers/openai.py) —
      render `ImageBlock` → OpenAI `image_url` content part (data URL for base64). Verify the
      adapter's content-flattening handles mixed text+image parts.
- [x] `TODO(P3.2c)` Token estimation ([llm/tokens.py](../../neurosurfer/llm/tokens.py)) — add a
      coarse image-token heuristic so compaction thresholds don't undercount.

### P3.3 — Capability gating

- [x] `TODO(P3.3)` Edit [llm/capabilities.py](../../neurosurfer/llm/capabilities.py) — add
      `supports_vision: bool` to `ProviderCapabilities`; set true for opus/sonnet 4.x and
      gpt-4o/o-series; false otherwise. Agents/tools check this before sending images and
      degrade gracefully (drop image + note) on non-vision models.

### P3.4 — Tooling that produces images

- [x] `TODO(P3.4a)` Edit browse tool ([tools/builtin/browse.py](../../neurosurfer/tools/builtin/browse.py))
      — optional `screenshot` mode returning an `ImageBlock` the loop appends to the next user
      turn (or surfaced via a tool-result convention). Define how a tool returns an image into
      history (today `ToolResult.content` is `str`) — extend `ToolResult` with optional
      `images: list[ImageBlock]` and have the loop append them.
- [x] `TODO(P3.4b)` Edit read_file ([tools/builtin/read_file.py](../../neurosurfer/tools/builtin/read_file.py))
      — when path is an image and model supports vision, return it as an `ImageBlock`
      (Pillow only needed for resize/local encoding; base64 passthrough needs no dep).
- [ ] `TODO(P3.4c)` (Optional) RAG: allow image inputs in the ingest/context path or note as
      future work — keep P3 focused on the message/provider plumbing.

### P3.5 — Loop / history plumbing

- [x] `TODO(P3.5)` Edit `MessageHistory` ([agents/conversation/messages.py](../../neurosurfer/agents/conversation/messages.py))
      — `add_user_images(...)` / extend `add_tool_results` to carry image blocks back into
      history. Ensure compaction ([agents/context/manager.py](../../neurosurfer/agents/context/manager.py))
      handles image blocks (drop/summarize old images first — they're token-heavy).

### P3.6 — Tests

- [x] `TODO(P3.6)` New file `tests/test_vision.py` (provider-fakes based, no network):
  - `ImageBlock` round-trips through both provider `_block_to_param` projections,
  - non-vision capability drops/avoids images,
  - `ToolResult.images` flow into history,
  - token heuristic counts images.

**P3 Definition of done:** an agent on a vision model receives an image (from `read_file` or
`browse` screenshot) and answers about it; non-vision models degrade cleanly.

---

# Phase 4 — OpenTelemetry export

**Goal:** export the existing trace/span data ([neurosurfer/tracing/](../../neurosurfer/tracing/))
as OTLP so runs show up in Langfuse / Phoenix / Datadog / Jaeger. **Additive** — the
in-house tracer stays the source of truth; OTel is an exporter sink.

### P4.1 — Exporter adapter

- [ ] `TODO(P4.1)` New file `neurosurfer/observability/otel.py` — translate neurosurfer
      spans ([tracing/span.py](../../neurosurfer/tracing/span.py),
      [tracing/models.py](../../neurosurfer/tracing/models.py)) into OTel spans:
  - One root span per agent run; child spans per turn / tool call / model call / workflow node.
  - Attributes follow GenAI semantic conventions where possible: `gen_ai.system`,
    `gen_ai.request.model`, `gen_ai.usage.input_tokens` / `output_tokens`, tool name, etc.
  - Pull token usage from `Usage` ([llm/types.py:98](../../neurosurfer/llm/types.py#L98)).

### P4.2 — Tracer hook

- [ ] `TODO(P4.2)` Edit [tracing/tracer.py](../../neurosurfer/tracing/tracer.py) /
      [tracing/config.py](../../neurosurfer/tracing/config.py) — add an optional OTel sink the
      tracer fans out to as spans open/close. Must be a no-op when OTel isn't installed/enabled
      (import-guard the `otel` extra).

### P4.3 — Configuration

- [ ] `TODO(P4.3)` Edit [config/observability.py](../../neurosurfer/config/observability.py) — read
      standard OTel env (`OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_EXPORTER_OTLP_HEADERS`,
      `OTEL_SERVICE_NAME`) + a `NEUROSURFER_OTEL_ENABLED` toggle. Off by default.
- [ ] `TODO(P4.3b)` Document the Langfuse/Phoenix endpoint recipe in docs.

### P4.4 — Tests

- [ ] `TODO(P4.4)` New file `tests/test_otel.py` — use OTel's `InMemorySpanExporter`; run a tiny
      traced agent loop; assert span tree shape + key GenAI attributes; assert total no-op when
      disabled.

**P4 Definition of done:** with OTel env set, a run produces a span tree visible in an OTLP
collector; disabled by default with zero overhead.

---

## Cross-cutting / housekeeping

- [ ] `TODO(X1)` README "What's in the box": add MCP (client + workflow server), vision,
      OTel bullets + the new extras.
- [x] `TODO(X2)` Tutorials: **done** `tutorials/04_mcp_servers.ipynb` (connect a server, discover
      tools, run an agent, the permission gate, persistence/CLI, + a §9 preview of serving
      workflows that becomes runnable when Phase 2 lands) and linked from `03`'s What's Next.
      Vision (Phase 3) is covered by the capstone `tutorials/05_capstone_insight_engine.ipynb`
      (function + react + MCP + **vision** in one graph), linked from `04`'s What's Next.
- [ ] `TODO(X3)` `.env.example`: add MCP/OTel env vars.
- [ ] `TODO(X4)` CHANGELOG entries per phase.
- [ ] `TODO(X5)` Ensure all new deps are optional extras; core install stays lean and the
      framework imports without `mcp`/`otel`/`pillow` present (import-guard everything).

---

## Open questions to settle before/at implementation

1. **MCP tool namespacing**: always prefix with server name, or only on collision? (Plan
   assumes optional `tool_prefix`, default = prefix on collision.)
2. **MCP server guardrails** (P2.4): what's the safe default when serving workflows remotely —
   `shell_policy="denied"` outright, or gated-with-deny? (Plan leans deny-by-default.)
3. **Vision image-return convention** (P3.4): extend `ToolResult` with `images: list[ImageBlock]`
   (chosen) vs. a separate event channel. Confirm before P3.
4. **MCP SDK version pin** — confirm the `mcp` package API (FastMCP, ClientSession, transports)
   against the installed version at implementation time.
