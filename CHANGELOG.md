# Changelog

All notable changes to neurosurfer are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

- **Retrieval, and the backends behind it** — plan 03, eight phases.
    - **Vector stores are a contract, not one class.** A declared filter grammar
      (`$eq`/`$ne`/`$in`/`$nin`/`$gt`/`$gte`/`$lt`/`$lte`/`$and`/`$or`/`$not`),
      `StoreCapability` flags so a caller asks rather than infers, upsert in the
      contract, and a **conformance suite** — "implements `BaseVectorDB`" now
      means "passes `tests/vectorstores/conformance.py`". **Qdrant** joins
      Chroma and the in-memory store and passed the suite unmodified.
    - **Embeddings is a plugin point.** `openai-compat:<model>@<base_url>` and
      hosted `openai` alongside sentence-transformers, resolved from a spec
      string, with batching and retry. A local stack no longer needs `torch` to
      embed. A collection records which model wrote its vectors and refuses a
      query embedded by a different one.
    - **Hybrid retrieval, reranking, MMR and citations.** Dense + BM25 fused by
      reciprocal rank, an optional cross-encoder rerank stage, diversity, and
      `char_start`/`char_end` spans through to `ContextBuilder`. Both off by
      default. `rag/evaluation.py` is the harness, and it was written before the
      things it measures.
      **How much hybrid helps depends heavily on the corpus.** On the synthetic
      fixture — built around a rare literal a dense model cannot represent —
      recall@1 goes 0.619 → 0.905. On this project's own 59 documentation pages
      with a real embedder it is *neutral* at k=1–3 and modestly ahead from k=5
      (MRR@10 0.610 → 0.653). Point the harness at your own corpus rather than
      trusting either number.
    - **Four retrieval shapes past classic** — contextual retrieval,
      parent-document, multi-query and HyDE, sentence-window and semantic
      chunking — plus an ingest manifest so a corpus with one edited file costs
      one file's embeddings rather than the whole directory's.
- **Google Gemini and Claude on Amazon Bedrock.** Gemini natively over `httpx`
  (no new dependency); Bedrock as a thin subclass of the Anthropic provider,
  since it serves the same API — only the client and the `anthropic.`-prefixed
  model id differ.
- **`react` nodes can return a shaped answer again.** `output_schema` was
  withdrawn from the kind because it was inert; the loop now runs and one
  structured call shapes its answer, so offering the field is honest. Costs one
  extra model call, billed to the node.
- **A node kind declares its tool-round budget.** `NodeKindSpec.tool_rounds` —
  `1` for `base`, `None` (unbounded) for `react`. It was a literal inside
  `run_base_node`, which made the single most consequential difference between
  the two kinds the one thing no consumer of the specs could read; the executor
  now takes the number from the spec. A new warning,
  `agent.tools_exceed_rounds`, uses it: a `base` node holding two or more tools
  may be a sequence, and a sequence is what one round cannot do. A warning, not
  an error — two independent lookups in a single parallel round work fine.
- **Control flow in graphs.** Six new node kinds — `router`, `loop`, `map`,
  `subgraph`, `input`, `output` — with typed workflow state, a safe expression
  evaluator, error routing (`on_error`), retries, and a `GraphBuilder` fluent API.
  A workflow can now branch on a classification, iterate until a judge is
  satisfied, fan out over a collection, and pause for a person. See the
  [control-flow guide](guides/graph-workflows.md).
- **A tool registry, and self-knowledge built from it.** Tools declare a `title`,
  an icon, `capabilities`, `secret_inputs` and `credential_help`, and are grouped
  by domain under `neurosurfer/registry/core/`. On top of that sits a
  content-hash-versioned capability manifest and a `KnowledgeBase`, so a need
  resolves against a **declared tag** — `file.write` → `write_file` — rather than
  against words a description happens to share.
- **The Architect plans first, grounds, and refuses.** It writes a plan, checks
  every capability against what actually exists, and reports a request it cannot
  build *before* designing a node for it. What it builds is then verified by
  being **run**, with branch coverage over the paths a test exercised, and the
  verification is fingerprinted so an unchanged design is not re-run.
- **Validation is a rule table.** Each rule declares which node kinds it speaks
  about and at what severity, so "what can go wrong with an input node" is a
  query rather than a read. Messages are plain sentences; field names, import
  paths and parser errors live in a separate `detail`.
- **Execution and Architect HTTP surfaces** — `/v1/workflows`, `/v1/runs` (with
  SSE), and `/v1/architect/*`, including a build that parks to ask a person and
  resumes when answered.
- **Every node kind is also a class** — `BaseNode`, `ReactNode`, `ToolNode`,
  `FunctionNode`, `PythonNode`, `RouterNode`, `LoopNode`, `MapNode`,
  `SubgraphNode`, `InputNode`, `OutputNode`, plus `ContainerNode` for the three
  that run a nested body. A second door into the same room:
  `GraphNode(kind="base", …)` is unchanged and still works, YAML on disk is
  untouched, and `Graph` upgrades whatever it is given — so `isinstance(node,
  RouterNode)` is true however the node was made. The engine dispatches on the
  class rather than on a kind string. The classes carry identity only; what a
  kind *requires* stays declared in `neurosurfer/graph/engine/kinds/` and read
  from there by the validator. Every name carries the `Node` suffix because the
  bare ones were not sayable at a call site — `Tool`, `Input`, `Output`, `Map`,
  `Function` and `Python` all already mean something else here, and `Tool` was
  an outright collision with `neurosurfer.tools.base.Tool`, the ABC every
  registered tool subclasses.


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

### Changed

- **A declared input no step reads now blocks the workflow.**
  `declared_inputs_are_read_by_something` was a warning, so a workflow that
  accepts a parameter and ignores it registered and ran, and the backstop was a
  human noticing the answer had nothing to do with what they passed. That is not
  a backstop a workflow the Architect builds and verifies on its own ever gets:
  `gpt-5-mini`'s first build of a ticket-routing intent declared `ticket_text`,
  named it in none of its five steps, and would have registered. A model does not
  act on a warning it is allowed to ignore; as an error the repair loop has to
  fix it before the build can register.

  It **downgrades itself to a warning when it cannot see** — a step whose
  parameters this cannot read (a `tool` node, whose arguments live in a
  registered schema, or a callable that will not import or inspect) hides the
  very reads that would clear the input, and refusing to run over a fact that was
  never established is worse than the gap.

  Promoting it exposed four false positives it had been reporting quietly all
  along, each of which would now have blocked a working graph, and all four are
  fixed: an **output node's `value`** was never scanned for placeholders, so
  `value: "hello {who}"` read nothing; a **`dict`-mode input step**, whose whole
  job is collecting the declared inputs, was judged not to read them; a **text
  input step's own key** was not counted as a read of the input that lands on it;
  and a **`tool` node** is handed the inputs mapping as kwargs, which is now
  doubt rather than silence. If a workflow of yours stops registering, it is
  telling you a parameter it accepts goes nowhere — name it in a step as
  `{name}`, or drop it from `inputs`.

- **A loop has one stop condition, `until`, and `break_when` is gone.** Asking an
  author to pick a *mechanism* — plain English or sandboxed expression — was
  asking the wrong question; a loop stops for one reason, so it gets one field.
  `until` is read as whichever of two things it is: a **function** (a callable,
  or the name of one in the graph's new `functions:` file), which receives a
  `LoopIteration` and returns `True` to stop or `(stop, "reason")` to also set
  the next iteration's `{feedback}`; or **plain English**, judged by an internal
  LLM decision each iteration. The function form subsumes the expression it
  replaces and goes well past it — it gets the body's whole
  `GraphExecutionResult` plus the iteration index, history and loop vars, where
  an expression could only reach the parent state through a namespace, and it is
  real Python rather than a restricted evaluator. Which form a string is, is a
  **lookup, not a heuristic**: a name the sidecar defines is the function,
  anything else is prose. A graph that declares a sidecar and names something
  absent from it is an error rather than a prompt, so a typo cannot quietly
  become an English condition sent to a model.
- **Graphs can carry a Python sidecar: `functions: helpers.py`.** Named at the top
  of the graph, resolved relative to the graph file, and copied with the package
  on export — the same self-containment `nodes/<id>.py` already gives `function`
  nodes, but declared once so YAML can then use bare names. A sidecar is imported
  by path and therefore stands alone: no relative imports.
- **The loop's exit judge has a third verdict, UNRELATED.** A plain-English
  condition about a different subject than the body produces — "stop when winter
  is here" over a body writing coffee taglines — can never be satisfied, so
  CONTINUE would be a lie that costs the full ceiling, every iteration plus a
  judge call each, to tell. The loop stops, logs why, and still returns the work
  it did, with `structured_output["stopped_reason"] == "condition_unrelated"`.
  It rides on the call already being made, so noticing costs nothing, and it is
  asked *after* an iteration so the judge has real output rather than a
  description of intent. Deliberately narrow: merely demanding, vague or not-yet
  conditions are CONTINUE, and an unparseable verdict fails safe to CONTINUE —
  a judge that could not be read must not be what stops a loop.
- **Built-in tools moved** from `neurosurfer/tools/builtin/` to
  `neurosurfer/registry/core/<domain>/`. `from neurosurfer.tools.builtin import
  ReadFileTool` still works — the package re-exports every tool — but **importing
  a tool by submodule path does not**: `from neurosurfer.tools.builtin.search
  import SearchTool` is now `from neurosurfer.registry.core.filesystem.search
  import SearchTool`.
- **MCP server configs moved, and an existing one will not be found.**
  `McpStore.default()` was `~/.neurosurfer/mcp.json`; it is now
  `mcp_config_path()` — `config/mcp.json` under the data root, i.e.
  `./.neurosurfer/config/mcp.json` unless `NEUROSURFER_HOME` is set. Two
  consequences, and **neither raises**: the old file is still on disk and nothing
  reads it, so the server list reads as *empty*; and the location is now relative
  to the **working directory** rather than to `$HOME`, so launching from a
  different folder yields a different, empty configuration. Move the file and set
  `NEUROSURFER_HOME` to restore host-wide behaviour — see
  [Upgrading](docs/about/upgrading.md). The move itself is right: it puts MCP
  config under the same single root as workflows, runs, traces and authored tools
  rather than leaving one file behind in `$HOME`. Only the silence was wrong.
- **A node is told what it names, and nothing ambient.** *This is the change most
  likely to affect an existing workflow.* A node's turn is now **what its own
  task text names, plus the outputs of the steps it declared in `depends_on`** —
  and that is all. The block that recited every graph input underneath each
  node's instructions is gone.

  It could not be made correct, only less wrong: a `map` body was handed the
  whole collection it was iterating, once per item, beside the single item it was
  working on; a value the instruction had already interpolated was printed again
  underneath it. And any rule for "which inputs matter to this node" is a worse
  version of one the author already wrote, in the placeholders of the instruction.

  **What to check in your own workflows:** a step that needs a graph input must
  interpolate it — `goal="Research {topic}…"`. A step that reads its input only
  because the engine used to recite it will now run, go green, and answer as
  though it had been passed nothing. `validate_package` reports a declared input
  that no step names; it cannot see an input the graph never declared.

- **A node's system prompt is identical for every node of every graph.** The task
  moved out of it and into the user turn. Rendering the task into the system
  prompt made it differ per node — and inside a `map`, per item — so **prompt
  caching could never fire**, a fifty-item map sending fifty distinct system
  prompts. This also matches the convention every provider documents.
- **Validation is the first step of every run**, not only of registration. A
  graph that cannot run is refused before a model is called rather than partway
  through.
- **A run handed a value no step reads now says so.** Passing
  `{"user_intent": …}` to a graph that declares no inputs and interpolates
  nothing used to be accepted in silence, and the run would go green while the
  model answered "please provide the topic". It logs a warning naming the key
  and the fix. Deliberately quiet where it cannot know: a `function`, `python`
  or `tool` node is handed the whole mapping as keyword arguments, so any key
  could be the one it takes, and their presence silences the check rather than
  risk crying wolf on a working graph.
- **The executor is a package.** `neurosurfer/graph/engine/executor.py` is now
  `executor/` (scheduler, iteration, routing, deterministic kinds, io, llm).
  `from neurosurfer.graph.engine.executor import GraphExecutor` is unchanged;
  reaching into the old module's internals by path is not.

### Fixed

- **The Architect gave a different *kind* of answer on different models.**
  `gpt-5.1` turned "summarise an article and write a title" into
  `WorkflowInfeasible` while `gpt-5-mini` and a local 9B built it — and the cause
  was structural, not a quirk of one model. The model **writes the acceptance
  criteria it is then judged against**, the judge fails closed, and a more capable
  model writes a stricter bar: it derived "no information not present in the
  source", which no reading of one output can certify and no prompt can promise.
  With an unbounded repair loop and an unguarded `declare_blocked`, its only exits
  were grind forever or give up. Four changes, each closing one of those:

    - **`declare_blocked` is gated.** It is for a capability nothing can provide —
      a missing integration, a credential nobody supplied, an unsafe or
      contradictory request. A complete, valid, fully grounded design is never
      infeasible, and the tool now refuses to say it is and names the alternative.
      The rule was in the system prompt, and prose is the one place a model can
      ignore.
    - **The repair loop is bounded.** After `max_verification_attempts` (3) judged
      failures the workflow registers **with the loudest caveat in the codebase**
      rather than not existing — the same trade already made for a broken test
      rig. Structural gates are unaffected: an invalid graph stays unregisterable
      however many attempts were spent.
    - **Unfalsifiable acceptance criteria are dropped before the run.** A
      criterion demanding a *guarantee* or the absence of something unstated is
      not a test. Deliberately narrow — "exactly three sentences", "no more than
      200 words" and "does not include the raw table" all survive; only unprovable
      absences go.
    - **A run that stops short hands over its work.** If the loop ends with no
      terminal state but the design passes every gate, it registers instead of
      raising. Found by the new matrix on a 9B that built a good workflow and then
      spent its remaining turns editing an output node.

- **Models that refuse function tools at their default reasoning effort now
  work.** The newest OpenAI reasoning models 400 on chat-completions the moment a
  request carries `tools` unless `reasoning_effort` is `"none"`. We never sent the
  parameter, so `gpt-5.6-terra` could not run the Architect at all. The provider
  now recognises that one error, retries with `reasoning_effort="none"`, and
  remembers it for the rest of the session — learned at runtime rather than from a
  model-name list, because such a list is wrong the week after it is written. The
  trade is explicit in the warning it logs: tool calling works, reasoning does
  not, and having both needs the Responses API.

- **A fix made after `register_workflow` reached nothing.** `register()` snapshots
  the staged package into the registry, so anything edited afterwards lived only
  in the session — and with `review_mode="warn"` the design review reports its
  findings *after* the package is written. A real transcript: the reviewer said a
  node titled the article where the request asked for a title of the summary, the
  tool invited a fix, the model patched the node, and the registered `graph.yaml`
  still carried the flaw. The review was advice nobody could act on.

  Two halves. `BuildSession.sync_registration()` re-saves when the design has
  moved on since the write — fingerprint-compared, so an unchanged design writes
  nothing — and `build()` calls it on the terminal path. And the tool message no
  longer contradicts itself: it used to say *"The build is complete — you may
  finish now"* and *"consider fixing and re-registering"* in the same breath, and
  a small model takes the shorter road. With a review finding, the "you may
  finish" half is dropped.

- **`remove_node` said nothing**, so a node added, removed and added again read as
  "added twice" in the build log — a model thrashing and a model repeating itself
  looked identical. It narrates like every other mutating tool now.

- **`InMemoryVectorStore` can be instantiated.** It was exported by name and
  recommended by `docs/guides/rag.md`, and never implemented `delete_documents`
  — so the abstract base refused to construct it. Nothing in the package or the
  tests ever built one, which is exactly why it survived. It was also
  non-conformant once it could be built: `add_documents` appended rather than
  upserting, and both `similarity_search` and `list_all_documents` accepted a
  `metadata_filter` and ignored it.
- **Chroma scores were wrong on every collection it created.** Chroma's default
  space is squared L2 and the code returned `1.0 - distance` as though it were
  cosine, so orthogonal vectors scored **-1.0** instead of `0.0` and any
  `similarity_threshold` was applied to the wrong scale. New collections are
  created as cosine; existing ones are read for the space they actually have and
  converted, so an old store reports correct scores without being rebuilt.
- **Chroma's `list_all_documents` returned unusable ids.** It minted a fresh
  `uuid4()` per row, so the ids it handed back matched nothing and
  `delete_documents` on them was a silent no-op.
- **`RAGAgent` could only ever use sentence-transformers.** It built
  `_LocalEmbedder(spec)` directly, so a *name* meant one backend however it was
  written; it now routes through the embeddings registry.
- **The gateway refuses `--workers N` instead of corrupting run history.** The
  workflow run store is per-process, so runs created on one worker were invisible
  to the others and `GET /v1/runs/{id}` answered from whichever process took the
  request.
- **`README.md` claimed a memory tool** that does not exist, and undercounted
  the tools and providers that do.
- **A `react` node whose turn is all reasoning is no longer a failed node.**
  `RunResult.final_text` accumulates `TextDelta`, so a local reasoning model that
  ends a turn having emitted only `ThinkingDelta` — no tool call, no text — left
  it empty, and the node was reported as having "finished without producing an
  answer", taking every node downstream with it. `CanonicalResponse.text()`
  already falls back to thinking for the one-shot path; the streamed path
  disagreed purely because it accumulates deltas. `RunResult` now also carries
  `final_thinking` — a separate channel, because `TextDelta` is the answer and
  `ThinkingDelta` is reasoning and concatenating them would hand a caller
  reasoning labelled as an answer — and `run_react_node` uses it last, after
  `report` and `final_text`. Measured on `qwen/qwen3.5-9b` driving the capstone
  tutorial's vision node: two failed runs in six before, none in five after.
- **A code node's parameters count as reading a graph input.** `function` and
  `python` nodes are called `fn(**{**graph_inputs, **dependency_results,
  **scope})`, so a parameter named `db_path` reads `db_path` — the same argument
  that already exempted `tool` nodes. Judging one kind by its parameters and the
  other by its templates made
  `declared_inputs_are_read_by_something` report the capstone tutorial as
  ignoring two inputs its functions consume on every run. A callable declaring
  `**kwargs` reads whatever it is handed, so the rule stays silent entirely.
- **Naming an image in a long prompt no longer kills the run.** Every user turn
  is scanned for image paths so a prompt like "explain /tmp/chart.png" attaches
  the image without routing through `read_file` first. The scan tries the
  *longest* candidate first — which, for a graph node's turn, is the whole task
  text up to the extension. That is past `NAME_MAX`, and `Path.is_file()` only
  swallows `ENOENT`/`ENOTDIR`/`EBADF`/`ELOOP`, so the `ENAMETOOLONG` escaped to
  the caller: a `react` node with a dashboard path in its context died in 0 ms,
  before the model was asked anything, and every node downstream was skipped.
  A candidate that cannot even be *asked* about is now simply not a file.
- **Trace export never runs on the agent's thread.** Every exporter hook, and
  the `flush()` at each run finish, ran inline on whatever thread the agent was
  on — so a run waited on the monitoring backend's network. `flush()` is not the
  cheap thing its name suggests: OpenTelemetry's `BatchSpanProcessor` already
  owns a queue and a worker, and `force_flush` exists to *bypass* them and drain
  on the caller. All exporter calls now go to a single daemon worker over a
  bounded FIFO (`neurosurfer/observability/dispatch.py`); the agent thread
  enqueues and returns. With tracing on and no collector listening, nine spans
  blocked the run for **0.116s instead of 74s**. Order is preserved (one worker,
  FIFO), which also serialises exporter state that concurrent `map` bodies used
  to mutate from several threads at once. Delivery is best-effort by design: the
  queue is bounded and drops rather than growing without limit, and an `atexit`
  drain lets a script that ends right after a run still ship what it has. Tests
  and shutdown paths that need to observe delivery can call
  `neurosurfer.observability.dispatch.drain()`.
- **An exporter that is named but not configured is skipped, not built.**
  `NEUROSURFER_EXPORTERS=otel` with no `OTEL_EXPORTER_OTLP_ENDPOINT` set built
  the exporter anyway, and the OTel SDK filled in its own
  `http://localhost:4318` — so an install that had pointed at no collector still
  opened one, and paid to find out nothing was there. Auto-detection never had
  this problem: it turns `otel` on *because* the endpoint is set. The explicit
  list now applies the same requirement and warns what is missing. Passing a
  constructed instance to `register_exporter` still bypasses the check, since
  that is a deliberate choice by the caller.
- **An unreachable OTLP collector no longer costs ~8s per node, and no longer
  prints a traceback per batch.** With tracing on and nothing listening on the
  endpoint, two things went wrong. It was *loud*: `OTLPSpanExporter.export`
  re-raised the transport error and `BatchSpanProcessor` logged it with
  `logger.exception`, so a workflow's own output disappeared under stacks whose
  frames all named `urllib3` and none named neurosurfer. And it was *slow*,
  which mattered more: `force_flush` is a blocking export on the calling thread
  and runs at every run finish, and on Windows a connect to a closed port is not
  refused instantly the way it is on Linux — ~2s of SYN retry, doubled because
  `localhost` resolves to both `::1` and `127.0.0.1`, doubled again by the
  exporter's blind retry. **8.2s per flush, measured**; tutorial 03's router
  cell took 74s against a model answering in under two. The exporter is now
  wrapped so the first failure logs one warning — naming the endpoint, the root
  cause, what the attempt cost, and both ways to stop it — and then **disables
  tracing for the session**, so later spans are dropped without touching the
  network. Same misconfiguration on Linux was always ~0ms, which is why this
  only ever showed up on Windows. A reachable collector is unaffected.
- **A `base` step cut off mid-plan no longer reports success.** It gets one round
  of tool calls; asked to fetch a page and then write a file, it spent the round
  on the fetch, was refused the second, and returned an empty answer that the run
  recorded as a success. It now fails, naming the tools it did call and pointing
  at `react`. A truncated step that still produced text keeps it.
- **Trace step ids are unique under concurrency.** A `map` node runs its body
  concurrently and step ids came from an unguarded counter, so two steps could
  share one. Steps also carry `node_id`, which is what lets a reader group a run
  into a per-node tree.
- **A validator no longer scolds a correct two-step workflow.** The "fewer than
  three LLM nodes is almost certainly under-designed" heuristic sat in the gate
  that decides whether a package may register, and has been removed.
- **A `react` node that ends with `finish()` returns what it finished with.** It
  returned the loop's last assistant text instead, so the one kind of node that
  states its answer explicitly was the one whose answer was discarded.
- **A bound argument reaches an agent node again.** A `base` or `react` node
  carrying `tool_args` or `tool_settings` failed with `AttributeError:
  'GraphExecutor' object has no attribute '_render_tool_args'` — the two helpers
  became module-level functions when the executor was split and the call sites
  were not moved with them. This is the only path by which a credential reaches
  a tool without passing through a prompt, and it now has offline coverage.
- **A router's branches are checked.** A `routes` target that names a node which
  does not exist, or one that does not list the router in `depends_on`, is
  reported instead of pruning the whole branch at run time.
- **An `.env` asking for OpenAI gets OpenAI.** `LLM_PROVIDER=openai` could be
  overridden by a stored provider profile, so a notebook demonstrated a model
  other than the one its own configuration named.


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
