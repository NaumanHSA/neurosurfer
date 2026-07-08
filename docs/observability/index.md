# Observability

Neurosurfer can ship every agent run to an external monitoring backend so you can
**see and debug what your agents do** — the LLM turns, the tool calls, token usage
and cost — in a real UI. Two backends are supported:

- **[Langfuse](https://langfuse.com)** — batteries-included LLM observability
  (traces, token cost, sessions, evals). The best out-of-box experience.
- **[OpenTelemetry](https://opentelemetry.io)** — the vendor-neutral standard.
  Emits GenAI-semantic-convention spans over OTLP, so **any** OTel backend ingests
  them: Arize Phoenix, Grafana Tempo, Datadog, Honeycomb — or Langfuse's own OTLP endpoint.

Install the extra:

```bash
pip install "neurosurfer[observability]"
```

## How it works

Every agent already yields a stream of typed events (`ToolStarted`, `ToolFinished`,
`TurnCompleted`, …). Tracing attaches a **side-channel observer** to that stream —
it observes, never consumes — and translates it into backend calls. Nothing about
how you consume `agent.run(...)` changes; the mapping is:

| Agent activity | Trace observation |
| --- | --- |
| a run | a **trace** (the root) |
| one LLM turn | a **generation** — model + input/output tokens ⇒ cost |
| a tool call (start→finish) | a **span** |
| a spawned **sub-agent** | a nested **span** under the parent run (same trace) |
| mode change / context compaction | a trace **event** |

**Nesting.** A run started *inside* another run (a spawned sub-agent) automatically
nests under it — one trace shows `parent → sub-agent → tool`, not disconnected
top-level traces. This works because the active run publishes its trace context,
which sub-agents inherit (across `await` and parallel `asyncio.gather` spawns alike).

**Sessions.** Pass `session_id=` when constructing an agent and every run of that
agent groups under one Langfuse **session** — the CLI does this per conversation, so
a multi-message chat is one session (reset on `/clear`) instead of N stray traces.

## Turn it on (zero code)

Tracing is **auto-on from the environment** — set the backend's connection vars and
it activates on the next run. No code change.

### Langfuse

```bash
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."
export LANGFUSE_HOST="https://cloud.langfuse.com"   # or your self-hosted URL
```

!!! tip "Pick the right region"
    Langfuse Cloud runs **EU** (`https://cloud.langfuse.com`) and **US**
    (`https://us.cloud.langfuse.com`) as separate instances. Keys from one region
    return `401 Unauthorized` against the other — set `LANGFUSE_HOST` to match where
    your project lives.

### OpenTelemetry

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4318"
```

Both can be active at once. Now just run an agent — the trace appears in the UI:

```python
result = await agent.complete("What is the capital of France?")
```

Force everything off (even when keys are present) with
`NEUROSURFER_EXPORTERS=none`, or pin an explicit set with
`NEUROSURFER_EXPORTERS=langfuse,otel`.

## Turn it on (in code)

For explicit control, configure the registry before your first run:

```python
from neurosurfer.observability.exporters import configure_exporters

configure_exporters(["langfuse"], service_name="my-app")
```

Or register your own exporter instance — subclass
`neurosurfer.observability.exporters.base.TraceExporter` and override the hooks you
need (`on_run_start`, `on_turn`, `on_tool_start`, `on_tool_finish`, `on_run_finish`, …):

```python
from neurosurfer.observability.exporters import register_exporter
from neurosurfer.observability.exporters.base import MemoryExporter

mem = MemoryExporter()          # captures the lifecycle in-memory (great for tests)
register_exporter(mem)
```

## Guarantees

- **Zero overhead when off.** With no backend configured, no observer is created.
- **Never breaks a run.** A misbehaving or unreachable exporter is isolated — its
  errors are swallowed, the agent run is unaffected.
- **Optional dependency.** A base install (without the `observability` extra) simply
  resolves to no exporters; nothing to import, nothing to fail.

!!! note "Graph & workflow nesting"
    Agent runs — including spawned **sub-agents** — nest into a single trace today.
    Nesting a multi-node [workflow](graph-workflows.md) (graph node → agent → tool)
    under one trace is the remaining step on the roadmap; the same trace-context
    mechanism will carry it.
