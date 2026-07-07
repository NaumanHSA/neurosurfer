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
| mode change / context compaction | a trace **event** |

## Turn it on (zero code)

Tracing is **auto-on from the environment** — set the backend's connection vars and
it activates on the next run. No code change.

### Langfuse

```bash
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."
export LANGFUSE_HOST="https://cloud.langfuse.com"   # or your self-hosted URL
```

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
    Standalone agent runs are fully traced today. Nesting a multi-node
    [workflow](graph-workflows.md) into a single trace (node → agent → tool) is on
    the roadmap.
