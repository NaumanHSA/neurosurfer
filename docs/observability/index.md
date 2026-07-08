# Observability

Neurosurfer can ship **every agent run** to an external monitoring backend so you can see and debug
what your agents do — the LLM turns, the tool calls, token usage, and cost — in a real UI. It's a
cross-cutting layer: turn it on with environment variables and **no code changes**.

Two backends ship in the box:

- **[Langfuse](langfuse.md)** — batteries-included LLM observability (traces, token cost, sessions,
  evals). The best out-of-the-box experience.
- **[OpenTelemetry](opentelemetry.md)** — the vendor-neutral standard. Emits GenAI-convention spans
  over OTLP, so **any** OTel backend ingests them (Honeycomb, Arize Phoenix, Grafana Tempo, Datadog…).

Or write your own — see [Custom Exporters](custom-exporters.md).

```bash
pip install "neurosurfer[observability]"
```

## How it works

Every agent already yields a stream of typed [events](../learn/concepts.md#events) (`ToolStarted`,
`TurnCompleted`, …). Tracing attaches a **side-channel observer** to that stream — it *observes,
never consumes* — and translates it into backend calls. Nothing about how you consume `agent.run(...)`
changes. The mapping:

| Agent activity | Trace observation |
| --- | --- |
| a run | a **trace** (the root) |
| one LLM turn | a **generation** — model + input/output tokens ⇒ cost |
| a tool call (start → finish) | a **span** |
| a spawned **sub-agent** | a nested **span** under the parent (same trace) |
| a **workflow** node | a nested span (`workflow → node → agent → tool`) |
| mode change / context compaction | a trace **event** |

**Nesting is automatic.** A run started *inside* another run — a spawned [sub-agent](../guides/subagents.md)
or a [workflow](../guides/graph-workflows.md) node — inherits the active trace context and nests under
it, so one trace shows `parent → child → tool` instead of disconnected top-level traces. This works
across `await` and parallel `asyncio.gather` spawns alike.

## Enablement

Tracing is **auto-on from the environment** — set a backend's connection vars and it activates on the
next run. No code change.

| Set this | Turns on |
|---|---|
| `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY` | [Langfuse](langfuse.md) |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | [OpenTelemetry](opentelemetry.md) |

Both can be active at once. Override detection with `NEUROSURFER_EXPORTERS=langfuse,otel`, or force
everything off with `NEUROSURFER_EXPORTERS=none`. For explicit control in code, use
`configure_exporters([...])` — see [Custom Exporters](custom-exporters.md#choosing-exporters-in-code).

## Guarantees

- **Zero overhead when off.** With no backend configured, no observer is even created.
- **Never breaks a run.** A misbehaving or unreachable exporter is isolated — its errors are
  swallowed, the agent run is unaffected.
- **Optional dependency.** A base install (without the `observability` extra) resolves to no
  exporters; nothing to import, nothing to fail.

!!! note "Two tracing subsystems — don't conflate them"
    `neurosurfer.observability.exporters` is what this section covers: **trace exporters** (Langfuse,
    OTel) that ship agent runs to an external backend. Separately, `neurosurfer.tracing` is a vendored
    **span tracer** for local console/step tracing of workflows. Different purposes — when you want a
    dashboard, it's the exporters.

## In this section

- **[Langfuse](langfuse.md)** — setup, cloud regions, self-hosting, sessions, and what each trace shows.
- **[OpenTelemetry](opentelemetry.md)** — OTLP, the span shape, and pointing at any backend (Honeycomb…).
- **[Custom Exporters](custom-exporters.md)** — the `TraceExporter` contract and writing your own.
