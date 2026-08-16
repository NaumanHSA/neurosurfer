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
or a [workflow](../graph/index.md) node — inherits the active trace context and nests under
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

## Export never runs on the agent's thread

Tracing is a side channel. It is allowed to be slow, to fail, and to lose data; it is **not**
allowed to make a run slower.

Every exporter hook and every flush is submitted to a **single daemon worker over a bounded FIFO**.
The agent thread enqueues and returns.

This matters more than it sounds. Exporter calls used to run inline, and `flush()` is not the cheap
thing its name suggests — OpenTelemetry's `BatchSpanProcessor` already owns a queue and a worker,
and `force_flush` exists to *bypass* them and drain on the caller. With tracing on and no collector
listening, nine spans blocked a run for **0.116s instead of 74s**.

### What it guarantees

- **Order is preserved.** One worker draining a FIFO means `on_run_start` really does reach an
  exporter before the `on_turn` that follows it.
- **Exporter state stops being racy.** A `map` node runs its body concurrently, so several agent
  threads used to call into the same exporter's state at once. Everything now lands on one thread.

### What it does not

- **Delivery is best-effort by design.** The queue is bounded and **drops** rather than growing
  without limit — a process that dies of its own monitoring is worse than one missing spans.
- An `atexit` drain lets a script that ends right after a run still ship what it has.

```python
from neurosurfer.observability import dispatch

dispatch.drain(timeout=5.0)   # tests and shutdown paths that must observe delivery
dispatch.dropped()            # how many items the bounded queue discarded
```

## Guarantees

- **Zero overhead when off.** With no backend configured, no observer is even created.
- **Never breaks a run.** A misbehaving or unreachable exporter is isolated — its errors are
  swallowed, the agent run is unaffected.
- **An exporter named but not configured is skipped, not built.** `NEUROSURFER_EXPORTERS=otel` with
  no endpoint set used to build the exporter anyway, and the OTel SDK filled in its own
  `http://localhost:4318` — so an install pointing at no collector still opened one. The explicit
  list now applies the same requirement auto-detection does, and warns what is missing. Passing a
  constructed instance to `register_exporter` still bypasses the check, since that is a deliberate
  choice by the caller.
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
