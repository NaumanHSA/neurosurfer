# Custom Exporters

If neither Langfuse nor OpenTelemetry fits — you want to log to a file, push to an internal service,
or shape spans your own way — implement the **`TraceExporter`** contract. It's a handful of
no-op-by-default hooks; override only what you need.

## The contract

Every exporter receives an agent run's lifecycle as method calls, driven by the side-channel observer.
All hooks are keyword-only with defaults, so the contract can grow without breaking existing
exporters.

| Hook | Called when | Key arguments |
|---|---|---|
| `on_run_start` | a run begins | `name`, `input` |
| `on_turn` | one LLM turn completes | `usage`, `model`, `stop_reason`, `input`, `output` |
| `on_tool_start` | a tool call begins | `call_id`, `name`, `args` |
| `on_tool_finish` | a tool call returns | `call_id`, `name`, `result`, `is_error` |
| `on_event` | mode change / compaction | `kind`, `**data` |
| `on_error` | the run raised | `message` |
| `on_run_finish` | the run ends | `status`, `output` |
| `flush` | run end | — (push buffered data) |
| `close` | process teardown | — (release resources) |

Every hook also receives the run's `TraceContext` (first positional arg) — carrying `trace_id`,
`span_id`, `parent_span_id`, `session_id`, and `metadata`. **Tool calls pair by `call_id`** (present
on both `on_tool_start` and `on_tool_finish`), so keep your own `call_id → span` map.

## A minimal exporter

```python
from neurosurfer.observability.exporters.base import TraceExporter

class LogExporter(TraceExporter):
    name = "log"

    def on_run_start(self, ctx, *, name, input=None):
        print(f"[{ctx.trace_id}] run start: {name}")

    def on_turn(self, ctx, *, usage, model, stop_reason, input=None, output=None):
        print(f"[{ctx.trace_id}] turn: {model} "
              f"{usage.input_tokens}->{usage.output_tokens} ({stop_reason})")

    def on_tool_finish(self, ctx, *, call_id, name, result, is_error):
        print(f"[{ctx.trace_id}] tool {name}: {'error' if is_error else 'ok'}")

    def on_run_finish(self, ctx, *, status, output=None):
        print(f"[{ctx.trace_id}] run {status}")
```

## Registering it

Add an instance before your first run:

```python
from neurosurfer.observability.exporters import register_exporter

register_exporter(LogExporter())
```

The built-in **`MemoryExporter`** captures every lifecycle call in-memory — ideal for tests:

```python
from neurosurfer.observability.exporters.base import MemoryExporter

mem = MemoryExporter()
register_exporter(mem)
# ... run an agent ...
assert mem.hooks()[0] == "run_start"     # ordered hook names
turns = mem.of("turn")                    # payloads for a given hook
```

## Choosing exporters in code

Override environment auto-detection explicitly:

```python
from neurosurfer.observability.exporters import configure_exporters

configure_exporters(["langfuse"], service_name="my-app")   # replace the active set by name
```

- `configure_exporters(names, *, service_name=...)` — set the active exporters by name, replacing any
  prior state. Call before the first run.
- `register_exporter(instance)` — add an already-constructed exporter.
- `get_active_exporters()` — the resolved active set (from the environment on first use).
- `reset_exporters()` — clear resolved state (tests re-resolve next call).

## Isolation guarantees

The observer wraps every hook call in a guard: a **misbehaving or raising exporter never breaks the
agent run** — its error is swallowed and logged. A missing backend SDK is warn-and-skipped, so a
custom exporter can sit alongside Langfuse/OTel without either affecting the others. They share
nothing but the read-only `TraceContext`.

## Next

- [Overview](index.md) — how the whole tracing layer fits together.
- [Langfuse](langfuse.md) · [OpenTelemetry](opentelemetry.md) — the built-in backends.
