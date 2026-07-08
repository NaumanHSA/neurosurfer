# OpenTelemetry

[OpenTelemetry](https://opentelemetry.io) (OTel) is the vendor-neutral tracing standard. Neurosurfer's
`otel` exporter emits **GenAI-semantic-convention** spans over **OTLP/HTTP**, so you instrument once
and point at any compatible backend — only the endpoint and headers change.

## Turn it on

Set the standard OTel environment and the exporter activates:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4318"
export OTEL_SERVICE_NAME="neurosurfer"
```

The detector returns the exporter **type** (`otel`) — the *destination* is whatever endpoint you
point it at. There's no per-backend exporter; the backend is decided by the URL and auth headers.

## Send to a specific backend

### Honeycomb

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT="https://api.honeycomb.io"   # base URL; the exporter appends /v1/traces
export OTEL_EXPORTER_OTLP_HEADERS="x-honeycomb-team=YOUR_API_KEY"
export OTEL_SERVICE_NAME="neurosurfer"
```

Use `https://api.eu1.honeycomb.io` for the EU instance. Classic (32-hex) keys also need a dataset:
`x-honeycomb-team=YOUR_KEY,x-honeycomb-dataset=neurosurfer`.

### Others

The same pattern works for **Arize Phoenix**, **Grafana Tempo**, **Datadog**, **SigNoz**, **New
Relic**, or **Langfuse's own OTLP endpoint** — set `OTEL_EXPORTER_OTLP_ENDPOINT` (and headers where a
key is required). For a local collector, run one on `:4318` and point the endpoint at it.

!!! note "The exporter uses OTLP/HTTP"
    The http/protobuf exporter POSTs to `<endpoint>/v1/traces`, so give it the **base** URL, not the
    `/v1/traces` path. Auth headers come from `OTEL_EXPORTER_OTLP_HEADERS` (comma-separated
    `key=value`).

## The span shape

One trace per run, with GenAI-convention attributes any OTel backend renders:

```
<AgentType>.run                 root · gen_ai.operation.name = agent
├─ llm.turn                     gen_ai.operation.name = chat · gen_ai.usage.input_tokens / output_tokens
├─ tool.<name>                  gen_ai.operation.name = execute_tool · gen_ai.tool.name / arguments / result
└─ …
```

The exporter builds its **own** `TracerProvider`, so a host app that already configured OpenTelemetry
is never disturbed; spans are parented explicitly because a run's lifecycle crosses `await`
boundaries.

!!! tip "Cost vs. tokens"
    OTel backends get token counts on the `llm.turn` spans but don't compute LLM **cost** the way
    [Langfuse](langfuse.md) does. Run both exporters at once (`NEUROSURFER_EXPORTERS=langfuse,otel`) if
    you want cost in Langfuse and spans in your OTel stack.

## Next

- [Langfuse](langfuse.md) — the cost/eval-focused backend.
- [Custom Exporters](custom-exporters.md) — if neither backend fits, write your own.
