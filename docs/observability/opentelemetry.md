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

### When there is no collector listening

The first export failure reports itself and then turns the exporter off for the session:

```
Trace exporter 'otel': the OTLP collector at http://localhost:4318/v1/traces is not
reachable (ConnectionRefusedError: …). That attempt blocked the run for 8.2s, so
tracing is now disabled for this session and later spans are dropped without touching
the network. Point OTEL_EXPORTER_OTLP_ENDPOINT at a running collector, or set
NEUROSURFER_EXPORTERS=none to keep it off from the start.
```

The run itself is unaffected — spans are dropped, nothing raises. Because the exporter stays
off once disabled, a collector started *after* a run begins will not pick it up; restart the
process.

"Failure" here means any of three things, because the OTLP exporter has reported it three ways
across versions: it raised, it returned a failed batch, or it simply took more than two seconds.
Older releases only watched for the exception — and when
`opentelemetry-exporter-otlp-proto-http` 1.44 started retrying internally and *returning* failure
instead, the exporter was never disabled and every flush paid the full retry schedule again, on
Linux as well as Windows. A local collector answers in milliseconds; anything near a second has
been out on the network.

!!! warning "On Windows, a missing collector is expensive — once"
    Exporters are flushed at every run finish, and `force_flush` blocks the calling thread.
    Linux refuses a connection to a closed port immediately; Windows retries the SYN for ~2s,
    and `localhost` resolves to **both** `::1` and `127.0.0.1`, so a single attempt costs ~4s
    and the OTLP exporter's built-in retry doubles it to **~8.2s**. That is why the exporter
    disables itself rather than retrying per batch — otherwise every node in a workflow pays it.
    If you do not want tracing at all, set `NEUROSURFER_EXPORTERS=none` and it costs nothing.

If you see this and did not expect tracing to be on at all, check `NEUROSURFER_EXPORTERS`: an
explicit value **overrides** auto-detection, so `otel` listed there turns the exporter on even
with no endpoint set, and the OTel SDK then falls back to its own default of
`http://localhost:4318`.

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
