# Langfuse

[Langfuse](https://langfuse.com) is the batteries-included backend: traces, automatic token **cost**,
sessions, and evals in a purpose-built LLM UI. It's the recommended default.

## Turn it on

Set the credentials and Langfuse activates on the next run — no code change:

```bash
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."
export LANGFUSE_HOST="https://cloud.langfuse.com"   # region or self-hosted URL
```

!!! tip "Pick the right region"
    Langfuse Cloud runs **EU** (`https://cloud.langfuse.com`) and **US**
    (`https://us.cloud.langfuse.com`) as separate instances. Keys from one region return
    `401 Unauthorized` against the other — set `LANGFUSE_HOST` to match where your project lives.

## Self-hosting

Point `LANGFUSE_HOST` at your own instance to keep traces local — no cloud, no per-project dashboard
lag:

```bash
export LANGFUSE_HOST="http://localhost:3000"
```

[Deployment](../server/deployment.md) shows how to run the gateway alongside a self-hosted Langfuse
stack (web, worker, Postgres, ClickHouse, Redis, MinIO). Services reach it over the internal Docker
network, so you never expose Langfuse publicly.

## What a trace contains

| Level | Comes from | Shows |
|---|---|---|
| **Trace** | the run | name, session, input, output, status |
| **Generation** | each LLM turn | model, token usage → **cost**, input, output |
| **Span** | each tool call | tool name, args, result, error status |
| **Event** | mode change / compaction | a point-in-time marker |

**What a generation shows.** Each LLM turn records its **input** — the messages sent to the model
that turn (image payloads elided to keep traces small) — and its **output**: the model's *full*
assistant response, i.e. `thinking` + `text` + `tool_use` blocks, not just the answer text. So a
turn's own reasoning and tool calls appear on that turn, not only in the next turn's history.

## Sessions & trace names

- **`session_id=`** groups related runs into one Langfuse **session** — the CLI sets one per
  conversation (reset on `/clear`), so a multi-message chat is a single session instead of N stray
  traces.
- **`trace_name=`** labels a run's trace (default is `<AgentType>.run`). The CLI names its runs
  `neurosurfer-cli`, making CLI traces easy to tell apart from SDK or graph runs.

```python
agent = AgenticLoop(
    provider=provider, tools=tools, system_prompt="…",
    guardrails=guardrails, io=io, cwd=cwd,
    session_id="chat-42",          # group these runs
    trace_name="support-bot",       # label the trace
)
```

## Troubleshooting

!!! warning "Recent traces seem to lag ~10 minutes in the UI"
    That's usually the Langfuse **dashboard** "Fast" toggle, which reads a cached view — not your
    exporter. Turn it off to see fresh traces immediately. Your run flushes at the end, so the trace
    is sent promptly; the delay is UI-side, not in Neurosurfer.

- **`401 Unauthorized`** — wrong region (`LANGFUSE_HOST`) for your keys, or a typo in a key.
- **Nothing appears** — confirm both `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set
  (Langfuse activates only when *both* are present), and that `NEUROSURFER_EXPORTERS` isn't `none`.

## Next

- [OpenTelemetry](opentelemetry.md) — send to Honeycomb/Phoenix/Grafana instead of (or alongside)
  Langfuse.
- [Custom Exporters](custom-exporters.md) — build your own backend adapter.
