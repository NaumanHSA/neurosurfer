# Architecture

Neurosurfer is a set of layers you can adopt independently. Use a bare provider for one call, an
agent for multi-step tool use, a graph for orchestration, or the gateway to serve any of them behind
an OpenAI-compatible API.

## The layers

```
┌──────────────────────────────────────────────────────────────┐
│  Gateway (app/server)   OpenAI-compatible /v1/chat/completions │
│                         serve agents · proxy upstreams · hooks │
├──────────────────────────────────────────────────────────────┤
│  Orchestration          Graph engine (DAG) · Workflows         │
│  (graph, architect)     Architect (build a workflow from text) │
├──────────────────────────────────────────────────────────────┤
│  Agents                 AgenticLoop · ReactAgent · Agent       │
│  (agents)               sub-agents · context mgmt · guardrails │
├───────────────┬──────────────────────┬───────────────────────┤
│  Tools        │  RAG                 │  Observability          │
│  (tools, mcp) │  (rag, vectorstores) │  (observability)        │
├───────────────┴──────────────────────┴───────────────────────┤
│  Providers (llm)        Anthropic · OpenAI / OpenAI-compatible │
│                         one Provider protocol, canonical types │
└──────────────────────────────────────────────────────────────┘
```

- **Providers** ([`neurosurfer.llm`](../guides/providers.md)) normalise every model behind one
  `Provider` protocol and a canonical message/response model, so nothing above cares which vendor you
  use.
- **Agents** ([`neurosurfer.agents`](../guides/agents.md)) turn a provider + tools into a run loop
  that streams typed events, gates dangerous actions, and manages the context window.
- **Tools** ([`neurosurfer.tools`](../guides/tools.md)) are what an agent can *do* — file ops, shell,
  web search, sandboxed Python, HTTP, browser — plus [MCP](../guides/mcp.md) tools from external
  servers.
- **RAG** ([`neurosurfer.rag`](../guides/rag.md)) adds ingest → chunk → embed → retrieve with
  token-aware context injection, backed by pluggable vector stores.
- **Orchestration** ([`neurosurfer.graph`](../guides/graph-workflows.md)) runs multi-node DAGs of
  functions, tools, and agents; the [Architect](../architect/index.md) designs those graphs from a
  plain-English description.
- **Gateway** ([`neurosurfer.app.server`](../server/index.md)) exposes any of the above as a model at
  `/v1/chat/completions`, with SSE streaming, upstream proxying, and request/response hooks.
- **Observability** ([`neurosurfer.observability`](../observability/index.md)) is a cross-cutting
  layer: every agent run emits a trace to Langfuse or any OTLP backend with **zero code change**.

## How a request flows

1. You build a **provider** (from env/profile) and a **tool pool**.
2. You construct an **agent** and call `run(prompt)` — an async generator of
   [events](concepts.md#events).
3. Each turn the agent streams the model, and on tool calls it **gates** them through guardrails and
   the `io` handler, executes, and feeds results back — until the model finishes.
4. If tracing is enabled, a side-channel observer maps those events to a trace **without touching**
   the agent.
5. Behind the gateway, that same run is wrapped as an OpenAI `/v1/chat/completions` response.

See [Core Concepts](concepts.md) for the canonical types and the event lifecycle, and
[Permissions & Safety](permissions.md) for how gating works.
