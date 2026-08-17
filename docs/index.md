# Neurosurfer

**Neurosurfer** is a Python framework for building AI agents — models that don't only answer
questions but *do* things: call tools, read files, search the web or your own documents, and work
through a task in several steps. Wire those steps into a graph yourself, run it against Anthropic,
OpenAI, Gemini, Bedrock or a model on your own machine, and serve any of it behind an
OpenAI-compatible API.

**The Architect** is the second way in: describe what you want in plain English and it builds the
workflow for you — working out which tools the job needs, running what it made to check it works,
and saying plainly when something can't be built rather than returning results it invented.

![Neurosurfer architecture](assets/diagrams/neurosurfer-architecture.png){ .architecture-diagram }


<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **Get started**

    ---

    Install the package and run your first agent in a few minutes.

    [:octicons-arrow-right-24: Quickstart](getting-started/quickstart.md)

-   :material-console:{ .lg .middle } **Interactive CLI Agent**

    ---

    Chat with an agent, manage provider profiles, wire up MCP servers, and build workflows — all
    from one REPL, no Python required.

    [:octicons-arrow-right-24: CLI Agent guide](cli/index.md)

-   :material-robot:{ .lg .middle } **Agents**

    ---

    `AgenticLoop`, `ReactAgent`, and one-shot `Agent` with streaming events.

    [:octicons-arrow-right-24: Agents guide](guides/agents.md)

-   :material-server-network:{ .lg .middle } **Gateway**

    ---

    Serve any agent behind an OpenAI-compatible `/v1/chat/completions` API.

    [:octicons-arrow-right-24: Server guide](server/index.md)

-   :material-graph:{ .lg .middle } **Graph & Workflows**

    ---

    Eleven node kinds, branching, loops, fan-out, and a validation gate — saved as runnable
    packages.

    [:octicons-arrow-right-24: Graph guide](graph/index.md)

-   :material-school:{ .lg .middle } **Tutorials**

    ---

    Hands-on Colab notebooks from install to a full capstone project.

    [:octicons-arrow-right-24: Tutorials](tutorials/index.md)

</div>

## What's in the box

- **Agent family** — `AgenticLoop` (native multi-step tool-use), `ReactAgent` (text-parsing ReAct
  for models without a native tool API), and `Agent` (one-shot, optionally with structured output).
- **Provider layer** — Anthropic Claude, OpenAI, and any OpenAI-compatible server (Ollama, LM
  Studio, vLLM, llama.cpp) behind one `Provider` protocol.
- **Tools** — 15+ built-in tools (web search, sandboxed Python, file ops, HTTP, headless browser,
  read-only SQL, and more) plus a simple framework for your own.
- **A tool registry** — every tool declares a capability tag from a closed vocabulary, so a need
  resolves against a declaration rather than against words a description happens to share.
- **RAG** — ingest → chunk → embed → retrieve → token-aware context injection.
- **Graph & Workflows** — a standalone DAG engine with eleven node kinds, branching, loops,
  fan-out, human-in-the-loop pauses, and a validation gate — saved as runnable Workflow packages.
- **Architect** — describe a workflow in plain English; it plans, grounds every capability against
  what exists, builds the graph, and **proves it by running it** — or refuses with a reason.
- **MCP client** — connect external Model Context Protocol servers, or **discover and install** one
  from the registry when a capability is missing.
- **OpenAI-compatible gateway** — `/v1/models` + `/v1/chat/completions` with SSE streaming, upstream
  proxying, native-agent backends, and request/response hooks — plus workflow and architect APIs.
- **Interactive CLI agent** — a full REPL for chat, provider profiles, MCP servers, and workflow
  building, plus a `serve` command for the gateway.

## Install

```bash
pip install -U neurosurfer
# with web search + gateway:
pip install -U "neurosurfer[search,serve]"
```

See [Installation](getting-started/installation.md) for the full list of optional extras, and [Upgrading](about/upgrading.md) if you are moving from an earlier release.

Neurosurfer is a set of layers you can adopt independently. Use a bare provider for one
call, an agent for multi-step tool use, a graph for orchestration, or the gateway to serve
any of them behind an OpenAI-compatible API.

## The layers

- **Providers** ([`neurosurfer.llm`](guides/providers.md)) normalise every model behind one
  `Provider` protocol and a canonical message/response model, so nothing above cares which vendor you
  use.
- **Agents** ([`neurosurfer.agents`](guides/agents.md)) turn a provider + tools into a run loop
  that streams typed events, gates dangerous actions, and manages the context window.
- **Tools** ([`neurosurfer.tools`](guides/tools.md)) are what an agent can *do* — file ops, shell,
  web search, sandboxed Python, HTTP, browser, SQL — plus [MCP](guides/mcp.md) tools from
  external servers.
- **The registry** ([`neurosurfer.registry`](guides/tool-registry.md)) is what makes those tools
  *findable*: each declares a capability tag from a closed vocabulary, so a need resolves against a
  declaration rather than against words a description happens to share.
- **RAG** ([`neurosurfer.rag`](guides/rag.md)) adds ingest → chunk → embed → retrieve with
  token-aware context injection, backed by pluggable vector stores.
- **Orchestration** ([`neurosurfer.graph`](graph/index.md)) runs multi-node DAGs of functions,
  tools, and agents — with branching, loops, fan-out, and a validation gate that refuses a graph
  before a model is called.
- **Authoring** ([`neurosurfer.architect`](architect/index.md)) designs those graphs from a
  plain-English description, grounds every capability it names against the registry, and proves its
  work by running it. The runtime never imports the authoring layer.
- **Gateway** ([`neurosurfer.app.server`](server/index.md)) exposes any of the above as a model at
  `/v1/chat/completions`, with SSE streaming, upstream proxying, and request/response hooks — plus
  the [workflow](server/workflows-api.md) and [architect](server/architect-api.md) APIs.
- **Observability** ([`neurosurfer.observability`](observability/index.md)) is a cross-cutting
  layer: every agent run emits a trace to Langfuse or any OTLP backend with **zero code change**.

## How a request flows

1. You build a **provider** (from env/profile) and a **tool pool**.
2. You construct an **agent** and call `run(prompt)` — an async generator of
   [events](learn/concepts.md#events).
3. Each turn the agent streams the model, and on tool calls it **gates** them through guardrails and
   the `io` handler, executes, and feeds results back — until the model finishes.
4. If tracing is enabled, a side-channel observer maps those events to a trace **without touching**
   the agent.
5. Behind the gateway, that same run is wrapped as an OpenAI `/v1/chat/completions` response.

See [Core Concepts](learn/concepts.md) for the canonical types and the event lifecycle, and
[Permissions & Safety](learn/permissions.md) for how gating works.

## License

Licensed under the **Apache-2.0 License**. See
[LICENSE](https://github.com/NaumanHSA/neurosurfer/blob/main/LICENSE).
