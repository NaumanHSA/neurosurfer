# Neurosurfer

**Neurosurfer** helps you build intelligent apps that blend **LLM reasoning**, **tools**, and
**retrieval** — with a ready-to-run **OpenAI-compatible FastAPI gateway**. Start lean, add power as
you go, on CPU or GPU.

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **Get started**

    ---

    Install the package and run your first agent in a few minutes.

    [:octicons-arrow-right-24: Getting Started](getting-started/quickstart.md)

-   :material-robot:{ .lg .middle } **Agents**

    ---

    `AgenticLoop`, `ReactAgent`, and one-shot `Agent` with streaming events.

    [:octicons-arrow-right-24: Agents guide](guides/agents.md)

-   :material-server-network:{ .lg .middle } **Gateway**

    ---

    Serve any agent behind an OpenAI-compatible `/v1/chat/completions` API.

    [:octicons-arrow-right-24: Server guide](server/index.md)

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
  and more) plus a simple framework for your own.
- **RAG** — ingest → chunk → embed → retrieve → token-aware context injection.
- **Graph & Workflows** — a standalone DAG engine and persisted, runnable Workflow packages.
- **Architect** — describe a workflow in plain English; it designs and builds the graph for you.
- **MCP client** — connect external Model Context Protocol servers and expose their tools to agents.
- **OpenAI-compatible gateway** — `/v1/models` + `/v1/chat/completions` with SSE streaming, upstream
  proxying, native-agent backends, and request/response hooks.
- **Interactive CLI** — a REPL for chat and a `serve` command for the gateway.

## Install

```bash
pip install -U neurosurfer
# with web search + gateway:
pip install -U "neurosurfer[search,serve]"
```

See [Getting Started](getting-started/quickstart.md) for the full list of optional extras.

## License

Licensed under the **Apache-2.0 License**. See
[LICENSE](https://github.com/NaumanHSA/neurosurfer/blob/main/LICENSE).
