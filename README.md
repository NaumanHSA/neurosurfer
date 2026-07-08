<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/NaumanHSA/neurosurfer/main/docs/assets/banner/neurosurfer-banner-light.png">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/NaumanHSA/neurosurfer/main/docs/assets/banner/neurosurfer-banner-dark.png">
  <img alt="Neurosurfer — AI Agent Framework" src="https://raw.githubusercontent.com/NaumanHSA/neurosurfer/main/docs/assets/banner/neurosurfer-banner-dark.png" width="62%">
</picture>

<br/>
<br/>

<a href="https://naumanhsa.github.io/neurosurfer/getting-started/quickstart/"><picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/NaumanHSA/neurosurfer/main/docs/assets/buttons/quick-start-light.svg"><img height="42" alt="Quick Start" src="https://raw.githubusercontent.com/NaumanHSA/neurosurfer/main/docs/assets/buttons/quick-start-dark.svg"></picture></a>
<a href="https://naumanhsa.github.io/neurosurfer/"><picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/NaumanHSA/neurosurfer/main/docs/assets/buttons/documentation-light.svg"><img height="42" alt="Documentation" src="https://raw.githubusercontent.com/NaumanHSA/neurosurfer/main/docs/assets/buttons/documentation-dark.svg"></picture></a>
<a href="https://naumanhsa.github.io/neurosurfer/tutorials/"><picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/NaumanHSA/neurosurfer/main/docs/assets/buttons/examples-light.svg"><img height="42" alt="Examples" src="https://raw.githubusercontent.com/NaumanHSA/neurosurfer/main/docs/assets/buttons/examples-dark.svg"></picture></a>
<a href="https://pypi.org/project/neurosurfer/"><picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/NaumanHSA/neurosurfer/main/docs/assets/buttons/pypi-light.svg"><img height="42" alt="PyPI" src="https://raw.githubusercontent.com/NaumanHSA/neurosurfer/main/docs/assets/buttons/pypi-dark.svg"></picture></a>
<a href="https://github.com/NaumanHSA/neurosurfer"><picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/NaumanHSA/neurosurfer/main/docs/assets/buttons/github-light.svg"><img height="42" alt="GitHub" src="https://raw.githubusercontent.com/NaumanHSA/neurosurfer/main/docs/assets/buttons/github-dark.svg"></picture></a>

<br/>
<br/>

![PyPI](https://img.shields.io/pypi/v/neurosurfer?style=flat-square&labelColor=111111&color=111111)
![Python](https://img.shields.io/pypi/pyversions/neurosurfer?style=flat-square&labelColor=111111&color=111111)
![License](https://img.shields.io/badge/license-Apache--2.0-111111?style=flat-square&labelColor=111111)
![Status](https://img.shields.io/badge/status-stable-111111?style=flat-square&labelColor=111111)

</div>

**Neurosurfer** helps you build intelligent apps that blend **LLM reasoning**, **tools**, and **retrieval** — with a ready-to-run **OpenAI-compatible FastAPI gateway**. Start lean, add power as you go, on CPU or GPU.

---

## What's new

- **Observability — pluggable trace exporters** *(latest)* — ship **every agent run** to a real monitoring backend with **no code changes**. Ships with **[Langfuse](https://naumanhsa.github.io/neurosurfer/observability/langfuse/)** (traces, token cost, sessions) and **[OpenTelemetry](https://naumanhsa.github.io/neurosurfer/observability/opentelemetry/)** (GenAI-semconv spans over OTLP → Phoenix / Grafana / Datadog). Auto-on from the environment; runs → traces, LLM turns → generations, tool calls → spans, sub-agents & workflow nodes nest automatically. `pip install "neurosurfer[observability]"` — see [Observability](#observability).
- **Trace nesting & sessions** — a run spawned inside another run nests under it in the same trace (propagated across `await` and `asyncio.gather`); agents accept a `session_id` so a whole conversation groups into one session.
- **v1.0.0 — first stable release** *(2026-07-01)* — the public API (`neurosurfer.agents`, `.llm`, `.tools`, `.rag`, `.graph`, `.architect`, `.mcp`, `.app.server`) is now stable under semantic versioning.

> Full history in the [Changelog](CHANGELOG.md).

---

## What's in the box

- **Agent family** — `AgenticLoop` (native multi-step tool-use), `ReactAgent` (text-parsing ReAct for models without a native tool API), and `Agent` (one-shot, optionally with structured output).
- **LLM providers** — Anthropic Claude, OpenAI, and any OpenAI-compatible server (Ollama, LM Studio, vLLM, llama.cpp) behind one `Provider` protocol.
- **Rich tool ecosystem** — 15+ built-in tools: web search (DuckDuckGo/SerpAPI), sandboxed Python execution, file ops, HTTP, headless browser, memory — plus a simple framework for your own.
- **RAG pipeline** — ingest → chunk → embed → retrieve → token-aware context injection.
- **Graph & Workflows** — a standalone DAG engine and persisted, runnable Workflow packages.
- **Architect** — describe a workflow in plain English; it designs and builds the graph for you.
- **MCP client** — connect external Model Context Protocol servers and expose their tools to agents.
- **OpenAI-compatible gateway** — `/v1/models` + `/v1/chat/completions` with SSE streaming; proxy upstream backends or route to your own agents; request/response hooks.
- **Observability** — pluggable trace exporters (Langfuse, OpenTelemetry) with zero-overhead-when-off tracing.
- **Interactive CLI** — a `neurosurfer` REPL for chat and `neurosurfer serve` for the gateway.

---

## Tutorials

Hands-on notebooks — open any of them directly in Google Colab:

| # | Tutorial | What you'll build |
|---|----------|-------------------|
| 0 | **[Installation](https://colab.research.google.com/github/NaumanHSA/neurosurfer/blob/main/tutorials/00_installation.ipynb)** | Install Neurosurfer and its optional extras; verify your setup. |
| 1 | **[Providers & Agents](https://colab.research.google.com/github/NaumanHSA/neurosurfer/blob/main/tutorials/01_providers_and_agents.ipynb)** | Connect cloud and local providers, then run `AgenticLoop`, `ReactAgent`, and one-shot `Agent`. |
| 2 | **[Custom Tools](https://colab.research.google.com/github/NaumanHSA/neurosurfer/blob/main/tutorials/02_custom_tools.ipynb)** | Write your own tools and give agents new capabilities. |
| 3 | **[Graph Agents](https://colab.research.google.com/github/NaumanHSA/neurosurfer/blob/main/tutorials/03_graph_agents.ipynb)** | Compose multi-step workflows with the graph engine and Workflow packages. |
| 4 | **[MCP Servers](https://colab.research.google.com/github/NaumanHSA/neurosurfer/blob/main/tutorials/04_mcp_servers.ipynb)** | Connect external Model Context Protocol servers and expose their tools to agents. |
| 5 | **[Capstone: Insight Engine](https://colab.research.google.com/github/NaumanHSA/neurosurfer/blob/main/tutorials/05_capstone_insight_engine.ipynb)** | Put it all together — a database-backed insight engine over MCP. |

---

## Quick start

**Install:**
```bash
pip install -U neurosurfer
# with web search + gateway:
pip install -U "neurosurfer[search,serve]"
```

**Run the interactive CLI:**
```bash
neurosurfer
```

**Run the OpenAI-compatible gateway:**
```bash
neurosurfer serve --host 0.0.0.0 --port 8000
# proxy an upstream backend:
neurosurfer serve --upstream-url http://localhost:1234
```

**Multi-step agent (Anthropic):**
```python
import asyncio, os
from pathlib import Path
from neurosurfer.llm.providers.anthropic import AnthropicProvider
from neurosurfer.agents import AgenticLoop, Guardrails
from neurosurfer.tools import default_pool

provider = AnthropicProvider(api_key=os.environ["ANTHROPIC_API_KEY"], model="claude-opus-4-8")

class AutoIO:  # auto-approving IOHandler for scripts (see the Agents guide)
    async def ask(self, question, options=None): return (options or ["yes"])[0]
    async def request_plan_approval(self, plan): return True, ""
    async def request_shell_approval(self, command, reason): return True
    async def request_write_approval(self, path, summary): return "once"
    def notify(self, message): pass

async def main():
    agent = AgenticLoop(
        provider=provider, tools=default_pool(),
        system_prompt="Use tools to answer, then finish.",
        guardrails=Guardrails(), io=AutoIO(), cwd=Path.cwd(),
    )
    async for event in agent.run("Search the web for the latest news on AI agents."):
        if hasattr(event, "text"):
            print(event.text, end="", flush=True)

asyncio.run(main())
```

**One-shot with structured output:**
```python
import asyncio
from pathlib import Path
from pydantic import BaseModel
from neurosurfer.agents import Agent, Guardrails
from neurosurfer.tools import default_pool

class Summary(BaseModel):
    title: str
    points: list[str]

agent = Agent(
    provider=provider, tools=default_pool(),
    system_prompt="Answer concisely.",
    guardrails=Guardrails(), io=AutoIO(), cwd=Path.cwd(),
    output_schema=Summary,
)
result = asyncio.run(agent.complete("Summarise the Neurosurfer framework in 3 bullet points."))
print(result.title, result.points)  # `result` is a validated Summary instance
```

**Register an agent as an OpenAI-compatible model:**
```python
from neurosurfer.app.server import NeurosurferServer
from neurosurfer.agents import AgenticLoop

server = NeurosurferServer()
server.register_agent(AgenticLoop(provider=provider), model_id="my-agent")
server.run()  # → http://localhost:8000/v1/chat/completions
```

---

## Observability

See and debug **what your agents actually do** — the LLM turns, tool calls, token usage, and cost — in a real dashboard. Tracing is a cross-cutting, side-channel layer: it *observes* the event stream every agent already emits, never consumes it, so nothing about how you call `agent.run(...)` changes.

- **Zero code changes** — auto-on from the environment. Set a backend's connection vars and it activates on the next run.
- **Two backends in the box** — **Langfuse** (batteries-included LLM observability) and **OpenTelemetry** (vendor-neutral GenAI-semconv spans over OTLP → Honeycomb, Phoenix, Grafana Tempo, Datadog…). Or write your own `TraceExporter`.
- **Automatic nesting** — a run is a **trace**; each LLM turn a **generation** (with token cost); each tool call a **span**; spawned sub-agents and workflow nodes nest under the parent (`workflow → node → agent → tool`).
- **Safe by design** — zero overhead when off, and a misbehaving or unreachable exporter never breaks a run.

```bash
pip install "neurosurfer[observability]"

# Langfuse — auto-detected from the environment:
export LANGFUSE_PUBLIC_KEY=pk-...  LANGFUSE_SECRET_KEY=sk-...
# …or any OTel backend:
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
```

That's it — run any agent and the traces show up. Full guide: **[Observability docs](https://naumanhsa.github.io/neurosurfer/observability/)**.

---

## Key features

- **Native tool-use** — provider-native function calling; no sentinel leakage, clean structured output.
- **ReactAgent** — text-parsing ReAct for models without native tool APIs (Ollama, llama.cpp, small local models).
- **Sandboxed code execution** — subprocess-isolated Python with process-group kill, memory cap, and env sanitisation.
- **Web search** — DuckDuckGo (free) or SerpAPI (Google), BM25-ranked result injection within a token budget.
- **Workflow builder** — plain-English → multi-step DAG → registered workflow; runs on any provider.
- **OpenAI gateway** — drop-in server; register agents as model IDs; hooks for auth, request rewriting, response filtering.

---

## Install options

| Extra | What you get |
|---|---|
| *(base)* | Agents, LLM providers, tools, RAG, server, CLI |
| `search` | Web search tool (DuckDuckGo, BM25 ranking, HTML extraction) |
| `browser` | Headless browser tool via Playwright |
| `local` | `tiktoken` for accurate token counting with local models |
| `rag` | ChromaDB, sentence-transformers, PDF/DOCX/PPTX readers |
| `serve` | FastAPI + uvicorn for the OpenAI-compatible gateway |
| `observability` | Langfuse + OpenTelemetry trace exporters |
| `local-models` | PyTorch + Transformers for local model inference |
| `dev` | pytest, ruff, mypy, build tools |

```bash
pip install "neurosurfer[search,serve,rag,observability]"
```

---

## License

Licensed under the **Apache-2.0 License**. See [`LICENSE`](LICENSE).

## Support

- Star the project on [GitHub](https://github.com/NaumanHSA/neurosurfer)
- Ask & share in [Discussions](https://github.com/NaumanHSA/neurosurfer/discussions)
- File [Issues](https://github.com/NaumanHSA/neurosurfer/issues)
- Security: report privately to **naumanhsa965@gmail.com**

## Citation

```bibtex
@software{neurosurfer,
  author  = {Neurosurfer Team},
  title   = {Neurosurfer: A Production-Ready AI Agent Framework},
  year    = {2026},
  url     = {https://github.com/NaumanHSA/neurosurfer},
  license = {Apache-2.0}
}
```

---

<div align="center">
  <sub>Built by the Neurosurfer team · Apache-2.0</sub>
</div>
