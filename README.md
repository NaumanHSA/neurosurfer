<div align="center">
  <img src="https://raw.githubusercontent.com/NaumanHSA/neurosurfer/main/docs/assets/banner/neurosurfer_banner_white.png" alt="Neurosurfer — AI Agent Framework" width="50%"/>
  <img src="https://raw.githubusercontent.com/NaumanHSA/neurosurfer/main/docs/assets/neurosurfer_water_wave.svg" alt="Neurosurfer — AI Agent Framework" width="100%"/>

  <a href="https://naumanhsa.github.io/neurosurfer/#quick-start" target="_blank"><img src="https://raw.githubusercontent.com/NaumanHSA/neurosurfer/main/docs/assets/buttons/quick_start_button.png" height="40" alt="Quick Start"></a>
  <a href="https://naumanhsa.github.io/neurosurfer/examples/" target="_blank"><img src="https://raw.githubusercontent.com/NaumanHSA/neurosurfer/main/docs/assets/buttons/examples_button.png" height="40" alt="Examples"></a>
  <a href="https://naumanhsa.github.io/neurosurfer/" target="_blank"><img src="https://raw.githubusercontent.com/NaumanHSA/neurosurfer/main/docs/assets/buttons/documentation_button.png" height="40" alt="Documentation"></a>
  <a href="https://pypi.org/project/neurosurfer/" target="_blank"><img src="https://raw.githubusercontent.com/NaumanHSA/neurosurfer/main/docs/assets/buttons/pypi_button.png" height="40" alt="PyPI"></a>
</div>

**Neurosurfer** helps you build intelligent apps that blend **LLM reasoning**, **tools**, and **retrieval**, with a ready-to-run **OpenAI-compatible FastAPI gateway**. Start lean, add power as you go — CPU-only or GPU-accelerated.

---

## 🚀 What's in the box

- 🤖 **Agent family** — `AgenticLoop` (native multi-step tool-use), `ReactAgent` (text-parsing ReAct for non-function-calling models), `Agent` (one-shot with structured output)
- 🧠 **LLM providers** — Anthropic Claude, OpenAI, and any OpenAI-compatible server (Ollama, LM Studio, vLLM, llama.cpp)
- 📚 **RAG pipeline** — ingest → chunk → embed → retrieve → token-aware context injection
- 🔧 **Rich tool ecosystem** — 15+ built-in tools: web search (DuckDuckGo/SerpAPI), sandboxed Python execution, file ops, HTTP, headless browser, memory, and more
- ⚙️ **OpenAI-compatible gateway** — `/v1/models` + `/v1/chat/completions` with SSE streaming; proxy upstream backends or route to your own agents; request/response hooks
- 🏗️ **Workflow builder** — describe a workflow in plain English, the Architect designs and builds a multi-step DAG; registered workflows run on any model
- 🧪 **CLI** — `neurosurfer` interactive REPL + `neurosurfer serve` for the gateway

---

<h2>🎓 Tutorials</h2>

Hands-on notebooks — open any of them directly in Google Colab:

| # | Tutorial | Open | What you'll build |
|---|----------|------|-------------------|
| 0 | **Installation** | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NaumanHSA/neurosurfer/blob/main/tutorials/00_installation.ipynb) | Install Neurosurfer and its optional extras; verify your setup. |
| 1 | **Providers & Agents** | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NaumanHSA/neurosurfer/blob/main/tutorials/01_providers_and_agents.ipynb) | Connect cloud and local providers, then run `AgenticLoop`, `ReactAgent`, and one-shot `Agent`. |
| 2 | **Custom Tools** | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NaumanHSA/neurosurfer/blob/main/tutorials/02_custom_tools.ipynb) | Write your own tools and give agents new capabilities. |
| 3 | **Graph Agents** | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NaumanHSA/neurosurfer/blob/main/tutorials/03_graph_agents.ipynb) | Compose multi-step workflows with the graph engine and Workflow packages. |
| 4 | **MCP Servers** | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NaumanHSA/neurosurfer/blob/main/tutorials/04_mcp_servers.ipynb) | Connect external Model Context Protocol servers and expose their tools to agents. |
| 5 | **Capstone: Insight Engine** | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NaumanHSA/neurosurfer/blob/main/tutorials/05_capstone_insight_engine.ipynb) | Put it all together — a database-backed insight engine over MCP. |

---

## ⚡ Quick Start

**Install:**
```bash
pip install -U neurosurfer
# With web search + gateway:
pip install -U "neurosurfer[search,serve]"
```

**Run the interactive CLI:**
```bash
neurosurfer
```

**Run the OpenAI-compatible gateway:**
```bash
neurosurfer serve --host 0.0.0.0 --port 8000
# Proxy an upstream backend:
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

## 🏗️ Architecture

```
neurosurfer/
  agents/        ← AgenticLoop · ReactAgent · Agent + events, permissions, subagents
  llm/           ← providers (Anthropic, OpenAI, OpenAI-compatible) + retry + tokens
  tools/         ← Tool framework + 15+ built-in tools (web_search, python_exec, …)
  rag/           ← ingestor · chunker · retriever · token-aware context builder
  vectorstores/  ← ChromaDB + in-memory store
  embeddings/    ← embedder protocol + backends
  cache/         ← LLM response cache + embedding cache
  graph/
    engine/      ← DAG executor (standalone core primitive)
    workflow/    ← persisted Workflow packages (register, load, run)
  architect/     ← conversational Architect (describe → design → build workflows)
  mcp/           ← Model Context Protocol client (connect external tool servers)
  app/
    server/      ← OpenAI-compatible FastAPI gateway with SSE streaming + hooks
    cli/         ← interactive REPL + subcommands (serve · provider · doctor)
```

---

## ✨ Key Features

- **Native tool-use** — provider-native function calling; no sentinel leakage, clean structured output
- **ReactAgent** — text-parsing ReAct for models without native tool APIs (Ollama, llama.cpp, small local models)
- **Sandboxed code execution** — subprocess-isolated Python with process-group kill, memory cap, and env sanitisation
- **Web search** — DuckDuckGo (free) or SerpAPI (Google), BM25-ranked result injection within a token budget
- **Workflow builder** — plain-English → multi-step DAG → registered workflow; runs on any provider
- **OpenAI gateway** — drop-in server; register agents as model IDs; hooks for auth, request rewriting, response filtering

---

## 📦 Install Options

| Extra | What you get |
|---|---|
| *(base)* | Agents, LLM providers, tools, RAG, server, CLI |
| `search` | Web search tool (DuckDuckGo, BM25 ranking, HTML extraction) |
| `browser` | Headless browser tool via Playwright |
| `local` | `tiktoken` for accurate token counting with local models |
| `rag` | ChromaDB, sentence-transformers, PDF/DOCX/PPTX readers |
| `serve` | FastAPI + uvicorn for the OpenAI-compatible gateway |
| `local-models` | PyTorch + Transformers for local model inference |
| `dev` | pytest, ruff, mypy, build tools |

```bash
pip install "neurosurfer[search,serve,rag]"
```

---

## 📝 License

Licensed under the **Apache-2.0 License**. See [`LICENSE`](LICENSE).

## 🌟 Support

- ⭐ Star the project on [GitHub](https://github.com/NaumanHSA/neurosurfer)
- 💬 Ask & share in [Discussions](https://github.com/NaumanHSA/neurosurfer/discussions)
- 🐛 File [Issues](https://github.com/NaumanHSA/neurosurfer/issues)
- 🔒 Security: report privately to **naumanhsa965@gmail.com**

## 📚 Citation

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
  <sub>Built with ❤️ by the Neurosurfer team</sub>
</div>
