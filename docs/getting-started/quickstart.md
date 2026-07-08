# Getting Started

This page takes you from an empty environment to a running agent and gateway.

## Prerequisites

- **Python `>= 3.11`** and **pip** (or `uv` / `pipx` / `poetry`).
- An LLM provider — either an API key (Anthropic / OpenAI) or a local
  OpenAI-compatible server (Ollama, LM Studio, vLLM, llama.cpp).
- **GPU is optional.** CPU-only works fine for cloud providers and small local models.

## Install

The base install is lightweight; heavier capabilities live behind optional extras so you only pull
what you use.

```bash
pip install -U neurosurfer
```

### Optional extras

| Extra | What you get |
|---|---|
| *(base)* | Agents, LLM providers, tools, CLI |
| `search` | Web search tool (DuckDuckGo, BM25 ranking, HTML extraction) |
| `browser` | Headless browser tool via Playwright (`playwright install chromium`) |
| `local` | `tiktoken` for accurate token counting with local models |
| `rag` | ChromaDB, sentence-transformers, PDF/DOCX/PPTX readers |
| `serve` | FastAPI + uvicorn for the OpenAI-compatible gateway |
| `mcp` | Model Context Protocol client SDK |
| `dev` | pytest, ruff, mypy, build tools |

Combine extras as needed:

```bash
pip install -U "neurosurfer[search,serve,rag]"
```

## Configure a provider

Providers read their credentials from arguments or environment variables. The simplest path is to
export a key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
# or
export OPENAI_API_KEY=sk-...
```

See the [Providers guide](../guides/providers.md) for local servers (Ollama, LM Studio, vLLM,
llama.cpp) and the full `Provider` API.

## Your first agent (Python)

An agent needs a `provider`, a tool pool, a `system_prompt`, `guardrails`, an `io` handler (how it
asks for approvals), and a working directory. For scripts, supply a small **auto-approving** `io`:

```python
import asyncio, os
from pathlib import Path
from neurosurfer.llm.providers.anthropic import AnthropicProvider
from neurosurfer.agents import AgenticLoop, Guardrails
from neurosurfer.tools import default_pool

provider = AnthropicProvider(api_key=os.environ["ANTHROPIC_API_KEY"], model="claude-opus-4-8")

class AutoIO:
    """Auto-approving IOHandler for scripts. See the Agents guide for details."""
    async def ask(self, question, options=None): return (options or ["yes"])[0]
    async def request_plan_approval(self, plan): return True, ""
    async def request_shell_approval(self, command, reason): return True
    async def request_write_approval(self, path, summary): return "once"
    def notify(self, message): pass

async def main():
    agent = AgenticLoop(
        provider=provider,
        tools=default_pool(),
        system_prompt="You are a helpful assistant. Use tools, then finish.",
        guardrails=Guardrails(),
        io=AutoIO(),
        cwd=Path.cwd(),
    )
    async for event in agent.run("What are three recent trends in AI agents?"):
        if hasattr(event, "text"):
            print(event.text, end="", flush=True)

asyncio.run(main())
```

`agent.run(...)` is an async generator that yields streaming [events](../guides/agents.md#events).
Text arrives as `TextDelta` (which has a `.text` attribute), so the `hasattr` check above prints the
answer as it streams.

!!! warning
    `AutoIO` approves every action without prompting — only use it in trusted, sandboxed contexts,
    and constrain tools with [`Guardrails`](../guides/agents.md#permissions-and-guardrails).

## Structured output (one-shot)

When you want a validated object instead of free text, use the one-shot `Agent` with a Pydantic
`output_schema` and call `complete()`:

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
    provider=provider,
    tools=default_pool(),
    system_prompt="Answer concisely.",
    guardrails=Guardrails(),
    io=AutoIO(),
    cwd=Path.cwd(),
    output_schema=Summary,
)
result = asyncio.run(agent.complete("Summarise Neurosurfer in three bullet points."))
print(result.title, result.points)  # result is a validated Summary instance
```

## The CLI

Neurosurfer ships an interactive REPL and a gateway command:

```bash
neurosurfer              # interactive chat REPL
neurosurfer doctor       # check your provider configuration
neurosurfer serve        # start the OpenAI-compatible gateway
```

See the [CLI guide](../reference/cli.md) for provider profiles, slash commands, and `serve` flags.

## Where to next

- [Providers](../guides/providers.md) — cloud and local models behind one protocol.
- [Agents](../guides/agents.md) — the agent family, events, permissions, and sub-agents.
- [Tools](../guides/tools.md) — built-in tools and writing your own.
- [RAG](../guides/rag.md) — retrieval-augmented generation.
- [Graph & Workflows](../guides/graph-workflows.md) and the [Architect](../architect/index.md).
- [Tutorials](../tutorials/index.md) — hands-on Colab notebooks.
