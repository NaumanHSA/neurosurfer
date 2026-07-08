# Providers & Agents

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NaumanHSA/neurosurfer/blob/main/tutorials/01_providers_and_agents.ipynb)

Connect a provider, then drive all three agent types.

## Minimal walkthrough

Build a provider (Anthropic, OpenAI, or any OpenAI-compatible server):

```python
from neurosurfer.llm.providers.openai import OpenAICompatProvider

provider = OpenAICompatProvider(
    base_url="http://localhost:1234/v1",
    api_key="not-needed",
    model="qwen/qwen3.5-9b",
    context_window=32_768,
)
```

Run an `AgenticLoop` and stream its events:

```python
from pathlib import Path
from neurosurfer.agents import AgenticLoop, Guardrails, TextDelta
from neurosurfer.tools import default_pool

agent = AgenticLoop(
    provider=provider, tools=default_pool(),
    system_prompt="Use tools, then finish.",
    guardrails=Guardrails(), io=AutoIO(), cwd=Path.cwd(),
)

async for ev in agent.run("List the .md files here and summarise the README."):
    if isinstance(ev, TextDelta):
        print(ev.text, end="", flush=True)
```

- Use **`ReactAgent`** (same constructor) for local models without a native tool API.
- Use the one-shot **`Agent`** with `output_schema=` for a single validated result.

## Full notebook

The [Colab notebook](https://colab.research.google.com/github/NaumanHSA/neurosurfer/blob/main/tutorials/01_providers_and_agents.ipynb)
compares cloud vs local providers and all three agents side by side.

**Next:** [Providers guide](../guides/providers.md) · [Agents guide](../guides/agents.md) ·
[Tutorial 2 →](custom-tools.md)
