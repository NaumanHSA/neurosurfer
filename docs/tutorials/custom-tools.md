# Custom Tools

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NaumanHSA/neurosurfer/blob/main/tutorials/02_custom_tools.ipynb)

Give an agent a new capability by writing your own tool.

## Minimal walkthrough

A tool subclasses `Tool`, declares an input schema, and implements an async `run`:

```python
from neurosurfer.tools import Tool, ToolResult

class Weather(Tool):
    name = "weather"
    description = "Get the current weather for a city."
    input_schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }

    async def run(self, args, ctx) -> ToolResult:
        city = args["city"]
        # ... call your weather API ...
        return ToolResult(content=f"It's 22°C and sunny in {city}.")
```

Add it to a pool and hand that pool to an agent:

```python
from neurosurfer.tools import build_pool, default_pool

pool = default_pool()
pool.add(Weather())          # or build_pool(["read_file", "weather"])

agent = AgenticLoop(provider=provider, tools=pool, system_prompt="…",
                    guardrails=Guardrails(), io=AutoIO(), cwd=Path.cwd())
```

The model can now call `weather` like any built-in tool.

## Full notebook

The [Colab notebook](https://colab.research.google.com/github/NaumanHSA/neurosurfer/blob/main/tutorials/02_custom_tools.ipynb)
covers schemas, returning images/data, gating, and the tool registry.

**Next:** [Tools guide](../guides/tools.md) · [Tools catalog](../reference/tools-catalog.md) ·
[Tutorial 3 →](graph-agents.md)
