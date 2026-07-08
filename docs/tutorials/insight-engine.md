# Insight Engine

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NaumanHSA/neurosurfer/blob/main/tutorials/05_capstone_insight_engine.ipynb)

Put the pieces together: an agent that answers questions over a database, its tools served via MCP.

## What it combines

This capstone stitches together everything from the earlier tutorials:

- A **provider** and an **`AgenticLoop`** as the reasoning core.
- A **database exposed over MCP**, so the agent queries data through governed MCP tools.
- **Guardrails** to keep queries read-only and bound what the agent can touch.
- Optionally, **[observability](../observability/index.md)** to trace each run.

```python
# sketch — the notebook has the full, runnable version
agent = AgenticLoop(
    provider=provider,
    tools=pool_with_db_over_mcp,       # MCP tools from a DB server
    system_prompt="Answer questions about the dataset. Query, then summarise.",
    guardrails=Guardrails(shell_policy="denied", network_policy="denied"),
    io=AutoIO(), cwd=Path.cwd(),
)
answer = await agent.run_collect("Which region grew fastest last quarter?")
print(answer.final_text)
```

## Full notebook

Follow the [Colab notebook](https://colab.research.google.com/github/NaumanHSA/neurosurfer/blob/main/tutorials/05_capstone_insight_engine.ipynb)
for the complete build — schema, MCP wiring, and example questions.

**Next:** [Tutorial 6 →](sql-agent.md)
