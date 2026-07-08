# SQL Agent

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NaumanHSA/neurosurfer/blob/main/tutorials/06_capstone_sql_agent.ipynb)

Build a SQL agent, then compare the two agent strategies — `ReactAgent` vs `AgenticLoop` — on the
same task.

## The idea

Give an agent a tool that runs SQL against a database and let it answer natural-language questions.
The interesting part is running the **same task through both agent types** to feel the difference:

- **`AgenticLoop`** — uses the provider's **native** tool-calling. Cleaner and more reliable when the
  model supports it.
- **`ReactAgent`** — drives the SQL tool by **parsing text** (ReAct). Works with local models that
  lack a native tool API, at the cost of some robustness.

```python
# same tools + prompt, two agents
common = dict(provider=provider, tools=sql_pool,
              system_prompt="Answer using the query_sql tool. Then finish.",
              guardrails=Guardrails(), io=AutoIO(), cwd=Path.cwd())

loop  = AgenticLoop(**common)
react = ReactAgent(**common)

for agent in (loop, react):
    result = await agent.run_collect("How many orders shipped last month?")
    print(type(agent).__name__, "→", result.final_text)
```

Turn on [observability](../observability/index.md) to see each agent's turns and tool calls side by
side in a trace.

## Full notebook

The [Colab notebook](https://colab.research.google.com/github/NaumanHSA/neurosurfer/blob/main/tutorials/06_capstone_sql_agent.ipynb)
has the full schema, the SQL tool, and the head-to-head comparison.

**Back to:** [Tutorials overview](index.md)
