# Graph Agents

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NaumanHSA/neurosurfer/blob/main/tutorials/03_graph_agents.ipynb)

Compose multiple steps into a DAG the engine runs for you.

## Minimal walkthrough

A graph is nodes (functions, tools, or agents) wired by dependencies. The engine runs independent
nodes in parallel and feeds outputs downstream:

```python
from neurosurfer.graph import Graph, GraphNode, GraphExecutor

graph = Graph(nodes=[
    GraphNode(id="fetch",   fn=fetch_docs),
    GraphNode(id="summarise", fn=summarise, depends_on=["fetch"]),
    GraphNode(id="report",  fn=make_report, depends_on=["summarise"]),
])

result = await GraphExecutor(graph).run(inputs={"query": "quarterly numbers"})
print(result["report"])
```

Nodes can be **agents**, so a step is a full tool-using run. Persisted, runnable **Workflow packages**
wrap a graph with its tools and metadata for reuse.

## Full notebook

The [Colab notebook](https://colab.research.google.com/github/NaumanHSA/neurosurfer/blob/main/tutorials/03_graph_agents.ipynb)
builds a multi-node workflow end to end, including agent nodes and parallel branches.

**Next:** [Graph & Workflows guide](../guides/graph-workflows.md) ·
[Architect](../architect/index.md) · [Tutorial 4 →](mcp-servers.md)
