# Graph & Workflows

Neurosurfer has a graph layer for **multi-step** pipelines:

- **`neurosurfer.graph.engine`** — a standalone DAG engine (`Graph`, `GraphNode`, `GraphExecutor`).
  Think of it as the framework's LangGraph analog.
- **`neurosurfer.graph.workflow`** — a persisted, versioned **Workflow package** layered on the
  engine: save a graph to disk, register it, and run it later.

The [Architect](architect.md) builds these graphs for you from plain English — this page covers
building and running them directly.

## Build a graph

A `Graph` is a set of `GraphNode`s. Each node has an `id`, a `kind`, a `goal`, and optional
`depends_on` edges — a node receives the outputs of its dependencies as context.

```python
from neurosurfer.graph import Graph, GraphNode

researcher = GraphNode(
    id="researcher",
    kind="base",
    description="Fact-finding node.",
    goal="Research the topic and produce exactly 5 key bullet points.",
)

writer = GraphNode(
    id="writer",
    kind="base",
    description="Turns research notes into prose.",
    goal="Write a clear, 2-paragraph explanation from the research notes above.",
    depends_on=["researcher"],     # receives researcher's output
)

graph = Graph(
    name="content_pipeline",
    description="Research a topic, then explain it.",
    nodes=[researcher, writer],
)
```

### Node kinds

- **`base`** — a single bounded LLM step (like the one-shot `Agent`).
- **`react`** — a multi-step tool-using node (like `ReactAgent`); give it a `tools` allow-list:

```python
GraphNode(
    id="scout",
    kind="react",
    description="Explores the project and gathers file info.",
    goal="Use list_dir to explore, read README.md, summarise, then call finish.",
    tools=["list_dir", "read_file", "finish"],
)
```

## Run a graph

`GraphExecutor` runs the DAG on a provider, resolving dependencies and passing outputs downstream:

```python
from neurosurfer.graph import GraphExecutor

executor = GraphExecutor(graph, provider=provider)
result = executor.run({"user_intent": "Explain how attention works in Transformers"})

print(result.execution_summary())
print("succeeded:", result.succeeded)
print("errors:", result.errors or "none")
```

The `GraphExecutionResult` exposes `execution_summary()`, `succeeded`, `errors`, and per-node output.

## Workflow packages

A **Workflow package** is a graph saved as a versioned, multi-file package you can persist, share,
register, and run later. Load one and run it with `WorkflowRunner`:

```python
from neurosurfer.graph.workflow import load_package, WorkflowRunner

pkg = load_package(pkg_dir)
print(pkg.name, pkg.version, [n.id for n in pkg.graph.nodes])

runner = WorkflowRunner(provider, cwd=repo_root)   # cwd = working dir for tool contexts
result = runner.run(pkg, inputs={"user_intent": "Explain gradient descent"})

print(result.execution_summary())
print(result.final.get("writer", "(none)"))        # output of the 'writer' node
```

### Registering workflows

`WorkflowRegistry` stores packages so you can look them up by name and run them anywhere:

```python
from neurosurfer.graph.workflow import WorkflowRegistry

registry = WorkflowRegistry()
pkg = registry.get("content_pipeline")

result = WorkflowRunner(provider, cwd=repo_root).run(
    pkg, inputs={"user_intent": "RNNs vs Transformers"},
)
```

Use `save_package` / `load_package` to move packages between the filesystem and the registry. The
registry is also what the [gateway](../server/index.md) and [CLI](../cli.md) use to run registered
workflows on any provider.

!!! tip "Don't want to hand-build graphs?"
    The [Architect](architect.md) designs and builds a Workflow package from a plain-English
    description — then you run it exactly as above.
