# Graph & Workflows

Neurosurfer has a graph layer for **multi-step** pipelines:

- **`neurosurfer.graph.engine`** — a standalone DAG engine (`Graph`, `GraphNode`, `GraphExecutor`).
  Think of it as the framework's LangGraph analog.
- **`neurosurfer.graph.workflow`** — a persisted, versioned **Workflow package** layered on the
  engine: save a graph to disk, register it, and run it later.

The [Architect](../architect/index.md) builds these graphs for you from plain English — these pages
cover building and running them directly.

## The pages

| Page | Answers |
|---|---|
| [Node kinds](node-kinds.md) | What the eleven kinds are and what each one needs. |
| [Control flow](control-flow.md) | Branching, looping, fanning out, recovering from errors. |
| [State & secrets](state.md) | What a node can see — and what it must never see. |
| [Validation](validation.md) | What is checked, when, and what a failure means. |
| [Workflow packages](packages.md) | Persisting, registering, and running a graph later. |
| [Building in Python](building.md) | The `GraphBuilder` fluent API. |

## Build a graph

A `Graph` is a set of `GraphNode`s. Each node has an `id`, a `kind`, instructions, and optional
`depends_on` edges.

**What a node receives is what its own text names, plus the outputs of its `depends_on`.** Nothing
ambient: the graph's inputs are not appended to every node's prompt as a block, so a step that
needs one interpolates it by name.

```python
from neurosurfer.graph import Graph, GraphNode

researcher = GraphNode(
    id="researcher",
    kind="base",
    description="Fact-finding node.",
    goal="Research {topic} and produce exactly 5 key bullet points.",   # names its input
)

writer = GraphNode(
    id="writer",
    kind="base",
    description="Turns research notes into prose.",
    goal="Write a clear, 2-paragraph explanation from the research notes above.",
    depends_on=["researcher"],     # names nothing; the edge carries the notes
)

graph = Graph(
    name="content_pipeline",
    description="Research a topic, then explain it.",
    nodes=[researcher, writer],
    inputs=[{"name": "topic", "type": "string"}],
)
```

A declared input that no step names is reported by the validator, because the run would otherwise
go green while the model answers as though it had been passed nothing.

!!! warning "Upgrading an existing workflow?"
    This scoping rule changed, and a workflow written against the older behaviour still validates
    and still runs — it just answers as though handed nothing. See
    [Upgrading](../about/upgrading.md#1-a-node-is-told-what-it-names-nothing-ambient).

### The kinds, briefly

- **`base`** — one bounded LLM step. Reasons; cannot act repeatedly.
- **`react`** — a multi-step tool-using node. Reasons *and* acts.
- **`tool`** — one registered tool, called directly. Acts; cannot reason.
- **`function`** / **`python`** — deterministic Python by import path.
- **`router`**, **`loop`**, **`map`**, **`subgraph`** — [control flow](control-flow.md).
- **`input`**, **`output`** — the edges of a run.

Full field reference: [Node kinds](node-kinds.md).

## Run a graph

`GraphExecutor` runs the DAG on a provider, resolving dependencies and passing outputs downstream:

```python
from neurosurfer.graph import GraphExecutor

executor = GraphExecutor(graph, provider=provider)
result = executor.run({"topic": "how attention works in Transformers"})

print(result.execution_summary())
print("succeeded:", result.succeeded)
print("errors:", result.errors or "none")
```

The `GraphExecutionResult` exposes `execution_summary()`, `succeeded`, `errors`, and per-node
output.

**Validation runs first.** A graph that cannot run is refused before a model is called, rather than
partway through — see [Validation](validation.md).

## Persist it

A graph saved as a versioned package can be registered and run anywhere:

```python
from neurosurfer.graph.workflow import load_package, WorkflowRunner

pkg = load_package(pkg_dir)
result = WorkflowRunner(provider, cwd=repo_root).run(pkg, inputs={"topic": "gradient descent"})
```

See [Workflow packages](packages.md).

!!! tip "Don't want to hand-build graphs?"
    The [Architect](../architect/index.md) designs and builds a Workflow package from a
    plain-English description — then you run it exactly as above.
