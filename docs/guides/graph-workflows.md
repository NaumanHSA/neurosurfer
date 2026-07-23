# Graph & Workflows

Neurosurfer has a graph layer for **multi-step** pipelines:

- **`neurosurfer.graph.engine`** — a standalone DAG engine (`Graph`, `GraphNode`, `GraphExecutor`).
  Think of it as the framework's LangGraph analog.
- **`neurosurfer.graph.workflow`** — a persisted, versioned **Workflow package** layered on the
  engine: save a graph to disk, register it, and run it later.

The [Architect](../architect/index.md) builds these graphs for you from plain English — this page covers
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

### Control flow

Beyond linear `depends_on` pipelines, a graph can **branch**, **loop**, **fan out**, and **recover
from errors**. These are the constructs the [Architect](../architect/index.md) reaches for
automatically — and you can author them directly in Python or YAML.

**Router — take one branch of many.** A `router` node *is* the classifier: it makes one LLM call,
picks a labelled route, and prunes the branches not taken (they are *skipped*, not errored). Every
target must `depends_on` the router.

```python
GraphNode(
    id="triage",
    kind="router",
    goal="Route this support ticket by urgency: {ticket}",
    routes={"urgent": "escalate", "routine": "reply"},   # N-way, one LLM call
    default="reply",          # used if the model's answer maps to no route
)
```

For deterministic routing on a prior node's output (no LLM call), use `cases` instead:
`cases=[{"when": "contains(lower(nodes.check), 'yes')", "to": "approve"}]`.

**Loop — iterate until good.** State the stop condition in plain English via `until`; a hidden
judge decides STOP/CONTINUE after each iteration, and its reason reaches the next pass as
`{feedback}`. `max_iterations` is a mandatory ceiling.

```python
GraphNode(
    id="refine",
    kind="loop",
    max_iterations=3,
    until="the review approves the draft",
    body=[...],               # nested nodes, run once per iteration
)
```

For deterministic loops (budgets, cursors, index checks) use `break_when="<expression>"` instead of
`until`.

**Map — fan out over a list.** Runs `body` once per item of `over` (bound to `item_var`, default
`item`), up to `concurrency` in parallel; the node's output is the ordered per-item results.

```python
GraphNode(id="per_item", kind="map", over="inputs.items", item_var="item", concurrency=4,
          body=[GraphNode(id="handle", kind="base", goal="Process one item: {item}")])
```

**Conditional & resilient edges** (available on any node):

| Field | Effect |
|---|---|
| `when: "<expr>"` | The node runs only if the expression is truthy; at a merge, pruned branches use OR-join (the join still runs if *any* incoming branch is live). |
| `on_error: "<node_id>"` | On failure, reroute to a fallback node instead of failing the branch (the error text is exposed as `vars.<id>__error`). |
| `writes: "<name>"` | Store the node's output as `{name}` for downstream templates and expressions. |
| `policy.retries: N` | Re-run a flaky node up to N times before it counts as failed. |

Expressions (in `when`, `cases`, `break_when`, `over`) use a **safe evaluator** — no `eval`, no
imports, no attribute access. Read state as `inputs.x`, `nodes.<id>`, `vars.<name>`; prefer
`contains(lower(nodes.x), 'label')` over exact equality against raw LLM text.

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
registry is also what the [gateway](../server/index.md) and [CLI](../cli/index.md) use to run registered
workflows on any provider.

!!! tip "Don't want to hand-build graphs?"
    The [Architect](../architect/index.md) designs and builds a Workflow package from a plain-English
    description — then you run it exactly as above.
