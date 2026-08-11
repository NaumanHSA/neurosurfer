# Building in Python

`GraphBuilder` is a fluent API that constructs the **same `Graph` IR as YAML**, with full
control-flow support. Every method returns `self`.

```python
from neurosurfer.graph import GraphBuilder

g = (
    GraphBuilder("triage", description="Route a ticket")
    .input("text", type="string", required=True)
    .base("classify", purpose="Classify urgency of: {text}", writes="label")
    .router(
        "route",
        cases=[{"when": "nodes.classify == 'urgent'", "to": "page"}],
        default="queue",
        depends_on=["classify"],
    )
    .base("page", purpose="Page on-call", depends_on=["route"])
    .base("queue", purpose="Add to queue", depends_on=["route"])
    .outputs("page", "queue")
    .build()
)
```

## Builder output and YAML are the same thing

`build()` runs the **same semantic validation the YAML loader does**, so builder output and YAML
output are identical and equally validated. There is no second code path to keep in step, and no
class of bug that reaches one and not the other.

The round-trip holds:

```python
from neurosurfer.graph import load_graph_from_dict

load_graph_from_dict(g.model_dump(mode="json"))   # reproduces g
```

## The methods

| Method | Adds |
|---|---|
| `.input(name, *, type="string", required=True)` | A declared graph input. |
| `.base(id, *, purpose=…, goal=…, …)` | A [`base`](node-kinds.md#base) node. |
| `.react(id, *, tools=[…], …)` | A [`react`](node-kinds.md#react) node. |
| `.tool(id, *, tools=[…], tool_args={…})` | A [`tool`](node-kinds.md#tool) node. |
| `.function(id, *, callable="mod:fn")` | A [`function`](node-kinds.md#function) node. |
| `.router(id, *, routes={…} \| cases=[…], default=…)` | A [`router`](node-kinds.md#router) node. |
| `.loop(id, *, body=[…], max_iterations=N)` | A [`loop`](node-kinds.md#loop) node. |
| `.map(id, *, over=…, body=[…], as_="item")` | A [`map`](node-kinds.md#map) node. |
| `.subgraph(id, *, body=[…])` | A [`subgraph`](node-kinds.md#subgraph) node. |
| `.input_node(id, *, purpose=…)` | An [`input`](node-kinds.md#input) node. |
| `.node(node_or_dict)` | Any pre-built `GraphNode` or its dict form. |
| `.outputs(*node_ids)` | Declares which nodes' outputs the graph returns. |
| `.to_dict()` / `.build()` | The dict form / the validated `Graph`. |

!!! note "`map` takes `as_`, not `item_var`"
    The trailing underscore is there because `as` is a Python keyword. It sets the same thing
    `item_var` sets in YAML — the name each item is bound to inside the body.

## Or construct nodes directly

The builder is a convenience, not the only door. Node classes produce the same IR:

```python
from neurosurfer.graph import Graph, RouterNode, BaseNode

graph = Graph(
    name="triage",
    nodes=[
        BaseNode(id="classify", instructions="Classify urgency of: {text}", writes="label"),
        RouterNode(id="route", cases=[{"when": "nodes.classify == 'urgent'", "to": "page"}],
                   default="queue", depends_on=["classify"]),
    ],
    inputs=[{"name": "text", "type": "string"}],
)
```

See [Node kinds](node-kinds.md#two-ways-to-write-the-same-node) for how classes and `kind=` strings
relate.

## Next

- [Node kinds](node-kinds.md) — the field reference.
- [Validation](validation.md) — what `build()` checks.
- [Workflow packages](packages.md) — persisting what you built.
