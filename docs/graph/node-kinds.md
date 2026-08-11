# Node Kinds

A graph is a set of nodes. There are **eleven kinds**, and the kind decides what the node needs,
what it is allowed to do, and whether it calls a model at all.

Three of them reason (`base`, `react`, `router`), three run code (`tool`, `function`, `python`),
three contain other nodes (`loop`, `map`, `subgraph`), and two mark the edges of the run
(`input`, `output`).

What a kind requires is declared once, in
[`neurosurfer/graph/engine/kinds/`](https://github.com/NaumanHSA/neurosurfer/tree/main/neurosurfer/graph/engine/kinds).
The validator, the engine and this page all read the same declaration, so "what can go wrong with a
tool node" is a query rather than a careful read.

## Choosing a kind

| I need to… | Kind |
|---|---|
| Write, summarise, classify, transform text | [`base`](#base) |
| Do work that needs a model to *decide* what to call, repeatedly | [`react`](#react) |
| Call one known tool with arguments I already have | [`tool`](#tool) |
| Run deterministic Python | [`function`](#function) |
| Take one branch of several | [`router`](#router) |
| Repeat until something is good enough | [`loop`](#loop) |
| Do the same thing to every item of a list | [`map`](#map) |
| Group a chunk of graph as one unit | [`subgraph`](#subgraph) |
| Stop and ask a person | [`input`](#input) |
| Declare what the workflow returns | [`output`](#output) |

The distinction that matters most: **`base` reasons and cannot act; `tool` acts and cannot reason;
`react` does both.** A node whose goal describes reaching outside the model — reading a file,
fetching a URL, sending a message — cannot be a `base` node with a well-worded prompt. See
[Validation](validation.md#capability-grounding).

## Two ways to write the same node

Every kind is available as a **class** and as a `kind=` **string**. They produce the same node.

```python
from neurosurfer.graph import GraphNode, RouterNode

a = RouterNode(id="triage", instructions="Route by urgency: {ticket}", routes={...})
b = GraphNode(kind="router", id="triage", instructions="Route by urgency: {ticket}", routes={...})
```

`Graph` upgrades whatever it is given, so `isinstance(node, RouterNode)` is true however the node
was made, and YAML on disk is untouched. The engine dispatches on the class rather than on a kind
string.

The classes are `BaseNode`, `ReactNode`, `ToolNode`, `FunctionNode`, `PythonNode`, `RouterNode`,
`LoopNode`, `MapNode`, `SubgraphNode`, `InputNode`, `OutputNode`, plus `ContainerNode` for the three
that run a nested body.

!!! note "Why every name ends in `Node`"
    The bare names were not sayable at a call site. `Tool`, `Input`, `Output`, `Map`, `Function` and
    `Python` all already mean something else here — and `Tool` was an outright collision with
    `neurosurfer.tools.base.Tool`, the ABC every registered tool subclasses.

## Fields every node can use

| Field | Effect |
|---|---|
| `depends_on` | Ids this node waits for. Their outputs are passed to it. |
| `when` | An expression; the node runs only if it is truthy. |
| `on_error` | On failure, reroute to a fallback node instead of failing the branch. |
| `writes` | Store the output as `{name}` for downstream templates and expressions. |
| `policy.retries` | Re-run a flaky node up to N times before it counts as failed. |

See [Control flow](control-flow.md) for how `when` and `on_error` interact with branching, and
[State & secrets](state.md) for `writes`.

---

## base

**One LLM call** — writing, summarising, classifying, transforming text.

```python
GraphNode(
    id="writer",
    kind="base",
    instructions="Write a 2-paragraph explanation from the research notes above.",
    depends_on=["researcher"],
)
```

| Field | Notes |
|---|---|
| `instructions` | What the step should do. The older `purpose` / `goal` / `expected_result` trio is still read when it is absent. |
| `tools` | Optional. The model gets **one round** with them. |
| `tool_settings`, `secrets` | See [State & secrets](state.md). |
| `provider`, `mode`, `output_schema` | Per-node model and structured output. |

**Constraints**

- For structured output, set an output schema — without one, a node asked for an object returns
  JSON as a string.
- Tools are optional and the model gets **one round**. A step that must call tools repeatedly to
  finish its job is a `react` node. A `base` step cut off mid-plan now **fails** rather than
  reporting success — see [Upgrading](../about/upgrading.md#4-a-truncated-base-step-now-fails-instead-of-reporting-success).

## react

**An LLM that calls tools in a loop** — for work that must touch the outside world and needs a
model to decide what to send.

```python
GraphNode(
    id="scout",
    kind="react",
    instructions="Use list_dir to explore, read README.md, summarise, then call finish.",
    tools=["list_dir", "read_file", "finish"],
)
```

| Field | Notes |
|---|---|
| `instructions` | What the agent should accomplish. |
| `tools` | **Required.** A react node with none cannot act, and is refused. |
| `tool_args` | Bound arguments supplied on every call and **hidden from the schema** the model is offered. This is how a credential reaches a tool without reaching the model. |
| `tool_settings`, `secrets` | See [State & secrets](state.md). |
| `provider` | Per-node model. |

**Constraints**

- The only kind that both reasons and acts.
- **No `mode` or `output_schema`.** The loop does not honour a schema today, so the field is not
  offered rather than being written, reviewed, and read by nothing.
- A node that ends with `finish()` returns **what it finished with**.

## tool

**Calls one registered tool with the arguments you bind. No LLM call.**

```python
GraphNode(
    id="fetch",
    kind="tool",
    tools=["http"],
    tool_args={"url": "{source_url}", "method": "GET"},
)
```

| Field | Notes |
|---|---|
| `tools` | **Required** — the single tool this node invokes. |
| `tool_args` | The whole instruction. Each argument is a literal, a reference to an upstream node or graph input, or a `${CREDENTIAL}`. |
| `tool_settings`, `secrets` | See [State & secrets](state.md). |

**Constraints**

- There is **no LLM call**, so nothing composes its arguments and nothing reads an instruction. If
  an argument has to be *worked out* — a query, a search phrase, a request body — the step is a
  `react` node with that tool attached.
- Every **required parameter of the chosen tool** must be supplied from somewhere: an argument
  here, or a graph input or upstream output of the same name. One missing calls the tool with
  nothing and fails at run time.
- A credential goes in `secrets` and is written `${NAME}` in an argument, **never in a prompt**.

## function

**Deterministic Python**: imports a callable and calls it with the inputs and upstream outputs that
match its signature.

```python
GraphNode(id="dedupe", kind="function", callable="my_module:drop_duplicates",
          depends_on=["gather"])
```

| Field | Notes |
|---|---|
| `callable` | **Required.** Import path to a Python callable, `module:attr`. |

**Constraints**

- Arguments are matched to the signature **by name**, so a parameter whose name is not a graph
  input or an upstream node id receives nothing. Nothing warns about this today.
- No `export`: the exporter is only consulted on the LLM path, so `export: true` here would be a
  setting that silently does nothing.

## python

**Today, an alias of `function`.** Same executor path, same required import path.

It does **not** run inline code, despite the name. The spec states what the engine does rather than
what the label suggests; resolving the gap — real inline code, or retiring the kind — is an open
decision.

## router

**Picks one downstream branch** — by LLM classification, or by deterministic predicates. The
branches not taken are *skipped*, not errored.

```python
GraphNode(
    id="triage",
    kind="router",
    instructions="Route this support ticket by urgency: {ticket}",
    routes={"urgent": "escalate", "routine": "reply"},   # N-way, one LLM call
    default="reply",
)
```

| Field | Notes |
|---|---|
| `routes` | Label → target node. The router classifies with **one LLM call** and picks a label. |
| `cases` | Ordered predicates, first match wins. **No LLM call** — free and reproducible. |
| `default` | Used when the model's answer maps to no route. |
| `repair` | `routes` only: retry once with corrective feedback when the model picks a label that does not exist. |

**Constraints**

- Declare `routes` **or** `cases`, never both.
- **Every target must declare this router in its `depends_on`.** A target that does not, or that
  names a node which does not exist, is reported by the validator rather than pruning the whole
  branch at run time.
- The `cases` form makes no LLM call, so `provider` means nothing for it.

See [Control flow](control-flow.md#router) for the deterministic form and merge behaviour.

## loop

**Runs a nested body over and over** until a condition holds, up to a hard ceiling.

```python
GraphNode(
    id="refine",
    kind="loop",
    max_iterations=3,
    until="the review approves the draft",
    body=[...],
)
```

| Field | Notes |
|---|---|
| `body` | **Required.** Nested nodes, run once per iteration. |
| `max_iterations` | The ceiling. |
| `until` | The **one** stop condition — a function or plain English. |
| `accumulate` | Collect per-iteration output. |
| `item_var` | The name the previous iteration's output is available under inside the body. |

**Constraints**

- No `until` means it runs to the ceiling.
- A plain-English `until` costs **one LLM call per iteration**; a function costs nothing. Prefer a
  function whenever the condition is checkable in code.
- A plain-English `until` about a different subject than the body produces stops the loop with a
  warning rather than running to the ceiling — so the condition must actually describe the body's
  output.

`break_when` was removed; see [Upgrading](../about/upgrading.md#5-break_when-is-gone-a-loop-stops-for-one-reason)
and [Control flow](control-flow.md#loop) for the full `until` contract.

## map

**Runs a nested body once per item** of a collection, and returns the ordered list of results.

```python
GraphNode(id="per_item", kind="map", over="inputs.items", item_var="item", concurrency=4,
          body=[GraphNode(id="handle", kind="base", instructions="Process one item: {item}")])
```

| Field | Notes |
|---|---|
| `body` | **Required.** Run once per item. |
| `over` | Expression selecting the collection. |
| `item_var` | The name each item is bound to inside the body. Default `item`. |
| `concurrency` | How many items run in parallel. |

**Constraints**

- The output is the **ordered list** of per-item results — the gather is implicit, so no node is
  needed to collect them.
- The body is handed **one item**, not the whole collection.

## subgraph

**A workflow inside a workflow**: runs a nested body once, and its final outputs become this node's
output.

| Field | Notes |
|---|---|
| `body` | **Required.** The nested graph. |
| `body_outputs` | Which body nodes' outputs surface. |

**Constraints**

- Body nodes may only depend on their **siblings** — a dependency pointing outside the body is
  refused when the graph loads.

## input

**Pauses the run and waits for a person** to supply a value.

| Field | Notes |
|---|---|
| `input_mode` | `dict` (named, typed fields) or `text`. |
| `depends_on`, `on_error` | Wiring. |

**Constraints**

- In `dict` mode the fields a person fills in **are the workflow's declared inputs**.
- With nothing supplied the run finishes `awaiting_input`, **not** `failed`; a client resumes it.
- **Resuming currently re-runs the whole graph**, so a node before this one runs twice.

No `instructions` is offered on a panel — the conversation happens at run time. The engine still
uses one if a graph sets it: it is the question a *headless* CLI run asks.

## output

**Declares what the workflow returns.** The graph stops here.

| Field | Notes |
|---|---|
| `value` | A template for what to return. |
| `depends_on` | A dependency whose output it passes through. |

**Constraints**

- Needs **something** to return: a dependency whose output it passes through, or a value template.
  With neither it returns nothing.
- Nothing may depend on one, and it cannot be an error target.
- An unresolved placeholder in the value is **refused, not warned about** — this text is what a
  caller receives.
- Takes precedence over the older graph-level `outputs:` list.

---

## Next

- [Control flow](control-flow.md) — branching, looping, fanning out, and recovering from errors.
- [State & secrets](state.md) — what a node can see, and what it must never see.
- [Validation](validation.md) — what is checked, and when.
- [Building in Python](building.md) — the `GraphBuilder` fluent API.
