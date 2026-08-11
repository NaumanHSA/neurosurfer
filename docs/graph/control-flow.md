# Control Flow

Beyond linear `depends_on` pipelines, a graph can **branch**, **loop**, **fan out**, and **recover
from errors**. These are the constructs the [Architect](../architect/index.md) reaches for
automatically — and you can author them directly in Python or YAML.

For what each kind requires, see [Node kinds](node-kinds.md).

## Router

**Take one branch of many.** A `router` node *is* the classifier: it makes one LLM call, picks a
labelled route, and prunes the branches not taken (they are *skipped*, not errored). Every target
must `depends_on` the router.

```python
GraphNode(
    id="triage",
    kind="router",
    instructions="Route this support ticket by urgency: {ticket}",
    routes={"urgent": "escalate", "routine": "reply"},   # N-way, one LLM call
    default="reply",          # used if the model's answer maps to no route
)
```

### Deterministic routing

For routing on a prior node's output with **no LLM call**, use `cases` instead of `routes` —
ordered predicates, first match wins:

```python
GraphNode(
    id="gate",
    kind="router",
    cases=[{"when": "contains(lower(nodes.check), 'yes')", "to": "approve"}],
    default="reject",
    depends_on=["check"],
)
```

Free, reproducible, and the right default whenever the decision is checkable in code.

### What the validator checks

A `routes` target that names a node which **does not exist**, or one that **does not list the
router in `depends_on`**, is reported — instead of silently pruning the whole branch at run time.

## Loop

**Iterate until good.** A loop runs its `body` repeatedly up to a mandatory `max_iterations`
ceiling.

```python
GraphNode(
    id="refine",
    kind="loop",
    max_iterations=3,
    until="the review approves the draft",
    body=[...],               # nested nodes, run once per iteration
)
```

### `until` is the only stop condition

It reads as one of two things:

- **a function** — a callable, or the name of one in the graph's `functions:` file. It receives a
  `LoopIteration` (`index`, `iteration`, `output`, `result` — the body's full
  `GraphExecutionResult` — `history`, `feedback`, `max_iterations`, `vars`, `is_last`) and returns
  `True` to stop, or `(stop, "reason")` to also set the next iteration's `{feedback}`.
  Deterministic and free; use it for anything code can check — budgets, cursors, counts,
  thresholds.
- **plain English** — judged by an internal LLM decision after each iteration. **One call per
  iteration**, so reach for it only when the judgement needs a reader.

Which one a string is, is a **lookup, not a guess**: a name the `functions:` file defines is the
function; anything else is prose. A graph that declares a `functions:` file and names something
absent from it is an **error**, not a prompt — that way a typo does not quietly become an English
condition sent to a model.

```yaml
functions: helpers.py          # sits beside graph.yaml, copied with it on export
nodes:
  - id: polish
    kind: loop
    max_iterations: 4
    until: tagline_is_short    # ← defined in helpers.py
```

```python
# helpers.py — a sidecar is imported by path, so it must stand alone (no relative imports)
def tagline_is_short(it):
    words = len(str(it.output).split())
    if words < 6:
        return True
    return False, f"{words} words — cut it to under six"
```

!!! note "A plain-English condition must describe what the body produces"
    The judge has a third verdict. Asked to stop "when winter is here" over a body writing coffee
    taglines, it answers UNRELATED: nothing the loop can produce would ever satisfy that, so
    continuing would spend the whole ceiling — every iteration plus a judge call each — to learn
    nothing. The loop stops, logs why, and still returns the work it did, with
    `structured_output["stopped_reason"] == "condition_unrelated"`.

    Deliberately narrow: a condition that is merely demanding, vague, or not yet met is CONTINUE,
    and an unparseable verdict fails safe to CONTINUE — a judge that could not be read must not be
    what stops a loop.

The full `LoopIteration` field table, and what a return value does, is on
[Python in a graph](functions.md#what-an-until-function-receives).

`break_when` was removed. See
[Upgrading](../about/upgrading.md#5-break_when-is-gone-a-loop-stops-for-one-reason).

## Map

**Fan out over a list.** Runs `body` once per item of `over` (bound to `item_var`, default `item`),
up to `concurrency` in parallel; the node's output is the ordered per-item results.

```python
GraphNode(id="per_item", kind="map", over="inputs.items", item_var="item", concurrency=4,
          body=[GraphNode(id="handle", kind="base", instructions="Process one item: {item}")])
```

The gather is implicit — no node is needed to collect the results. The body is handed **one item**,
never the whole collection.

## Subgraph

**A workflow inside a workflow.** Runs a nested body once; its final outputs become the node's
output. Body nodes may only depend on their **siblings** — a dependency pointing outside the body is
refused when the graph loads.

## Conditional and resilient edges

Available on any node:

| Field | Effect |
|---|---|
| `when: "<expr>"` | The node runs only if the expression is truthy; at a merge, pruned branches use OR-join (the join still runs if *any* incoming branch is live). |
| `on_error: "<node_id>"` | On failure, reroute to a fallback node instead of failing the branch (the error text is exposed as `vars.<id>__error`). The fallback lists this node in `depends_on`, and the engine prunes it when the node **succeeds** — so it runs on the failure path only, with no `when:` guard needed. |
| `writes: "<name>"` | Store the node's output as `{name}` for downstream templates and expressions. |
| `policy.retries: N` | Re-run a flaky node up to N times before it counts as failed. |

## Expressions

Expressions (in `when`, `cases`, `over`, and a loop's function form) use a **safe evaluator** — no
`eval`, no imports, no attribute access.

Read state as:

| Reference | Means |
|---|---|
| `inputs.x` | A declared graph input. |
| `nodes.<id>` | The output of a node. |
| `vars.<name>` | A value stored by a `writes:`. |

Prefer `contains(lower(nodes.x), 'label')` over exact equality against raw LLM text — a model that
answers "Yes." does not equal `"yes"`.

## Next

- [Node kinds](node-kinds.md) — the full field reference for each kind.
- [State & secrets](state.md) — what a node can see.
- [Validation](validation.md) — what is checked before a run starts.
