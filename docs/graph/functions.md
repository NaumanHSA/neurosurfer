# Python in a Graph

A graph is data, and data cannot hold a Python function — so anywhere a graph wants real code, it
**names** it instead. There are two places that happens, and they receive completely different
arguments.

| Where | What runs | What it receives |
|---|---|---|
| A [`function` node](node-kinds.md#function) | Deterministic work as a step | **Every** graph input, dependency output and scope variable, as keyword arguments |
| A loop's [`until`](control-flow.md#loop) | A stop decision | One `LoopIteration` object |

## The sidecar: `functions:`

Declare one Python file at the top of the graph and then use **bare names** anywhere below:

```yaml
functions: helpers.py

nodes:
  - id: polish
    kind: loop
    max_iterations: 4
    until: tagline_is_short        # ← defined in helpers.py
```

The file sits beside `graph.yaml`, is resolved relative to it, and is **copied with the package on
export** — so a workflow stays self-contained, the same guarantee `nodes/<id>.py` already gives
`function` nodes.

Without a sidecar you can still name code, using a full import path — which works, and makes every
call site carry module plumbing:

```yaml
    until: nodes.checks:tagline_is_short
```

### A sidecar stands alone

It is imported **by path**, under a private module name derived from that absolute path. Two
workflows may each have a `helpers.py` without colliding, and neither is importable as a normal
top-level module by accident.

The consequence: **no relative imports.** `from .utils import x` will not resolve. Import from
installed packages, or keep what you need in the one file.

!!! tip "Editing a sidecar is picked up"
    Modules are cached by path **and mtime**, so editing `helpers.py` and re-loading the graph picks
    up the change — which matters in a notebook holding a graph across edits.

## `until` — a function, or a sentence

`until` accepts either, and telling them apart is a **lookup, not a heuristic**: a name the sidecar
defines is the function; anything else is prose judged by an LLM.

A graph that declares a sidecar and names something **absent** from it is an **error**, not a
prompt. That is the whole point — a typo cannot quietly become an English condition sent to a model,
and a one-word English condition is still distinguishable from a missing function because one of
them is in the module and the other is not.

### What an `until` function receives

One `LoopIteration` object. One object rather than several arguments, because this contract becomes
public the moment anyone writes an `until` function — adding a field later must not break every one
of them.

```python
def tagline_is_short(it):
    words = len(str(it.output).split())
    if words < 6:
        return True
    return False, f"{words} words — cut it to under six"
```

| Attribute | Type | Is |
|---|---|---|
| `it.index` | `int` | 0-based, matching the `index` the body's templates see. |
| `it.iteration` | `int` | 1-based — the human count, and what `structured_output["iterations"]` reports. |
| `it.output` | `Any` | The reduced output of this pass. |
| `it.result` | `GraphExecutionResult` | This pass's **full body run** — every node's output, errors, skips and token usage, with nothing hidden behind a reduction. |
| `it.history` | `list[Any]` | Every iteration's `output` so far, **including this one**. |
| `it.feedback` | `str` | The reason carried out of the previous iteration; `""` on the first. |
| `it.max_iterations` | `int` | The ceiling, so a function can tell "last chance" from "keep going". |
| `it.vars` | `dict` | The parent workflow's variables, readable. **A copy — writes do not leak.** |
| `it.is_last` | `bool` | Property. `True` when the ceiling will stop the loop after this pass anyway. |

`result` is the substance, but a stop decision usually needs more than the latest run — *"has this
stopped improving"*, *"have I tried three times"*, *"did the score go down"* are all questions about
the **sequence**, and a result object cannot answer them. So it arrives alongside the position and
the history.

### What it returns

| Return | Effect |
|---|---|
| `True` | Stop. |
| `False` | Continue. |
| `(True, "reason")` | Stop, and record the reason. |
| `(False, "reason")` | Continue, and set the next iteration's `{feedback}`. |

The `reason` channel is the **same one the plain-English judge uses**, so a function can steer the
next attempt just as a judge would:

```python
def score_is_good(it):
    score = it.result.final.get("judge", "")
    if "PASS" in score:
        return True, "judge passed"
    if it.is_last:
        return True, "out of iterations"
    return False, f"attempt {it.iteration} scored {score}; tighten the opening line"
```

## `function` nodes — the important gotcha

A `function` node imports a callable and calls it with the graph's inputs, its dependencies'
outputs, and any container scope, **merged into one set of keyword arguments**:

```yaml
- id: dedupe
  kind: function
  callable: helpers:drop_duplicates
  depends_on: [gather]
```

The call is effectively:

```python
fn(**{**graph_inputs, **dependency_outputs, **container_scope})
```

Container scope comes **last**, so inside a `map` the current `item` wins over a graph input that
happens to share its name.

!!! danger "Everything is passed, so accept `**kwargs`"
    There is **no filtering against your signature.** Every graph input and every dependency output
    is passed, whether your function wants it or not — so a function that does not accept them all
    fails at run time with `TypeError: got an unexpected keyword argument`.

    ```python
    # ✗ fails the moment the graph declares a second input
    def only_topic(topic):
        ...

    # ✓ takes what it needs, tolerates the rest
    def only_topic(topic, **kwargs):
        ...
    ```

    Verified: a graph declaring `topic` and `extra` against `def only_topic(topic)` fails the node
    with `only_topic() got an unexpected keyword argument 'extra'`.

**Match by name.** A parameter whose name is not a graph input, an upstream node id, or a scope
variable receives nothing — and nothing warns about this today.

```python
def drop_duplicates(gather, **kwargs):     # `gather` = the node id it depends on
    seen, out = set(), []
    for line in str(gather).splitlines():
        if line not in seen:
            seen.add(line)
            out.append(line)
    return "\n".join(out)
```

### Per-node files

A `function` node can also carry its own file, `nodes/<id>.py`, which travels with the package. The
sidecar and per-node files solve the same self-containment problem; the sidecar is declared once so
bare names work everywhere.

### `python` is currently the same thing

The `python` kind routes to the same executor path and requires the same import path. **It does not
run inline code**, despite the name. See [Node kinds](node-kinds.md#python).

## Secrets never reach your code by accident

`tool_args` reach a `function` node's kwargs **after** `${NAME}` substitution, so for a node using a
credential that mapping holds the plaintext value. It is redacted in the trace and in the run
record — but it is live in your function. Treat it as you would any secret: do not log it, do not
return it.

## Next

- [Authoring in YAML](yaml.md) — where `functions:` is declared.
- [Control flow](control-flow.md#loop) — the loop semantics `until` participates in.
- [Node kinds](node-kinds.md#function) — the `function` node's field reference.
- [State & secrets](state.md) — the `${NAME}` boundary.
