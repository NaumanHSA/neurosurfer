# State & Secrets

What a node can see, how values move between nodes, and the one boundary the engine enforces
absolutely: **a secret reaches the tool and never the model.**

## What a node receives

**What its own text names, plus the outputs of the steps it declared in `depends_on`.** That is
all. There is no ambient block reciting the graph's inputs under every node's instructions.

```python
GraphNode(
    id="researcher",
    kind="base",
    instructions="Research {topic} and produce 5 bullet points.",   # names its input
)

GraphNode(
    id="writer",
    kind="base",
    instructions="Write a 2-paragraph explanation from the research notes above.",
    depends_on=["researcher"],     # names nothing; the edge carries the notes
)
```

A step that needs a graph input must **interpolate it**. A step that reads an upstream output needs
nothing but the edge.

!!! warning "This changed, and it fails silently"
    A workflow written against the older behaviour still validates and still runs — the model just
    answers as though it had been handed nothing. See
    [Upgrading](../about/upgrading.md#1-a-node-is-told-what-it-names-nothing-ambient).

### Why not simply pass everything

It could not be made correct, only less wrong. A `map` body was handed the whole collection it was
iterating, once per item, beside the single item it was working on. A value the instruction had
already interpolated was printed again underneath it. And any rule for "which inputs matter to this
node" is a worse version of the one the author already wrote — in the placeholders of the
instruction.

## Templates

`{name}` in a node's instructions is interpolated before the call. It resolves against:

| Reference | Source |
|---|---|
| `{topic}` | A declared graph input. |
| `{item}` | The current item inside a `map` body (name set by `item_var`). |
| `{feedback}` | Inside a `loop`, the reason the previous iteration did not stop. |
| `{name}` | Any value stored by a `writes:` on an earlier node. |

A declared input that **no step names** is reported by the validator, because the run would
otherwise go green while the model answers as though it had been passed nothing.

A run handed a value **no step reads** logs a warning naming the key and the fix. It is
deliberately quiet where it cannot know: a `function`, `python` or `tool` node is handed the whole
mapping as keyword arguments, so any key could be the one it takes, and their presence silences the
check rather than risk crying wolf on a working graph.

## Storing a value: `writes`

`writes: "<name>"` stores a node's output under that name, making it available to downstream
templates as `{name}` and to expressions as `vars.name`.

```python
GraphNode(id="classify", kind="base", instructions="Label the sentiment of: {review}",
          writes="label")
GraphNode(id="respond", kind="base", instructions="Write a reply for a {label} review.",
          depends_on=["classify"])
```

## Expressions

`when`, `cases` and `over` are evaluated by a **safe evaluator** — no `eval`, no imports, no
attribute access. Read state as `inputs.x`, `nodes.<id>`, `vars.<name>`. See
[Control flow](control-flow.md#expressions).

## The Python sidecar: `functions:`

A graph can carry a Python sidecar, named at the top of the graph and resolved relative to the
graph file:

```yaml
functions: helpers.py
```

It is copied with the package on export — the same self-containment `nodes/<id>.py` already gives
`function` nodes, but declared once so YAML can then use bare names (a loop's `until`, for
instance).

**A sidecar is imported by path and therefore stands alone: no relative imports.**

What your callables receive — and the `**kwargs` rule a `function` node makes non-optional — is on
[Python in a graph](functions.md).

## Secrets

A credential must never reach a prompt. Interpolated into a node's goal it would be in the request,
in the trace, in the run record, and in whatever the trace is exported to — none of which are
places anyone decided to put it.

So **secrets live outside the interpolation scope entirely.**

### Two syntaxes, two destinations

| Syntax | Reaches | Resolves to a secret? |
|---|---|---|
| `{NAME}` | The prompt | **No** — secrets were never in the mapping the prompt is rendered against. |
| `${NAME}` | `tool_args`, on a node that declared it in `secrets:` | Yes. |

```yaml
- id: query
  kind: tool
  tools: [sql]
  secrets: [WAREHOUSE_DSN]          # the declaration is the authorisation
  tool_args:
    operation: query
    dsn: ${WAREHOUSE_DSN}           # reaches the tool, never the model
    sql: "SELECT count(*) FROM orders"
```

### The declaration is a gate, not documentation

A `${NAME}` the node did **not** declare in `secrets:` is left exactly as written rather than
expanded. Honouring a reference the node never claimed would make `secrets:` documentation instead
of a gate.

### Redaction covers the record, not the call

The tool is given the real value; the **record** of the call is masked. Redaction applies to
strings only — a secret that reached a non-string field is a different bug, and silently rewriting
it would hide it. Values shorter than six characters are not masked, because blanking every
occurrence of `"1"` would turn every digit in a trace into a mask while protecting nothing.

### Bound arguments on a `react` node

`tool_args` on a `base` or `react` node are supplied on **every** call the model makes and
**removed from the schema it is offered**. That is how a credential reaches a tool on an agentic
node without the model ever seeing the field exists.

## Next

- [Node kinds](node-kinds.md) — which kinds accept `secrets` and `tool_args`.
- [Validation](validation.md) — what is checked before a run starts.
- [Workflow packages](packages.md) — what a package carries, and what it requires supplied.
