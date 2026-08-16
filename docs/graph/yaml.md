# Authoring in YAML

A graph is **data**. YAML is its native form — what a [workflow package](packages.md) stores on
disk, what the [Architect](../architect/index.md) writes, and what you edit by hand. Everything the
Python API can express, YAML can express.

```yaml
name: content_pipeline
description: Research a topic, then explain it.

inputs:
  - name: topic
    type: string
    required: true
    description: What to research.

nodes:
  - id: researcher
    kind: base
    instructions: Research {topic} and produce exactly 5 key bullet points.

  - id: writer
    kind: base
    instructions: Write a clear, 2-paragraph explanation from the research notes above.
    depends_on: [researcher]

outputs: [writer]
```

Load it with `load_graph`, or let a [package](packages.md) carry it:

```python
from neurosurfer.graph import load_graph

graph = load_graph("graph.yaml")
```

## Top-level keys

| Key | Required | Means |
|---|---|---|
| `name` | **yes** | The graph's identifier. |
| `nodes` | **yes** | The list of nodes. Order does not matter — `depends_on` defines the DAG. |
| `description` | no | What the workflow is for. |
| `inputs` | no | Declared inputs. See below. |
| `outputs` | no | Node ids whose outputs the graph returns. Superseded by an [`output` node](node-kinds.md#output) if one exists. |
| `functions` | no | A [Python sidecar](functions.md) file, resolved relative to this file. |
| `fail_fast` | no | Stop the whole run at the first node failure instead of continuing down live branches. |
| `strict_inputs` | no | Refuse a run that supplies a key no input declares. |

## Declaring inputs

```yaml
inputs:
  - name: topic
    type: string
    required: true
    description: What to research.
  - name: max_items
    type: integer
    required: false
```

`type` is `string`, `integer`, `number`, `boolean`, `array`, or `object`.

**A declared input that no step names is reported by the validator** — the run would otherwise go
green while the model answers as though it had been passed nothing. Interpolate it somewhere, or
drop the declaration.

## Node keys

Every node takes `id` (required) and `kind` (defaults to `base`). The rest depends on the kind —
see [Node kinds](node-kinds.md) for which fields each one actually uses.

### Instructions

```yaml
- id: writer
  kind: base
  instructions: Write a 2-paragraph explanation of {topic}.
```

`instructions` is the current field. The older trio is **still read** when `instructions` is
absent, which is why you will see it in existing packages:

```yaml
  purpose: You are a careful technical writer.
  goal: Write a 2-paragraph explanation of {topic}.
  expected_result: Two paragraphs, no bullet points.
```

Prefer `instructions` in new graphs.

### Wiring

```yaml
  depends_on: [researcher, scout]   # waits for both; receives both outputs
  when: "len(nodes.scout) > 0"      # run only if truthy
  on_error: fallback_node           # reroute on failure
  writes: draft                     # store output as {draft} / vars.draft
  policy:
    retries: 2
```

### Tools

```yaml
- id: fetch
  kind: tool
  tools: [http]
  tool_args:
    url: "{source_url}"
    method: GET
  tool_settings:
    http:
      timeout: 30
  secrets: [API_TOKEN]
```

`tool_args` are the call arguments; `tool_settings` are the author-configured settings keyed by tool
name. A `${NAME}` in `tool_args` resolves only if the node declared `NAME` in `secrets` — see
[State & secrets](state.md#secrets).

### Structured output

```yaml
- id: extract
  kind: base
  instructions: Extract the invoice fields from {document}.
  output_schema: my_package.models:Invoice
  mode: json
```

`output_schema` is an import path, `module:Attr`, resolved and checked by validation. Without one,
a node asked for an object returns JSON **as a string**.

### Per-node model

```yaml
  provider: openai
  model: gpt-4o-mini
```

Omit both to use the run's provider.

## Control flow in YAML

### Router

```yaml
- id: triage
  kind: router
  instructions: "Route this support ticket by urgency: {ticket}"
  routes:
    urgent: escalate
    routine: reply
  default: reply
  repair: true

- id: escalate
  kind: base
  depends_on: [triage]        # every target MUST depend on the router
  instructions: Draft an escalation for {ticket}.
```

Deterministic form, no LLM call:

```yaml
- id: gate
  kind: router
  depends_on: [check]
  cases:
    - when: "contains(lower(nodes.check), 'yes')"
      to: approve
  default: reject
```

Declare `routes` **or** `cases`, never both.

### Loop

```yaml
functions: helpers.py

nodes:
  - id: polish
    kind: loop
    max_iterations: 4
    until: tagline_is_short        # a name helpers.py defines
    body:
      - id: draft
        kind: base
        instructions: "Write a tagline. {feedback}"
```

`until` is either a **function name from the sidecar** or **plain English**. Which one it is, is a
lookup — see [Python in a graph](functions.md#until-a-function-or-a-sentence).

### Map

```yaml
- id: per_item
  kind: map
  over: inputs.items
  item_var: item
  concurrency: 4
  body:
    - id: handle
      kind: base
      instructions: "Process one item: {item}"
```

### Nested bodies

`loop`, `map` and `subgraph` take a `body:` — a full list of nodes with the same schema, nested.
Body nodes may only depend on their **siblings**; a dependency pointing outside the body is refused
when the graph loads.

## Two YAML details that bite

**Braces need quoting.** A value starting with `{` is a YAML flow mapping, not a string:

```yaml
instructions: {topic} summary          # ✗ YAML parse error
instructions: "{topic} summary"        # ✓
instructions: Summary of {topic}       # ✓ — brace is not first
```

**Multi-line prose wants a block scalar.** Use `>` to fold or `|` to keep newlines:

```yaml
  instructions: >
    Research {topic} thoroughly and produce exactly five bullet
    points, each one sentence long.
```

## Round-tripping

YAML and the Python API produce the **same** `Graph`, and the loader validates both identically:

```python
from neurosurfer.graph import load_graph_from_dict
load_graph_from_dict(graph.model_dump(mode="json"))   # reproduces graph
```

That is deliberate: there is no second code path, so no class of bug reaches one form and not the
other. See [Building in Python](building.md).

## Next

- [Node kinds](node-kinds.md) — every field, by kind.
- [Python in a graph](functions.md) — the sidecar and what your functions receive.
- [Control flow](control-flow.md) — the semantics behind the syntax above.
- [Validation](validation.md) — what is checked when this file loads.
