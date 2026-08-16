# Validation

`load_package` proves a package is *structurally* loadable — schema, node kinds, DAG acyclicity. It
does not prove the package can **run**: tool names may be invented, import paths may not resolve,
edges may point at nodes that do not exist, and a prompt may read a variable nothing produces.

`validate_package` is the gate.

```python
from neurosurfer.graph.workflow.validation import validate_package

report = validate_package(pkg)
print(report.ok)             # False if anything is an ERROR
for issue in report.errors:
    print(issue.message)     # a plain sentence
    print(issue.detail)      # field names, import paths, parser errors
```

## Validation runs on every run

Not only at registration. **A graph that cannot run is refused before a model is called**, rather
than partway through — so a broken workflow costs nothing instead of costing every node up to the
one that failed.

## Nothing registers unless it passes

Six callers use this gate, and the studio is only one of them: the Architect validates every build
against it, its ReAct agent checks its own work with it, and the eval harness scores against it.

**A rule added here is a rule the Architect obeys.**

## Rules are declared, not sequenced

Each rule says **which node kinds it speaks about** and **at what severity**. What that buys,
beyond tidiness, is that *"what can go wrong with an input node"* becomes a query rather than a
careful read of a thousand lines.

```
validation/
    models.py     Severity · ValidationIssue · ValidationReport
    registry.py   @node_rule / @graph_rule, and the rule tables
    context.py    ids, dependencies, the tool registry — derived once
    graph.py      rules about the workflow as a whole
    templates.py  the scope walk: placeholders, secrets, tool args
    nodes/        rules about a single node, by concern
```

## Messages are plain; details are separate

`message` is one sentence, for a person — no field names, no import paths. `detail` carries the
technical half: the field, the import path, the parser error.

They are two fields because a message cannot be made plain without deleting the half a developer
needs. `output_schema 'my:Model' does not import` names two things the author never typed.

## Severity

| Severity | Means |
|---|---|
| `ERROR` | The package will not register and the run is refused. |
| `WARNING` | It will run; something is likely wrong. |
| `INFO` | Worth knowing. |

`ValidationReport` holds **one list** of issues, each carrying its own severity, read through
views: `report.errors`, `report.gaps`, `report.warnings`, `report.info`, and `report.ok`.

A **gap** is an error of a particular identity — a missing capability rather than a malformed
field — shown and handled separately because the fix is different: you supply a tool, rather than
correcting a line.

## An input no step reads is an error

A workflow that **declares** an input and never names it accepts a parameter and ignores it. The
caller passes their article, no step interpolates it, the run goes green, and the answer is
confident and unrelated.

```
The workflow asks for 'article' but no step uses it, so the value a caller passes is ignored.
  → Name it in a step's instructions as {article}, or drop it from the workflow's inputs.
```

This blocks. It used to warn, and the backstop was a person noticing the answer had nothing to do
with what they passed — which is no backstop at all for a workflow the
[Architect](../architect/index.md) builds and verifies on its own.

**Every way a value can be read counts**, not just prompt placeholders: a `map`'s `over`
expression, a `when` guard, `tool_args`, an output node's `value`, a code node's parameter names,
and nodes nested inside container bodies. A rule that only looked at `instructions` would report a
perfectly good fan-out as ignoring its collection.

**It downgrades itself to a warning where it cannot see.** A `tool` node's arguments live in a
registered schema and a callable may fail to import or inspect; either hides the reads that would
clear the input. Refusing to run over a fact that was never established is worse than the gap, so
the rule keeps its voice and loses its veto — and the detail says which.

!!! warning "This can stop a workflow you already have"
    It is the one validation change that breaks something already on disk rather than in source.
    Run `validate_package` over your registry before upgrading; see
    [Upgrading](../about/upgrading.md).

## Capability grounding

The check that decides whether a workflow can run at all. A node whose goal is *"read the file"*
and whose tool list is empty is one LLM call being asked to produce the contents of a file it
cannot open. That graph validates clean and hallucinates at run time.

Two checks supply the missing question:

- **A toolless `react` node** — structural and unambiguous. A `react` node is *defined* as an LLM
  that calls tools in a loop; with no tools it is a `base` node in a costume.
- **A suspected capability** — lexical. Prompt text describing a reach outside the model (read a
  file, fetch a URL, check an inbox, send a message) on a node holding no tool.

Three rules keep the lexical half from becoming noise, each learned from a real false positive:

1. **Verb→object scoping.** "read the file" is external; "read the summary" is not. Bare verbs
   match everything.
2. **The verb must open a clause.** *"Analyze the content read from the file"* describes where its
   input came from, not a request to open anything.
3. **Only fields that state an action are read.** `expected_result` describes the *output* — "List
   of important unread emails" is a noun phrase about shape.
4. **A wired-up node may already have been handed its data.** *"Analyze the content of the file"* on
   a node that `depends_on` the node which read that file is correct as written.

Rule 4 is deliberately generous, and the trade is asymmetric: a missed warning leaves us where we
started, while a wrong one derails a build.

!!! note "Sources are forgiven; sinks are not"
    Rule 4 only forgives **source** capabilities — getting data in. No upstream node can send an SMS
    on your behalf, so sinks are always flagged.

## What a workflow requires supplied

A registered workflow records what has to be set before it can run — `SQLITE_DB_PATH (required)`,
and so on. Without it, a missing value surfaced as a connection error on the first node: the least
informative place and the latest possible moment.

```python
from neurosurfer.graph.workflow.requirements import workflow_requirements, missing_requirements
```

**Derived, never stored.** A copy in the package would be a second source of truth that drifts the
moment a server is reconfigured. Both facts it reads — the `${VAR}`s in a server's config and the
`secrets:` on a node — are already authoritative somewhere, so it asks them.

A requirement carries whether it is `satisfied`, and a `problem` when a value that **is** set still
cannot be used — *"is not a SQLAlchemy connection URL"*. Set is not the same as usable, and
conflating them let a hostname left over from an abandoned naming scheme satisfy a requirement
silently.

Enabled MCP servers are included whether or not the workflow names one of their tools, because
starting a run connects every enabled server — one of them missing a value fails the run regardless
of which tools it was going to use.

## What was removed

The **"fewer than three LLM nodes is almost certainly under-designed"** heuristic sat in the gate
that decides whether a package may register. A correct two-step workflow was refused. It is gone.

## Next

- [Node kinds](node-kinds.md) — the constraints each kind declares.
- [State & secrets](state.md) — the scope walk validation performs over placeholders.
- [Workflow packages](packages.md) — registering a validated package.
