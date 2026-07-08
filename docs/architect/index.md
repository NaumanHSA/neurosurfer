# Architect

The **Architect** turns a plain-English description into a runnable
[Workflow package](../guides/graph-workflows.md#workflow-packages). You describe *what* you want; it
designs the graph, writes the node logic, and — if the workflow needs a tool that doesn't exist yet —
**authors that tool**, validates it in a sandbox, and registers it (with your approval).

It lives in `neurosurfer.architect`:

```python
from neurosurfer.architect import ArchitectBuilder, ArchitectConversation, WorkflowInfeasible
```

!!! warning "Experimental — not yet production-grade"
    The Architect is the newest and least settled module in Neurosurfer. It works for well-scoped,
    single-purpose workflows, but **generated graphs vary in quality**, complex or ambiguous intents
    can produce brittle or infeasible designs, and the authored-tool path is best treated as a
    starting point you review — not code to ship unread. The APIs on these pages are stable enough to
    build on, but expect the behaviour and prompts behind them to keep changing. Use it to
    *bootstrap* a workflow, then refine the generated package by hand.

## The idea

Writing a [graph or Workflow package](../guides/graph-workflows.md) by hand means choosing nodes,
wiring dependencies, and picking tools. The Architect does that first draft for you: it reasons about
your intent, drafts a plan, checks whether the required capabilities exist, fills gaps, and assembles
a validated package you can run — or hand-edit.

Under the hood the Architect is **itself a workflow** — a fixed pipeline of LLM-driven nodes that
produces *other* workflows. That's why it's a good stress-test of the graph engine, and why its
output quality tracks the model you give it.

## Quick start

```python
from neurosurfer.architect import ArchitectBuilder, WorkflowInfeasible

builder = ArchitectBuilder(provider)
try:
    pkg_path = await builder.run(
        "Summarise a web article and extract the 5 key takeaways as a bullet list.",
    )
    print("registered workflow at:", pkg_path)
except WorkflowInfeasible as e:
    print("cannot build this workflow:", e)
```

Then run the result like any other workflow (see [Building Workflows](building.md)).

## In this section

- **[How It Works](how-it-works.md)** — the build pipeline, tool authoring, and feasibility checks.
- **[Building Workflows](building.md)** — the `ArchitectBuilder` / `ArchitectConversation` API,
  clarifying questions, callbacks, and running what it produces.

!!! note "Authoring vs. runtime"
    The Architect is the **authoring** layer; [`neurosurfer.graph`](../guides/graph-workflows.md) is
    the **runtime**. The runtime never imports the authoring layer, so shipping or running a generated
    workflow doesn't pull in the Architect.
