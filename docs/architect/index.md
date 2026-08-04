# Architect

The **Architect** turns a plain-English description into a runnable
[Workflow package](../guides/graph-workflows.md#workflow-packages). You describe *what* you want; it
designs the graph, writes the node logic, and — if the workflow needs a tool that doesn't exist yet —
**authors that tool**, validates it in a sandbox, and registers it (with your approval).

It lives in `neurosurfer.architect`:

```python
from neurosurfer.architect import ArchitectBuilder, ArchitectConversation, WorkflowInfeasible
```

!!! note "Maturing — output quality tracks the model"
    The Architect is the most demanding agent in Neurosurfer. It now **runs and LLM-judges its own
    builds** before finishing (closed-loop self-verification), which sharply improves reliability —
    but **generated-graph quality still tracks the model you give it**: strong tool-calling models
    (e.g. `gpt-5-mini` and up) produce solid, branching designs; smaller models occasionally emit a
    simpler-than-ideal graph and may need a re-run or the gated `verify="required"` loop. Treat the
    **authored-tool path** as a reviewed starting point, not code to ship unread. APIs on these pages
    are stable; the prompts and heuristics behind them keep improving.

!!! tip "The current entrypoint: `ArchitectAgent`"
    These pages document the original `ArchitectBuilder` pipeline. The recommended path is now the
    **ReAct `ArchitectAgent`** — a single planner with a validate/test/register toolbelt and
    closed-loop verification (`ArchitectAgent(provider).build(intent)`). See
    **[tutorial 06 — The Architect](https://github.com/NaumanHSA/neurosurfer/blob/main/tutorials/06_architect.ipynb)**
    for the end-to-end walkthrough.

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
