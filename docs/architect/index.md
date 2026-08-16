# Architect

The **Architect** turns a plain-English description into a runnable
[Workflow package](../graph/packages.md). You describe *what* you want; it
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
    closed-loop verification:

    ```python
    from neurosurfer.architect import ArchitectAgent

    # async — returns the registered package path, or raises WorkflowInfeasible
    path = await ArchitectAgent(provider).build(
        "Summarise a CSV and write the result to a file"
    )
    ```

## The idea

Writing a [graph or Workflow package](../graph/index.md) by hand means choosing nodes,
wiring dependencies, and picking tools. The Architect does that first draft for you: it reasons about
your intent, drafts a plan, checks whether the required capabilities exist, fills gaps, and assembles
a validated package you can run — or hand-edit.

Three properties are worth stating up front, because they are what separates this from a prompt that
emits YAML:

- **It plans first.** A plan grounds every external step before a node exists.
- **It grounds, and it refuses.** Every capability it names is checked against what actually
  exists. A request it cannot build is reported *before* a node is designed for it.
- **It proves its work by running it.** What it builds is verified by being executed and judged
  against derived acceptance criteria — not by looking plausible.

## Quick start

```python
from neurosurfer.architect import ArchitectAgent, WorkflowInfeasible

agent = ArchitectAgent(provider)
try:
    pkg_path = await agent.build(
        "Summarise a web article and extract the 5 key takeaways as a bullet list.",
    )
    print("registered workflow at:", pkg_path)
except WorkflowInfeasible as e:
    print("cannot build this workflow:", e)
```

Then run the result like any other workflow (see [Building Workflows](building.md)).

## In this section

- **[The Agent](agent.md)** — `ArchitectAgent`, its toolbelt, and the knobs on a build.
- **[Grounding & Refusal](grounding.md)** — how a capability is checked, and why a refusal is a
  feature.
- **[Verification](verification.md)** — acceptance criteria, fixtures, and the judge.
- **[Self-knowledge](knowledge.md)** — what the Architect knows about this installation.
- **[How It Works](how-it-works.md)** — the older `ArchitectBuilder` pipeline.
- **[Building Workflows](building.md)** — clarifying questions, callbacks, and running the output.

!!! note "Authoring vs. runtime"
    The Architect is the **authoring** layer; [`neurosurfer.graph`](../graph/index.md) is
    the **runtime**. The runtime never imports the authoring layer, so shipping or running a generated
    workflow doesn't pull in the Architect.
