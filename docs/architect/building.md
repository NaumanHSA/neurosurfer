# Building Workflows

The API to design, review, and run a workflow from intent. Two entry points:
`ArchitectBuilder` (design + register) and `ArchitectConversation` (an optional requirements
interview).

## Build from intent

`ArchitectBuilder.run(intent)` designs and registers a workflow, returning the path to the registered
package:

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

### Callbacks

`run()` accepts hooks for richer front-ends:

| Argument | Purpose |
|---|---|
| `answers` | Pre-collected clarifying answers (`question_id → answer`) so the build runs non-interactively. |
| `on_node_event(node_id, status)` | Fired live as each build node runs — for progress UI. |
| `progress_callback(node_id, status, duration_ms)` | Post-run per-node summary. |
| `approve_tool(draft, sandbox_result) -> bool` | Called **only** on a capability gap: the Architect authored a tool and sandbox-tested it; return `True` to register it. See [How It Works](how-it-works.md#authoring-missing-tools). |

## Clarify requirements first

For interactive apps, `ArchitectConversation` runs a short requirements interview and returns
`(intent, answers)` you feed straight into `ArchitectBuilder.run(...)`:

```python
from neurosurfer.architect import ArchitectConversation

async def ask(question: str, choices: list[str]) -> str:
    # render a menu / prompt and return the user's answer
    ...

convo = ArchitectConversation(provider)
intent, answers = await convo.run("I want a workflow that reviews pull requests", ask=ask)

pkg_path = await ArchitectBuilder(provider).run(intent, answers=answers)
```

This is exactly how the [CLI](../cli/index.md) drives its workflow builder — the REPL's
`/workflow` command supplies `ask` (an arrow-key menu with a free-text escape), then builds the
designed workflow.

## Run what it produced

A registered package runs like any other [Workflow](../guides/graph-workflows.md#workflow-packages):

```python
from neurosurfer.graph.workflow import WorkflowRegistry, WorkflowRunner

pkg = WorkflowRegistry().get(pkg_path)                 # or load_package(pkg_path)
result = WorkflowRunner(provider, cwd=".").run(pkg, inputs={"user_intent": "…"})
```

## Recommended workflow

Given the module is [experimental](index.md), the reliable path is:

1. **Bootstrap** — let the Architect design a first draft from a *narrow, specific* intent.
2. **Review** — open the generated package; read the graph and any authored tool.
3. **Refine** — edit the nodes/tools by hand where the draft is weak.
4. **Run** — execute the refined package through the [graph runtime](../guides/graph-workflows.md).

Narrow intents (“summarise a URL and extract N bullets”) work far better than broad ones (“build me a
research assistant”). Split big goals into smaller workflows.
