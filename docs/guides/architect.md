# Architect

The **Architect** turns a plain-English description into a runnable
[Workflow package](graph-workflows.md#workflow-packages). It designs the graph, and — if the
workflow needs a tool that doesn't exist yet — it authors that tool, validates it in a sandbox, and
registers it (with your approval). It lives in `neurosurfer.architect`.

```python
from neurosurfer.architect import ArchitectBuilder, ArchitectConversation, WorkflowInfeasible
```

## Build a workflow from intent

`ArchitectBuilder.run(intent)` designs and registers a workflow, returning the path to the
registered package:

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

Once registered, run it like any other workflow:

```python
from neurosurfer.graph.workflow import WorkflowRegistry, WorkflowRunner

pkg = WorkflowRegistry().get(pkg_path)   # or load_package(pkg_path)
result = WorkflowRunner(provider, cwd=".").run(pkg, inputs={"user_intent": "…"})
```

### Optional callbacks

`run()` accepts hooks for richer front-ends:

- `answers` — pre-collected clarifying answers (`question_id → answer`) so the build runs
  non-interactively.
- `on_node_event(node_id, status)` / `progress_callback(node_id, status, duration_ms)` — live and
  post-run progress.
- `approve_tool(draft, sandbox_result) -> bool` — called **only** when the Architect hits a
  capability gap and needs to author a new tool. It authors the tool, runs it in a sandbox, and
  registers it only if this callback returns `True`.

## Clarify requirements first

For interactive apps, `ArchitectConversation` runs a short requirements interview and returns
`(intent, answers)` you can feed straight into `ArchitectBuilder.run(...)`:

```python
from neurosurfer.architect import ArchitectConversation

async def ask(question: str, choices: list[str]) -> str:
    # render a menu / prompt and return the user's answer
    ...

convo = ArchitectConversation(provider)
intent, answers = await convo.run("I want a workflow that reviews pull requests", ask=ask)

pkg_path = await ArchitectBuilder(provider).run(intent, answers=answers)
```

This is exactly how the [CLI](../cli.md) drives its workflow builder — the REPL supplies `ask` (an
arrow-key menu with a free-text escape) and then builds the designed workflow.

!!! note "Runtime vs. authoring"
    The Architect is the **authoring** layer; [`neurosurfer.graph`](graph-workflows.md) is the
    **runtime**. The runtime never imports the authoring layer, so shipping/running a workflow
    doesn't pull in the Architect.
