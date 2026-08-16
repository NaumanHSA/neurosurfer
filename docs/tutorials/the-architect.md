# The Architect

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NaumanHSA/neurosurfer/blob/main/tutorials/06_the_architect.ipynb)

Hand an agent a plain-English intent and get back a registered, validated, **tested** workflow
package — one it built, ran, judged against criteria it derived itself, and repaired where it
failed.

## Minimal walkthrough

```python
from neurosurfer.architect import ArchitectAgent
from neurosurfer.graph.workflow.registry import WorkflowRegistry

agent = ArchitectAgent(
    provider,
    registry     = WorkflowRegistry(workflows_dir="./registry"),
    staging_root = "./staging",
    notify       = print,          # watch it work
)

path = await agent.build(
    "Read a text file of customer feedback, pull out the recurring complaints, "
    "and write a short summary for the support lead."
)
```

The result is an ordinary [WorkflowPackage](graph-agents.md) — run it like any other:

```python
from neurosurfer.graph.workflow.package import load_package
from neurosurfer.graph.workflow.runner import WorkflowRunner

pkg = load_package(path)
result = WorkflowRunner(provider).run(pkg, {"feedback_file_path": "feedback.txt"})
```

## Two agents, not one

Planning and building are deliberately separate calls, and the capability ladder between them runs
**in code** rather than being asked of the model:

| Stage | Call | What it guarantees |
|---|---|---|
| Plan | `plan_and_resolve` | one structured answer — the shape a weak model is reliable at |
| Ladder | `resolve_capability` | tools chosen in Python; the model never guesses a tool name |
| Build | `ArchitectAgent.build` | one node at a time, warnings read after every call |
| Gate | `validate_package` | errors block registration |
| Proof | `derive_acceptance` + `verify_workflow` | it ran, on a real fixture, judged fail-closed |

Holding all of that in one stream is what makes a small model drop requirements. By the time the
builder sees the plan, a step that needs a file already has `read_file` attached to it.

## When it refuses

A request needing a capability nothing provides raises `WorkflowInfeasible` with a checklist of
what is missing, rather than registering a workflow that pretends:

```python
from neurosurfer.architect import WorkflowInfeasible

try:
    await agent.build("Delete duplicate rows from my production Oracle database.")
except WorkflowInfeasible as e:
    print(e)     # names the specific gap, and two concrete routes forward
```

Needing to summarise, classify or write is never a reason to refuse — that is what a `base` node is
for. Only a genuinely absent capability blocks.

## Full notebook

The [Colab notebook](https://colab.research.google.com/github/NaumanHSA/neurosurfer/blob/main/tutorials/06_the_architect.ipynb)
runs the planner alone, builds a workflow end to end on a local 9B model, reads the `graph.yaml` it
wrote, demonstrates the validation gate, drives the verification engine directly, and shows a
refusal — about 3–10 minutes.

**Next:** [Architect reference](../architect/index.md) · [Graph agents](graph-agents.md)
