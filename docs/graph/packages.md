# Workflow Packages

A **Workflow package** is a graph saved as a versioned, multi-file package you can persist, share,
register, and run later. It is the unit the [gateway](../server/index.md), the
[CLI](../cli/index.md), and the [Architect](../architect/index.md) all deal in.

## Load and run

```python
from neurosurfer.graph.workflow import load_package, WorkflowRunner

pkg = load_package(pkg_dir)
print(pkg.name, pkg.version, [n.id for n in pkg.graph.nodes])

runner = WorkflowRunner(provider, cwd=repo_root)   # cwd = working dir for tool contexts
result = runner.run(pkg, inputs={"topic": "gradient descent"})

print(result.execution_summary())
print(result.final.get("writer", "(none)"))        # output of the 'writer' node
```

**Pass the names the package declares**, not a generic intent:

```python
print([i.name for i in pkg.graph.inputs])          # e.g. ['article']
result = runner.run(pkg, inputs={"article": "…"})
```

A run handed a value no step reads logs a warning naming the key and the fix — see
[State](state.md#templates).

## What a package carries

Beyond `graph.yaml`, a package is self-contained by design:

| File | Holds |
|---|---|
| `graph.yaml` | The graph IR — nodes, inputs, outputs, control flow. |
| `nodes/<id>.py` | Per-node Python for `function` nodes. |
| `<functions>.py` | The [sidecar](state.md#the-python-sidecar-functions), if the graph declares one. |
| metadata | Name, version, description, tags, author. |

The sidecar and per-node Python are **copied with the package on export**, so a package that runs
here runs there.

## Register

`WorkflowRegistry` stores packages so you can look them up by name and run them anywhere:

```python
from neurosurfer.graph.workflow import WorkflowRegistry, WorkflowRunner

registry = WorkflowRegistry()
pkg = registry.get("content_pipeline")

result = WorkflowRunner(provider, cwd=repo_root).run(
    pkg, inputs={"topic": "RNNs vs Transformers"},
)
```

Use `save_package` / `load_package` to move packages between the filesystem and the registry.

**Nothing registers unless [validation](validation.md) passes.**

## Where packages live

Under the single data root — `./.neurosurfer/workflows/` unless `NEUROSURFER_HOME` says otherwise.
Registered workflows are **host-level**: anything that can reach this installation can see all of
them. See [Storage layout](../guides/configuration.md#storage-layout).

## What a package requires supplied

A registered workflow records what has to be set before it can run, derived rather than stored:

```python
from neurosurfer.graph.workflow.requirements import workflow_requirements, missing_requirements
```

See [Validation](validation.md#what-a-workflow-requires-supplied).

## Running one over HTTP

The gateway exposes registered packages at `/v1/workflows`, with runs at `/v1/runs` and an SSE
event stream. See the [Workflows API](../server/index.md).

## Next

- [Validation](validation.md) — the gate a package must pass.
- [Building in Python](building.md) — authoring the graph a package wraps.
- [The Architect](../architect/index.md) — having a package designed for you.
