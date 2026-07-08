# Background Tasks

`TasksRuntime` runs coroutines **in the background** while an agent keeps working, and hands you a
`TaskHandle` to check status or collect the result later. It's the primitive behind long-running or
fire-and-forget work that shouldn't block the main run.

## Submitting work

```python
from neurosurfer.agents.runtime.tasks_runtime import TasksRuntime

runtime = TasksRuntime(max_concurrent=8)

handle = runtime.submit(
    do_expensive_thing(),          # a coroutine
    name="ingest-docs",            # optional; enforces no-overlap by name
    description="Ingesting the docs corpus",
)
```

`submit` returns a `TaskHandle`, or **`None`** if a guard blocked it — either the global
`max_concurrent` cap is reached, or a task with the same `name` is already live (the no-overlap
guard). A blocked coroutine is closed cleanly, so you never get an "coroutine was never awaited"
warning.

## The handle

```python
handle.done            # bool — has it finished?
handle.status          # HandleStatus — running / done / error / cancelled
handle.error           # the exception, if it failed
handle.result_value    # the return value once done (without awaiting)
await handle.result()  # await completion and get the value
handle.cancel()        # request cancellation
```

## Inspecting and shutting down

```python
runtime.active()             # handles still running
runtime.all()                # every handle this runtime has seen
runtime.is_live("ingest")    # is a named task currently running?
runtime.get_live("ingest")   # the live handle for a name, or None
runtime.cancel("ingest")     # cancel a named task
await runtime.shutdown()     # cancel + await all outstanding tasks
```

!!! tip "Name tasks you don't want to double-start"
    Passing `name=` makes `submit` idempotent-ish: while one `"ingest-docs"` is live, a second submit
    with the same name is refused (returns `None`) instead of running a duplicate.

## When to use it

- Kick off a slow ingest, export, or fetch and keep the conversation responsive.
- Run several independent jobs under a concurrency cap.
- Coordinate work that outlives a single agent turn.

For parallelism that's *part of the reasoning* (analyse N things and aggregate), prefer
[sub-agents](subagents.md) — they nest in the trace and share guardrails. Reach for `TasksRuntime`
when the work is plumbing rather than agent reasoning.
