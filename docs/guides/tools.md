# Tools

Tools are the actions an agent can take. Neurosurfer ships a curated pool of built-in tools and a
small framework for writing your own. The framework lives in `neurosurfer.tools`; the built-in
tools live in `neurosurfer.registry.core.<domain>`.

## The tool pool

`default_pool()` returns a `ToolPool` containing all built-in tools; `build_pool(names)` narrows it
to an allow-list:

```python
from neurosurfer.tools import default_pool, build_pool, all_tools

pool = default_pool()                       # every built-in tool
safe = build_pool(["read_file", "list_dir", "search"])   # read-only subset
```

Because the pool is small, all selected tool schemas are sent to the model every turn — there's no
deferred discovery step.

See the [Tools Catalog](tools-catalog.md) for what ships.

## Writing a custom tool

Subclass `Tool`, declare a Pydantic `input_model`, and implement `call()`:

```python
from pydantic import BaseModel, Field
from neurosurfer.tools import Tool, ToolResult, ToolContext

class AddArgs(BaseModel):
    a: float = Field(description="first number")
    b: float = Field(description="second number")

class AddTool(Tool):
    name = "add"
    title = "Add numbers"
    description = "Add two numbers and return the sum."
    input_model = AddArgs

    def is_read_only(self, args) -> bool:
        return True   # pure, side-effect-free ⇒ concurrency-safe

    async def call(self, args: AddArgs, ctx: ToolContext) -> ToolResult:
        return ToolResult.ok(str(args.a + args.b))
```

Build a pool that includes it and hand that pool to an agent:

```python
from neurosurfer.tools import ToolPool, all_tools

pool = ToolPool([*all_tools(), AddTool()])   # built-ins + your tool
# or just your own tools:  ToolPool([AddTool()])
```

## What a tool declares

### Identity

| Field | Purpose |
|---|---|
| `name` | **The identifier** — what a graph writes in `tools:`, what the model is offered, what every log line says. Stable, snake_case. |
| `title` | What a **person** calls it. Empty means "derive it" (`apply_edit` → "Apply Edit"). |
| `description` | What the model reads. |
| `icon` | Icon slug. Empty falls back to the tool's registry domain, so an unseen MCP tool still reads as *database* rather than a generic box. |

`title` is kept separate from `name` rather than replacing it: renaming the identifier breaks every
workflow that references it, and a title is exactly the thing you want to be free to reword. A
palette listing `sql`, `http`, `apply_edit` is showing its own vocabulary to somebody who never
agreed to learn it.

### Arguments, in two lifetimes

| Field | Filled in by | When |
|---|---|---|
| `input_model` | The **model** | Every call. |
| `settings_model` | The **author** | Once, at design time. |

They are two declarations rather than one wider one, and the reason is concrete. `write_file` in a
hosted studio had nowhere to say *where files go*: its `path` resolved against the gateway
process's working directory, so an agent asked to write a report wrote into the server's own
checkout, and no surface anywhere could have said otherwise.

Adding `root` to `input_model` would not fix it — **everything in `input_model` is offered to the
model, and a model offered a `root` invents one.** A directory is decided by whoever built the
workflow, identical on every call, and none of the model's business.

`settings_model = None` means "nothing to configure", which is the honest answer for `browse` and
must not become an empty settings panel.

!!! note "A settings class docstring is UI text"
    Pydantic puts it in the schema as `description`, and that is what a front-end renders above the
    fields. Keep it to a sentence somebody would want to read on a panel; the rationale goes in a
    comment.

### Confinement

| Field | Purpose |
|---|---|
| `root_setting` | The `settings_model` field that is this tool's filesystem root. |
| `path_inputs` | Which **call arguments** name a path. |

Everything in `path_inputs` is resolved under `root_setting` and refused outside it. Both are
declared rather than inferred: `search` takes a `pattern` and a `path` and only one is a place on
disk, and a wrapper inspecting values would confine whatever happened to look path-shaped that day.

The two are meaningless apart.

### Credentials

| Field | Purpose |
|---|---|
| `secret_inputs` | Which inputs are secrets — they never travel through a prompt. |
| `credential_help` | What a person needs in order to supply one. |

```python
class SqlTool(Tool):
    name = "sql"
    secret_inputs = frozenset({"dsn"})
    credential_help = (
        "A SQLAlchemy connection URL, e.g. postgresql+psycopg://user:pw@host/db. "
        "Store it in Settings → Secrets and reference it as ${NAME}."
    )
```

A secret reaches the tool and never the model. See
[State & secrets](../graph/state.md#secrets).

Override `check_secret(name, value)` to reject a value that is set but unusable — "is not a
SQLAlchemy connection URL". Set is not the same as usable.

### Capabilities

`capabilities` is the tag set a tool declares itself against, from a **closed vocabulary**:

```python
class ReadFileTool(Tool):
    name = "read_file"
    capabilities = frozenset({"file.read"})
```

This is what a need resolves against — `file.write` → `write_file` — rather than words a
description happens to share. See [Tool Registry](tool-registry.md).

### Where it runs

`runtime` says whether the tool reaches `localhost`. `"in_process"` means the driver runs here and
only the far end is elsewhere, so a DSN pointing at a local container works. A hosted service is a
perfectly good tool that simply cannot see `localhost:1433`.

### Operations

A tool with several distinct actions declares them rather than becoming several tools:

```python
from neurosurfer.tools.base import Operation

class SqlTool(Tool):
    name = "sql"
    operations = {
        "query": Operation(
            title="Run a query",
            description="Run one read-only SQL query and return the rows as a table.",
            input_model=QueryArgs,
            capabilities=frozenset({"db.query"}),
            read_only=True,
        ),
        ...
    }
```

A tool is a **type** of integration — "SQL" — and an operation is a thing it does. Each operation
carries its own `input_model`, so a caller configuring one is shown the arguments *that* operation
needs rather than the union of every operation's arguments with everything optional.

Resolution matches a **tool** when any of its operations declares the tag; the operation says which
one to call.

!!! note "Grouped by type, never by credential"
    A credential is a value a tool *uses*, not what it *is*: it rotates, and two SQL nodes may
    legitimately point at different databases. Grouping by credential would rearrange the palette
    when somebody changed a password.

### Behaviour flags

`is_read_only`, `is_concurrency_safe`, `is_destructive` let the engine run safe tools in parallel
and route destructive ones through the [permission gates](../learn/permissions.md).

## Returning a result

- `ToolResult.ok(text)` on success, `ToolResult.error(text)` on failure.
- `ToolResult.with_images(text, images)` returns screenshots/renders to vision models.
- `progress_message(args)` — override to show a friendly status line ("Reading README.md…") on the
  `ToolStarted` event.

`input_model` validation errors are returned to the model as a **correctable** tool error, so it
can retry.

## The ToolContext

Passed to `call()`; carries the working directory, the `io` handler, and other run state a tool may
need.

## The IOHandler

Tools that need a human decision call the agent's `io` handler — an `IOHandler` with:

```python
async def ask(self, question, options=None) -> str: ...
async def request_plan_approval(self, plan) -> tuple[bool, str]: ...
async def request_shell_approval(self, command, reason) -> bool: ...
async def request_write_approval(self, path, summary) -> str: ...   # "always" | "once" | "deny"
```

Interactive apps back this with a UI; scripts use an auto-approving handler (see the
[Agents guide](agents.md#the-io-handler)). Whether a decision is even requested depends on the
agent's [guardrails and mode](agents.md#permissions-and-guardrails).

## Next

- [Tools Catalog](tools-catalog.md) — every built-in tool.
- [Tool Registry](tool-registry.md) — how a capability finds a tool.
- [MCP](mcp.md) — external tools over the Model Context Protocol.
