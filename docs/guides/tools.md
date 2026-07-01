# Tools

Tools are the actions an agent can take. Neurosurfer ships a curated pool of built-in tools and a
small framework for writing your own. Everything lives in `neurosurfer.tools`.

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

## Built-in tools

| Tool | What it does |
|---|---|
| `read_file` | Read a file from the working directory. |
| `list_dir` | List a directory. |
| `search` | Search files/content in the workspace. |
| `write_file` | Create or overwrite a file (gated by `write_scope`). |
| `apply_edit` | Apply a targeted edit to a file. |
| `run_command` | Run a shell command (gated by `shell_policy`). |
| `python_exec` | Execute Python in a sandboxed subprocess. |
| `http` | Make an HTTP request (gated by `network_policy`). |
| `web_search` | Web search via DuckDuckGo / SerpAPI (needs the `search` extra). |
| `browse` | Drive a headless browser (needs the `browser` extra). |
| `data` | Read structured data files. |
| `ask_user` | Ask the user a question through the `io` handler. |
| `todo` | Maintain a working to-do list. |
| `spawn_agent` | Spawn a scoped sub-agent (see the [Agents guide](agents.md#sub-agents)). |
| `finish` | Signal the task is complete. |

Some tools require extras: `pip install "neurosurfer[search,browser]"`.

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

### The pieces

- **`input_model`** — a Pydantic model. Its JSON schema is what the model sees; validation errors are
  returned to the model as a correctable tool error, so it can retry.
- **`ToolResult`** — return `ToolResult.ok(text)` on success or `ToolResult.error(text)` on failure.
  Use `ToolResult.with_images(text, images)` to return screenshots/renders to vision models.
- **`ToolContext`** — passed to `call()`; carries the working directory, the `io` handler, and other
  run state a tool may need.
- **Behaviour flags** — `is_read_only`, `is_concurrency_safe`, `is_destructive` let the engine run
  safe tools in parallel and route destructive ones through the permission gates.
- **`progress_message(args)`** — override to show a friendly status line (e.g. "Reading README.md…")
  on the `ToolStarted` event.

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
