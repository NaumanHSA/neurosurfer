# Base Tool

> Module: `neurosurfer.tools.base_tool`  
> Works with: [`ToolSpec`](./tool-spec.md), [`Toolkit`](./toolkit.md) (docs to follow)

`BaseTool` defines the **contract** every tool must follow in Neurosurfer. Tools encapsulate side-effectful or domain-specific actions (e.g., SQL execution, RAG queries, report generation) behind a stable interface so agents can **discover**, **validate**, and **invoke** them safely. Each tool declares a [`ToolSpec`](./tool-spec.md) used for input validation, automatic documentation, and agent reasoning.

---

## What this module provides

- **`ToolResponse`** — a structured return type for tool executions (supports **streaming**).
- **`BaseTool`** — an abstract base class that all tools must inherit from; enforces a `spec` and a `__call__` implementation.

---

## `ToolResponse`

A lightweight dataclass describing a tool’s result. It supports **final answers**, **intermediate results**, and **streaming** (via Python generators).

### Fields

| Field | Type | Required | Description |
|---|---|:---:|---|
| `final_answer` | `bool` | ✓ | If `True`, the agent should treat `results` as the **final** answer and stop tool use. |
| `results` | `str \| Generator[Any, None, None]` | ✓ | The tool’s output. Use a **string** for single-shot responses or a **generator** to **stream tokens/lines/chunks**. |
| `extras` | `dict` |  | Arbitrary metadata to persist in agent memory and pass to subsequent tool calls (e.g., IDs, cursors, diagnostics). |

### Examples

**Single-shot response**

```python
from neurosurfer.tools.base_tool import ToolResponse

ToolResponse(
    final_answer=False,
    results="Found 3 matching rows in `users` table",
    extras={"row_ids": [1, 2, 3]}
)
```

**Streaming response**

```python
from typing import Generator
from neurosurfer.tools.base_tool import ToolResponse

def _stream_lines(lines) -> Generator[str, None, None]:
    for ln in lines:
        yield ln  # agent will stream these to the user

ToolResponse(
    final_answer=True,
    results=_stream_lines(["Step 1...", "Step 2...", "Done."]),
)
```

> Agents should detect generator outputs and stream them to the user/UI. If your tool streams, **set** `final_answer=True` only when it truly concludes the task.

---

## `BaseTool`

`BaseTool` enforces common behavior across all tools:

- A **declared** `spec` of type [`ToolSpec`](./tool-spec.md) (validated at init).
- A concrete `__call__(...) -> ToolResponse` implementation.
- A consistent invocation model (LLM/tooling context is passed via `**kwargs`).

### Required attribute

- `spec: ToolSpec` — must be defined on the subclass **class body**. It describes:
  - `name`: unique tool identifier (e.g., `"sql_query"`)
  - `description`: what the tool does
  - `when_to_use`: guidance for agents
  - `inputs`: list of `ToolParam` (name, type, description, required)
  - `returns`: a `ToolReturn` (type + description)

See full spec schema in [`ToolSpec`](./tool-spec.md).

### Lifecycle

```python
def __init__(self) -> None:
    # Ensures subclass has a valid spec.
    if not hasattr(self, "spec") or not isinstance(self.spec, ToolSpec):
        raise TypeError("YourTool must define a ToolSpec 'spec'.")
    self.spec.validate()
```

If validation fails (e.g., unknown param type, duplicate param names), a `ValueError` is raised from `ToolSpec.validate()`.

### Required method

```python
@abstractmethod
def __call__(self, *args: Any, **kwargs: Any) -> ToolResponse:
    ...
```

- **Inputs**: The agent **validates** and **packages** inputs per `spec.inputs` and calls your tool as `tool(**validated_inputs, **runtime_ctx)`.  
- **Runtime context** (in `**kwargs`) may include objects like `llm`, `db_engine`, `embedder`, `vector_store`, etc., injected by the calling agent. Your tool should **not** assume their presence—check and fail gracefully.
- **Return**: Always a `ToolResponse` (single-shot or streaming).

---

## Implementing a tool (practical examples)

### Minimal single-shot tool

```python
from neurosurfer.tools.base_tool import BaseTool, ToolResponse
from neurosurfer.tools.tool_spec import ToolSpec, ToolParam, ToolReturn

class EchoTool(BaseTool):
    spec = ToolSpec(
        name="echo",
        description="Echoes the input string back to the caller.",
        when_to_use="When you need to return the same text verbatim.",
        inputs=[
            ToolParam(name="text", type="string", description="Text to echo", required=True),
        ],
        returns=ToolReturn(type="string", description="The echoed text"),
    )

    def __call__(self, *, text: str, **kwargs) -> ToolResponse:
        return ToolResponse(final_answer=True, results=text)
```

### Tool that uses injected runtime context

```python
class SQLQueryTool(BaseTool):
    spec = ToolSpec(
        name="sql_query",
        description="Execute a SQL query and return rows as JSON.",
        when_to_use="When you need structured data from the database.",
        inputs=[
            ToolParam(name="query", type="string", description="The SQL query to run", required=True),
            ToolParam(name="limit", type="integer", description="Max rows to return", required=False),
        ],
        returns=ToolReturn(type="object", description="Query results as a JSON object"),
    )

    def __call__(self, *, query: str, limit: int = 100, **kwargs) -> ToolResponse:
        db = kwargs.get("db_engine")
        if db is None:
            return ToolResponse(final_answer=True, results="DB engine not available.")
        rows = db.execute(query).fetchmany(size=limit)
        return ToolResponse(final_answer=False, results={"rows": [dict(r) for r in rows]})
```

### Streaming tool

```python
from typing import Generator, Iterable

class StreamLinesTool(BaseTool):
    spec = ToolSpec(
        name="stream_lines",
        description="Stream lines back to the caller (demo).",
        when_to_use="When producing incremental output is better for UX.",
        inputs=[ToolParam(name="lines", type="array", description="List of lines to stream", required=True)],
        returns=ToolReturn(type="string", description="A stream of lines"),
    )

    def __call__(self, *, lines: Iterable[str], **kwargs) -> ToolResponse:
        def _gen() -> Generator[str, None, None]:
            for ln in lines:
                yield ln
        return ToolResponse(final_answer=True, results=_gen())
```

---

## Agent/tool invocation model

1. Agent chooses a tool based on its `ToolSpec` (`when_to_use`, `inputs`, `returns`).
2. Agent validates candidate inputs via `ToolSpec.check_inputs(raw)`.
3. Agent calls `tool(**validated_inputs, **runtime_ctx)`.
4. Tool returns `ToolResponse`:
   - `final_answer=True` → agent halts tool-use and presents the results.
   - `final_answer=False` → agent may call more tools using `extras` if present.
5. If `results` is a **generator**, agent streams results to the user/UI.

---

## Best practices

- **Keep specs precise**: Names/types/descriptions drive agent decision-making and validation.
- **Use `extras` sparingly**: Include only what’s useful for follow-on calls (IDs, cursors, references).
- **Fail soft**: If a dependency (e.g., DB) isn’t provided, return a helpful message rather than raising.
- **Be deterministic**: Idempotent tools make planning easier; document any side effects in `description`/`when_to_use`.
- **Stream when valuable**: Large or long-running results benefit from generator-based `results`.
- **Security**: Validate/escape user inputs before executing commands/queries; document constraints in the spec.

---

## Reference

- Source: `neurosurfer/tools/base_tool.py`  
- Related: [`ToolSpec`](./tool-spec.md) • [`Toolkit`](./toolkit.md)