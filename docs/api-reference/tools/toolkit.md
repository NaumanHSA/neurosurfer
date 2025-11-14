# Toolkit

> Module: `neurosurfer.tools.toolkit`  
> Works with: [`BaseTool`](./base-tool.md) • [`ToolSpec`](./tool-spec.md)

`Toolkit` is the **registry and manager** for tools in Neurosurfer. It validates and registers tools, keeps their specs, and produces **markdown descriptions** agents can consume to understand capabilities, inputs, and return types.

---

## Responsibilities

- **Register** tools (type-checked, spec-validated)
- **Prevent duplicates** by unique tool name
- **Expose** the registry for direct access
- **Render** human/LLM-friendly descriptions (`get_tools_description()`)

---

## Data & Attributes

| Attribute | Type | Description |
|---|---|---|
| `logger` | `logging.Logger` | Target for info/warnings about registrations. |
| `registry` | `dict[str, BaseTool]` | Tool name → tool instance. |
| `specs` | `dict[str, ToolSpec]` | Tool name → `ToolSpec`. |

---

## API

### `__init__( tools: List[BaseTool] = [], logger: logging.Logger | None = logging.getLogger(__name__))`

Creates a toolkit with the provided tools. `tools` is optional and must be a list of [`BaseTool`](./base-tool.md) instances. If `logger` is omitted, the module logger is used.

### `register_tool(tool: BaseTool) -> None`

Registers a tool. Enforces:

- `tool` **must** be an instance of [`BaseTool`](./base-tool.md)
- Tool name (`tool.spec.name`) **must be unique** within the registry

**On success:** the tool is available in `registry[name]` and `specs[name]`.  
**On failure:** raises
- `TypeError` if `tool` is not a `BaseTool`
- `ValueError` if a tool with the same name is already registered

**Example**

```python
from neurosurfer.tools.toolkit import Toolkit
from neurosurfer.tools.base_tool import BaseTool, ToolResponse
from neurosurfer.tools.tool_spec import ToolSpec, ToolParam, ToolReturn

class HelloTool(BaseTool):
    spec = ToolSpec(
        name="hello",
        description="Greets a person",
        when_to_use="When you need a greeting",
        inputs=[ToolParam(name="name", type="string", description="Person's name")],
        returns=ToolReturn(type="string", description="A greeting"),
    )
    def __call__(self, *, name: str, **kwargs) -> ToolResponse:
        return ToolResponse(final_answer=True, results=f"Hello, {name}!")

toolkit = Toolkit()
toolkit.register_tool(HelloTool())
assert "hello" in toolkit.registry
```

### `get_tools_description() -> str`

Returns a **markdown** string describing each registered tool in a consistent format:

```md
### `tool_name`
<description>
**When to use**: <when_to_use>
**Inputs**:
- `param`: <type> (required|optional) — <description>
**Returns**: <type> — <description>
```

This is useful for:
- **Agent prompts** (tooling instructions)
- **User-facing** docs or `/help` output
- **Debugging** tool registration

**Example**

```python
md = toolkit.get_tools_description()
print(md)
```

---

## Putting it together (Agent flow)

1. Build a `Toolkit` and **register** all tools.
2. Provide `toolkit.get_tools_description()` to the agent at planning time.
3. When the agent selects a tool:
   - Validate inputs using `tool.spec.check_inputs(raw)`
   - Call the tool: `tool(**validated, **runtime_ctx)` (see [`BaseTool`](./base-tool.md))
4. Consume the `ToolResponse` and decide whether to stop or continue based on `final_answer`.

---

## Best Practices

- **Unique names**: treat `spec.name` as a stable API surface (version if breaking changes occur).
- **Log registrations** at startup so you can audit available tools quickly.
- **Small, focused tools** compose better than monoliths.
- **Surface constraints** in the spec (`when_to_use`, param descriptions).

---

## Reference

- Source: `neurosurfer/tools/toolkit.py`  
- Related: [`BaseTool`](./base-tool.md) • [`ToolSpec`](./tool-spec.md)