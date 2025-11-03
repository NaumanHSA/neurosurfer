# Tools

Neurosurfer’s tooling system lets agents perform real actions—query data, retrieve knowledge, call internal services—through a **consistent, validated contract**. You can register **custom tools** or use **built-in tools** that ship with Neurosurfer. Each tool declares a spec (inputs, returns, when-to-use) and returns a structured `ToolResponse` so agents can plan, validate, and compose multi-step workflows safely.

<div class="grid cards" markdown>

-   :material-cog:{ .lg .middle } **BaseTool**

    ---

    The abstract contract every tool must implement. Enforces a class-level `spec` and a `__call__(...) -> ToolResponse` implementation.

    [:octicons-arrow-right-24: Documentation](./base-tool.md)

-   :material-wrench:{ .lg .middle } **Toolkit**

    ---

    Validates, registers, and describes tools for agents. Prevents duplicates and renders Markdown descriptions via `get_tools_description()`.

    [:octicons-arrow-right-24: Documentation](./toolkit.md)

-   :material-file-document-outline:{ .lg .middle } **ToolSpec**

    ---

    The schema for a tool (`name`, `description`, `when_to_use`, `inputs`, `returns`) with strict runtime validation via `check_inputs(...)`.

    [:octicons-arrow-right-24: Documentation](./tool-spec.md)

-   :material-puzzle-outline:{ .lg .middle } **Custom Tools**

    ---

    Your domain-specific actions (internal APIs, dashboards, analytics). Subclass `BaseTool`, declare a `ToolSpec`, return a `ToolResponse`.

    [:octicons-arrow-right-24: Quick Start](#quick-start-custom-tool)

-   :material-toolbox:{ .lg .middle } **Built-in Tools**

    ---

    Ready-to-use tools maintained by Neurosurfer (retrieval helpers, data utilities, and more). Same contract as your own tools.

    [:octicons-arrow-right-24: Catalog](./builtin-tools/index.md)

</div>

---

## Core Building Blocks

### BaseTool (execution contract)
- Subclass and define a `spec: ToolSpec` **on the class**.
- Implement `__call__(...) -> ToolResponse`.
- Use `ToolResponse.extras` to pass state between tools (e.g., IDs, cursors, intermediate artifacts).

### ToolSpec (validation & documentation)
- Declare **inputs** with precise types (`string`, `number`, `integer`, `boolean`, `array`, `object`).
- Provide **when-to-use** guidance so agents can select the right tool.
- Enforce correctness at runtime with `spec.check_inputs(raw_dict)`.

### Toolkit (registry & discovery)
- `register_tool(tool)` with type checks and duplicate prevention.
- `get_tools_description()` for agent prompts / `/help` output.
- Provide `registry[name]` lookup to execute tools directly.

---

## Quick Start (Custom Tool)

```python
from neurosurfer.tools.base_tool import BaseTool, ToolResponse
from neurosurfer.tools.tool_spec import ToolSpec, ToolParam, ToolReturn
from neurosurfer.tools.toolkit import Toolkit

# 1) Define a custom tool by extending BaseTool
class Calculator(BaseTool):
    spec = ToolSpec(
        name="calculator",
        description="Performs basic arithmetic on two numbers.",
        when_to_use="When you need a quick numeric computation.",
        inputs=[
            ToolParam(name="op", type="string", description="add|sub|mul|div"),
            ToolParam(name="a", type="number", description="Left operand"),
            ToolParam(name="b", type="number", description="Right operand"),
        ],
        returns=ToolReturn(type="number", description="Computation result"),
    )
    def __call__(self, *, op: str, a: float, b: float, **_) -> ToolResponse:
        ops = {"add": a+b, "sub": a-b, "mul": a*b, "div": a/b if b else float('inf')}
        return ToolResponse(final_answer=True, observation=str(ops.get(op, 'NaN')))

# 2) Register it in a toolkit
toolkit = Toolkit()
toolkit.register_tool(Calculator())

# 3) Let agents discover & invoke
print(toolkit.get_tools_description())                         # Markdown summary for prompts/docs
validated = toolkit.registry["calculator"].spec.check_inputs(
    {"op": "mul", "a": 6, "b": 7}
)
result = toolkit.registry["calculator"](**validated)           # -> ToolResponse
print("Answer:", result.observation)                           # "42"
```

---

## How Tools Fit Into Agents

1. **Discovery** — The agent reads `Toolkit.get_tools_description()` and selects candidate tools.  
2. **Validation** — The agent validates inputs with `ToolSpec.check_inputs(...)`.  
3. **Invocation** — The agent calls `tool(**validated_inputs, **runtime_ctx)` and gets a `ToolResponse`.  
4. **Control Flow** — If `final_answer=True`, the agent stops. Otherwise, it may chain more tools using `extras` for context.  
5. **Observability** — Tools should log responsibly and return meaningful `observation` text (or a streaming generator) for a great UX.

---

## Design Guidelines

- **Be explicit** — precise names and descriptions improve agent planning.
- **Validate strictly** — reject extra unknown inputs; type-check everything.
- **Small, composable tools** — easier to plan, test, and swap.
- **Stream when it helps** — long results or progressive tasks benefit from generator output in `ToolResponse.observation`.
- **Document side effects** — specify in `description`/`when_to_use` if a tool writes to disk, calls external services, or mutates state.
- **Version carefully** — if you change inputs/returns, consider versioning the tool name (`…_v2`).

---

## Next Steps

- Contracts: [BaseTool](./base-tool.md) • [ToolSpec](./tool-spec.md) • [Toolkit](./toolkit.md)  
- Catalog: [Built-in Tools](./builtin-tools/index.md)  
- Extend: add your own tools and register them in `Toolkit` to grow agent capabilities.