# ToolSpec

> Module: `neurosurfer.tools.tool_spec`  
> Pairs with: [`BaseTool`](./base-tool.md) • [`Toolkit`](./toolkit.md)

`ToolSpec` defines the **contract** for a tool: its **name**, **capabilities**, **inputs**, and **return type**. Agents rely on this metadata to decide **when** to use a tool, how to **validate inputs**, and how to interpret outputs. The spec is also used to **auto-generate documentation** and enforce **runtime validation**.

---

## Data Model

### Supported Types

The spec uses a constrained set of primitive types for validation:

- `string`, `integer`, `number`, `boolean`, `array`, `object`

> Internally, the validator also accepts `str` as an alias for `string`.

### `ToolParam` — input parameter

| Field | Type | Required | Description |
|---|---|:---:|---|
| `name` | `str` | ✓ | Parameter name used as the keyword in calls. |
| `type` | `str` | ✓ | One of the **Supported Types** above. |
| `description` | `str` | ✓ | Human‑readable description for agents/users. |
| `required` | `bool` |  | If `True`, must be present in inputs. Defaults to `True`. |

**Example**

```python
from neurosurfer.tools.tool_spec import ToolParam
ToolParam(name="query", type="string", description="SQL to execute", required=True)
```

### `ToolReturn` — return value

| Field | Type | Required | Description |
|---|---|:---:|---|
| `type` | `str` | ✓ | One of the **Supported Types** above. |
| `description` | `str` | ✓ | What the tool returns (shape/semantics). |

**Example**

```python
from neurosurfer.tools.tool_spec import ToolReturn
ToolReturn(type="object", description="Query results as a JSON object")
```

### `ToolSpec` — full specification

| Field | Type | Required | Description |
|---|---|:---:|---|
| `name` | `str` | ✓ | Unique tool identifier (e.g., `sql_query`). |
| `description` | `str` | ✓ | Brief description of what the tool does. |
| `when_to_use` | `str` | ✓ | Guidance to the agent about appropriate usage. |
| `inputs` | `list[ToolParam]` | ✓ | Parameter specs (unique names). |
| `returns` | `ToolReturn` | ✓ | Return value spec. |

**Example**

```python
from neurosurfer.tools.tool_spec import ToolSpec, ToolParam, ToolReturn

spec = ToolSpec(
    name="calculator",
    description="Performs arithmetic operations",
    when_to_use="Use when you must compute a numeric result",
    inputs=[
        ToolParam(name="operation", type="string", description="add|sub|mul|div", required=True),
        ToolParam(name="a", type="number", description="Left operand", required=True),
        ToolParam(name="b", type="number", description="Right operand", required=True),
    ],
    returns=ToolReturn(type="number", description="Result of the operation"),
)
spec.validate()
```

---

## Validation

### `validate() -> None`

Ensures a spec is well-formed. It checks:

- **Presence** of `name`, `description`, `when_to_use`
- **At least one** input param
- **Supported** types for each param and for the return value
- **Uniqueness** of parameter names

**Failure modes:** raises `ValueError` with a descriptive message (e.g., *“`calculator.b` has unsupported type 'float'`* or *“duplicate input 'query'”*).

### `check_inputs(raw: dict) -> dict`

Validates and **sanitizes runtime inputs** coming from an agent/LLM/user:

1. **Requireds**: all `required=True` params must be present.  
2. **No extras**: keys must match the declared param names exactly.  
3. **Type checks**: values must satisfy the declared param `type` using strict predicates:  
   - `integer`: `isinstance(v, int) and not isinstance(v, bool)`  
   - `number`: `isinstance(v, (int, float)) and not isinstance(v, bool)`  
   - `array`: `isinstance(v, list)`  
   - `object`: `isinstance(v, dict)`  
   - etc.

**Failure modes:** raises `ValueError` for missing requireds, unexpected keys, or type mismatches.  
**Success:** returns the validated `raw` dict.

### `to_json() -> dict`

Returns a JSON‑serializable dictionary representation of the spec (useful for introspection or emitting tool descriptions to a client).

---

## Usage with `BaseTool`

Every tool must declare a `spec: ToolSpec` on the class and will be validated in [`BaseTool`](./base-tool.md)`.__init__`. At runtime, agents can call:

```python
validated = tool.spec.check_inputs(raw_inputs)
result = tool(**validated, **runtime_ctx)
```

See the full tool lifecycle in [`BaseTool`](./base-tool.md).

---

## Examples

### Minimal echo tool

```python
from neurosurfer.tools.base_tool import BaseTool, ToolResponse
from neurosurfer.tools.tool_spec import ToolSpec, ToolParam, ToolReturn

class EchoTool(BaseTool):
    spec = ToolSpec(
        name="echo",
        description="Echoes text back to the caller.",
        when_to_use="Use for basic echo or smoke testing",
        inputs=[ToolParam(name="text", type="string", description="Text to echo", required=True)],
        returns=ToolReturn(type="string", description="The echoed text"),
    )
    def __call__(self, *, text: str, **kwargs) -> ToolResponse:
        return ToolResponse(final_answer=True, observation=text)
```

### Strict input validation

```python
# Raises ValueError: missing required 'text', or wrong types, or extras.
EchoTool().spec.check_inputs({"text": 123})
```

---

## Best Practices

- **Be explicit**: clear names and descriptions improve agent planning.
- **Keep inputs minimal**: fewer, well-typed parameters → better validation and UX.
- **Describe side effects** in the `description`/`when_to_use` if applicable (e.g., “writes to DB”).
- **Version breaking changes** in tool names if you must change inputs/returns radically.

---

## Reference

- Source: `neurosurfer/tools/tool_spec.py`  
- Related: [`BaseTool`](./base-tool.md) • [`Toolkit`](./toolkit.md)