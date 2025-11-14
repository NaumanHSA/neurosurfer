# Custom Tools — Examples & Patterns

This page shows how to create **custom tools** for Neurosurfer, validate their inputs with `ToolSpec`, register them in a `Toolkit`, and use them from agents (e.g., `ReActAgent`). It complements the `Toolkit`, `ToolSpec`, and `BaseTool` modules by providing practical, copy‑pasteable examples, including streaming tools and memory‑passing via `extras`.

---

## 📦 Overview

- **`BaseTool`**: subclass this and implement `__call__(...) -> ToolResponse`.
- **`ToolSpec`**: defines **name**, **description**, **when_to_use**, **inputs** (`ToolParam` list), and **returns** (`ToolReturn`). Used for auto‑docs and **strict runtime validation** via `check_inputs(...)`.
- **`Toolkit`**: a registry for tools (`register_tool(...)`, `get_tools_description()`).
- **Agents** (e.g., `ReActAgent`): discover tools via `Toolkit`, validate tool calls, and pass runtime context (e.g., `llm`, `db_engine`, `vectorstore`).

**Validation Rules Recap**
- Required inputs must be present.
- No extra/unknown inputs allowed.
- Types must match (`string`, `integer`, `number`, `boolean`, `array`, `object`).

---

## Smallest possible tool

A simple **Echo** tool that returns a string. It illustrates `ToolSpec` and `ToolResponse`.

```python
from neurosurfer.tools.base_tool import BaseTool, ToolResponse
from neurosurfer.tools.tool_spec import ToolSpec, ToolParam, ToolReturn

class EchoTool(BaseTool):
    spec = ToolSpec(
        name="echo",
        description="Repeat a message back to the user.",
        when_to_use="When you need to echo/confirm user input.",
        inputs=[
            ToolParam(name="message", type="string", description="The text to echo", required=True),
        ],
        returns=ToolReturn(type="string", description="The echoed message"),
    )

    def __call__(self, *, message: str, **_) -> ToolResponse:
        return ToolResponse(final_answer=False, results=f"[echo] {message}")
```

Register in a toolkit:

```python
from neurosurfer.tools import Toolkit

tk = Toolkit()
tk.register_tool(EchoTool())
print(tk.get_tools_description())  # Markdown description for agents
```

---

## Tool with **extras** (memory passing)

Use `extras` to pass intermediate results to subsequent tool calls in a ReAct loop.

```python
from neurosurfer.tools.base_tool import BaseTool, ToolResponse
from neurosurfer.tools.tool_spec import ToolSpec, ToolParam, ToolReturn

class SumTool(BaseTool):
    spec = ToolSpec(
        name="sum_numbers",
        description="Sum a list of numbers.",
        when_to_use="When you need the sum of numeric values.",
        inputs=[
            ToolParam(name="values", type="array", description="List of numbers", required=True),
        ],
        returns=ToolReturn(type="number", description="The total sum"),
    )

    def __call__(self, *, values: list, **_) -> ToolResponse:
        total = sum(float(x) for x in values)
        return ToolResponse(
            final_answer=False,
            results=f"Sum = {total}",
            extras={"sum": total}  # becomes available to the next tool call
        )
```

In a `ReActAgent`, those `extras` will be merged into the next tool’s inputs (as the agent code demonstrates).

---

## **Streaming** tool (generator results)

Return a generator for incremental output. Agents/UIs can stream it in real time.

```python
from typing import Generator
from neurosurfer.tools.base_tool import BaseTool, ToolResponse
from neurosurfer.tools.tool_spec import ToolSpec, ToolParam, ToolReturn

class StreamLinesTool(BaseTool):
    spec = ToolSpec(
        name="stream_lines",
        description="Emit N lines, one by one.",
        when_to_use="When user wants incremental updates.",
        inputs=[
            ToolParam(name="n", type="integer", description="Number of lines to emit", required=True),
        ],
        returns=ToolReturn(type="string", description="A stream of lines (chunked)"),
    )

    def __call__(self, *, n: int, **_) -> ToolResponse:
        def _gen() -> Generator[str, None, None]:
            for i in range(1, int(n) + 1):
                yield f"line {i}\n"
        return ToolResponse(final_answer=False, results=_gen())
```

---

## Tool that uses **runtime context** (e.g., `llm`)

Agents inject runtime context keys (e.g., `llm`, `db_engine`, `vectorstore`). Your tool can accept them as kwargs.

```python
from neurosurfer.tools.base_tool import BaseTool, ToolResponse
from neurosurfer.tools.tool_spec import ToolSpec, ToolParam, ToolReturn

class SummarizeWithLLM(BaseTool):
    spec = ToolSpec(
        name="llm_summarize",
        description="Summarize a passage using the active LLM context.",
        when_to_use="When a short summary is needed.",
        inputs=[
            ToolParam(name="text", type="string", description="Text to summarize", required=True),
        ],
        returns=ToolReturn(type="string", description="A concise summary"),
    )

    def __call__(self, *, text: str, llm=None, **_) -> ToolResponse:
        if llm is None:
            return ToolResponse(final_answer=True, results="No LLM available.")
        resp = llm.ask(user_prompt=f"Summarize in 3 bullet points:\n\n{text}", temperature=0.2, max_new_tokens=200)
        out = resp.choices[0].message.content
        return ToolResponse(final_answer=True, results=out)
```

Register and use in an agent:

```python
from neurosurfer.tools import Toolkit
from neurosurfer.agents import ReActAgent
from neurosurfer.models.chat_models.openai_model import OpenAIModel

llm = OpenAIModel(model_name="gpt-4o-mini")
tk = Toolkit()
tk.register_tool(SummarizeWithLLM())

agent = ReActAgent(toolkit=tk, llm=llm, verbose=True)
for chunk in agent.run("Use llm_summarize on: 'Transformers are sequence models...'"):
    print(chunk, end="")
```

---

## Validation failure (how it looks)

`Toolkit` + agent will reject malformed inputs based on the tool’s spec. You can also call `spec.check_inputs(...)` manually.

```python
from neurosurfer.tools.tool_spec import ToolSpec, ToolParam, ToolReturn

spec = ToolSpec(
    name="calc",
    description="Add two numbers.",
    when_to_use="Basic arithmetic when both inputs are provided.",
    inputs=[
        ToolParam(name="a", type="number", description="First", required=True),
        ToolParam(name="b", type="number", description="Second", required=True),
    ],
    returns=ToolReturn(type="number", description="Sum"),
)

# Missing 'b' will raise ValueError
try:
    spec.check_inputs({"a": 10})
except ValueError as e:
    print("Validation error:", e)
```

---

## End‑to‑end with `ReActAgent`

```python
from neurosurfer.tools import Toolkit
from neurosurfer.agents import ReActAgent
from neurosurfer.models.chat_models.openai_model import OpenAIModel

# Register your custom tools
tk = Toolkit()
tk.register_tool(EchoTool())
tk.register_tool(SumTool())
tk.register_tool(StreamLinesTool())

# Model + Agent
llm = OpenAIModel(model_name="gpt-4o-mini")
agent = ReActAgent(toolkit=tk, llm=llm, verbose=True)

# The agent will think → act → observe → repeat, picking tools by spec
for piece in agent.run("Add [2, 3, 5], then echo the total. Stream 3 lines at the end."):
    print(piece, end="")
```

---

## Design tips

- **Be strict in specs**: keep inputs minimal and types exact. Agents become more reliable.
- **Keep tools single‑purpose**: complex tasks emerge from composing small tools.
- **Use `extras`** for intermediate state (IDs, partial results, structured objects).
- **Prefer streaming** where latency matters; return a generator for `results`.
- **Document “when to use”** clearly — it improves tool selection in both ReAct and router agents.
- **Log sparingly**: tools may run often; prefer concise, actionable logs.

---

## Troubleshooting

- **“Unexpected inputs”** → the LLM/tool passed extra fields; remove or add them to the spec.
- **Type mismatch** → ensure `integer` vs `number` and lists (`array`) are correct.
- **Agent didn’t pick your tool** → improve `description` and `when_to_use`; reduce overlap with other tools.
- **No runtime context** → ensure the agent passes `llm`/`db_engine`/`vectorstore` into `__call__` kwargs.

---

### Minimal checklist (before registering a tool)

- Inherit from `BaseTool`
- Provide a valid `ToolSpec` (`validate()` passes)
- Implement `__call__(**kwargs) -> ToolResponse`
- Return `final_answer=True` if the tool’s output should be surfaced directly
- Register with `Toolkit.register_tool(...)`

All set. You can now plug custom tools into **ReActAgent**, **SQLAgent**, and the **ToolsRouterAgent** with predictable behavior.