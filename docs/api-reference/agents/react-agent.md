# ReActAgent

**Module:** `neurosurfer.agents.react`

## Overview

`ReActAgent` implements the ReAct (Reasoning + Acting) loop for complex, tool‑using tasks. It streams its reasoning, calls exactly **one tool per step**, observes results, and either continues iterating or emits a final answer. It is domain‑agnostic and can be used for anything: coding assistants, database agents, file managers, research helpers, etc.

Key capabilities:

1. **Robust Action parsing** — tolerant JSON extraction from LLM output (handles code fences, trailing commas, partial blocks).
2. **Schema‑aware input validation** — inputs are validated against each tool’s `ToolSpec`; optional **input pruning** safely drops unknown keys.
3. **Self‑repair** — when parsing or tool calls fail, the agent asks the LLM to repair the Action, with **bounded retries**.
4. **Streaming** — thoughts and final answers are streamed; tool outputs can stream too. Delimiter markers can be suppressed (see `skip_special_tokens`).
5. **Reusable core** — clean config (`ReActConfig`), retry policy (`RetryPolicy`), ephemeral memory, and toolkit wiring.

`ReActAgent` is designed to be subclassed for specialized agents (e.g., `SQLAgent`) while keeping shared behavior in the core.

---

## Constructor

```python
ReActAgent(
    toolkit: Toolkit,
    llm: BaseModel,
    *,
    logger: logging.Logger | None = None,
    specific_instructions: str = "",
    config: ReActConfig | None = None,
)
```

| Parameter | Type | Description |
| --- | --- | --- |
| `toolkit` | [`Toolkit`](../tools/index.md) | Registry of tools available to the agent. The agent will render descriptions from the toolkit into its system prompt. |
| `llm` | [`BaseModel`](../models/chat-models/base-model.md) | Any supported chat model (OpenAI‑style, Transformers/Unsloth, Llama.cpp, vLLM, etc.). Must implement `ask(...)` and `stop_generation()`. |
| `logger` | `logging.Logger \| None` | Optional logger; defaults to module logger. |
| `specific_instructions` | `str` | Extra system prompt addendum to steer behavior for a domain (e.g., SQL policy). |
| `config` | [`ReActConfig`](#reactconfig) \| `None` | Advanced configuration (temperature, retries, pruning, streaming markers, etc.). If `None`, defaults are used. |

---

## `ReActConfig`

```python
from dataclasses import dataclass, field
from neurosurfer.agents.react import RetryPolicy

@dataclass
class ReActConfig:
    temperature: float = 0.7
    max_new_tokens: int = 8000
    verbose: bool = True
    allow_input_pruning: bool = True      # drop extra inputs not in ToolSpec
    repair_with_llm: bool = True          # ask LLM to repair invalid Actions
    skip_special_tokens: bool = False     # when True, suppresses <__final_answer__> ... markers
    retry: RetryPolicy = field(default_factory=RetryPolicy)
```

### ReActConfig parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `temperature` | `float` | Default sampling temperature for LLM calls made by the agent loop. Overridable per `run(...)` call. |
| `max_new_tokens` | `int` | Default token cap for LLM generations inside the agent. Overridable per `run(...)` call. |
| `verbose` | `bool` | When `True`, prints additional debug info (e.g., results) via `rich`/logger. |
| `allow_input_pruning` | `bool` | If `True`, unknown keys in Action `inputs` are dropped before `ToolSpec` validation. If `False`, the agent attempts to **repair** the Action instead. |
| `repair_with_llm` | `bool` | If `True`, the agent prompts the LLM to output a corrected Action when parsing/validation fails or a tool errors. |
| `skip_special_tokens` | `bool` | If `True`, the agent **does not emit** `<__final_answer__>` / `</__final_answer__>` markers during streaming. Use this when your UI handles finalization itself. |
| `retry` | [`RetryPolicy`](#retrypolicy) | Controls retries for Action parsing and tool failures (counts and backoff). |

### `RetryPolicy`

```python
from dataclasses import dataclass

@dataclass
class RetryPolicy:
    max_parse_retries: int = 2   # attempts to repair missing/invalid Action
    max_tool_errors: int = 2     # attempts to repair & re-run a failing tool
    backoff_sec: float = 0.8     # linear backoff per retry
```

#### RetryPolicy parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `max_parse_retries` | `int` | Maximum number of times the agent will attempt to repair and regenerate an Action when none is found or JSON is invalid. |
| `max_tool_errors` | `int` | Maximum number of tool execution retries after using error feedback to repair the Action/inputs. |
| `backoff_sec` | `float` | Base number of seconds to wait between retries (linear backoff: multiplied by attempt index). |

---

## Response & Action Format

### Reasoning & Final Answer

The agent streams content. When the final answer begins, it typically emits markers:

```
Thought: ...
<__final_answer__>Final Answer: ...</__final_answer__>
```

- If `config.skip_special_tokens=True`, these markers are **suppressed** and only the final text streams.

### Tool Call (Action) Format

The LLM must end a step with a JSON Action (no prose after it):

```json
Action: {
  "tool": "tool_name",
  "inputs": { "param": "value" },
  "final_answer": false
}
```

- `tool` — name registered in `Toolkit`.
- `inputs` — must match the tool’s `ToolSpec` (unknown keys are dropped when `allow_input_pruning=True`).
- `final_answer` — see **ToolResponse.final_answer** below (this flag expresses the *intent*; the actual stop condition is ultimately governed by the tool’s returned `ToolResponse`).

If the Action is missing/invalid, or a tool call fails, the agent will attempt **self‑repair** using history + error messages, up to the configured retry limits.

---

## Main Methods

### `run(user_query: str, **kwargs) -> Generator[str, None, str]`

Runs the ReAct loop and **streams** text chunks (thoughts, tool IO, final answer). The generator’s return value is the final answer string.

```python
for chunk in agent.run("Summarize the latest design decisions.", temperature=0.3, max_new_tokens=2000):
    print(chunk, end="")
```

**Common kwargs (override config per call):**

| Kwarg | Type | Description |
| --- | --- | --- |
| `temperature` | `float` | Sampling temperature for this run (defaults to `config.temperature`). |
| `max_new_tokens` | `int` | Token cap for this run (defaults to `config.max_new_tokens`). |

### `stop_generation() -> None`

Signals both the underlying `llm` and the agent loop to stop as soon as possible (useful for UI Stop buttons).

```python
agent.stop_generation()
```

### `update_toolkit(toolkit: Toolkit) -> None`

Swap the toolkit at runtime; useful when dynamically adding tools.

```python
tk.register_tool(MyNewTool(...))
agent.update_toolkit(tk)
```

---

## Tooling Contract

Each tool should subclass `BaseTool` and define a `ToolSpec`:

```python
class MyTool(BaseTool):
    name = "my_tool"
    description = "One-line purpose for the LLM."
    spec = ToolSpec(
        name=name,
        description=description,
        inputs=[
            ToolParam(name="path", ptype=str, required=True, description="File path"),
            ToolParam(name="flag", ptype=bool, required=False, description="Optional flag"),
        ],
        returns=ToolReturn(rtype=str, description="Human-readable results")
    )

    def __call__(self, **kwargs) -> ToolResponse:
        params = self.spec.check_inputs(kwargs)
        # ... do work ...
        return ToolResponse(
            results="Done.",
            final_answer=False,
            extras={"some_key": "value"}  # becomes ephemeral memory for the next step
        )
```

### `ToolResponse.final_answer`

- Tools return a `ToolResponse` with `final_answer: bool`.  
- When `final_answer=True`, the agent **treats the tool’s results as the final user‑facing answer** and stops the loop immediately.  
- When streaming, the agent typically wraps the final text with `<__final_answer__> ... </__final_answer__>` markers; **if** `config.skip_special_tokens=True`, it **does not emit markers** and just streams the text.  
- Most tools should return `final_answer=False`. Only mark as final when the tool’s output is already the complete answer the user should see.

### Passing `extras` between tools (LLM‑invisible memory)

- `ToolResponse.extras` is a dictionary carried by the agent’s **ephemeral memory** into the **very next** tool call automatically.  
- This pass‑through **does not go through the LLM**; it is agent‑side only.  
- You can place **non‑serializable / rich Python objects** in `extras` (DB connections, compiled regexes, parsed ASTs, pandas objects, etc.) to avoid lossy stringification.  
- After a tool call completes, the agent injects `extras` into the input of the following tool (merged with that tool’s Action `inputs`).  
- Memory is **cleared after each tool call** to avoid accidental long‑term accumulation. Persist long‑term state in your own services or stores if needed.

---

## Error Handling & Self‑Repair

- **Missing/invalid Action** → The agent prompts the LLM to **repair** the Action (bounded by `max_parse_retries`).  
- **Tool validation errors** (unknown keys, missing requireds) → Either **drop extras** (`allow_input_pruning=True`) or ask the LLM to **repair** the inputs.  
- **Tool runtime errors** → The agent returns the error text to the LLM and retries with a **repaired Action** up to `max_tool_errors`, with backoff.

When retries are exhausted, the agent surfaces the failure as an results and may still produce a final answer if appropriate.

---

## Streaming Notes

- The agent streams model thoughts and final answers.  
- Tools may return **strings** or **generators** of strings; tool generators are proxied to the caller for live output.  
- Final‑answer markers are emitted unless `skip_special_tokens=True` (in which case only the raw text streams).

---
