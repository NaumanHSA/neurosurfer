# ToolsRouterAgent

**Module:** `neurosurfer.agents.tools_router`

## Overview

`ToolsRouterAgent` is a lightweight, production‑ready **tool router**. It uses an LLM to select **exactly one** tool from your [`Toolkit`](../tools/toolkit.md), validates the proposed inputs against the tool’s `ToolSpec`, and then executes the tool. It supports both **streaming** and **non‑streaming** outputs, optional **input pruning**, and **bounded retries** for both routing and tool execution.

Typical flow:

1. **Route:** Ask the LLM to emit a strict JSON decision `{ "tool": "...", "inputs": { ... } }`  
2. **Validate:** Clean/validate inputs against the tool’s schema (`ToolSpec`)  
3. **Execute:** Call the tool and proxy its output (stream or text)  
4. **Recover:** On invalid JSON / missing tool / tool error, retry with backoff (bounded by `RouterRetryPolicy`).

> This router is model‑agnostic. It works with OpenAI‑style and HF‑style clients that implement `ask(...)` and (for streaming) return OpenAI‑like `ChatCompletionChunk` deltas.

---

## Constructor

```python
ToolsRouterAgent(
    toolkit: Toolkit,
    llm: BaseChatModel,
    logger: logging.Logger = logging.getLogger(__name__),
    verbose: bool = False,
    specific_instructions: str = "",
    config: ToolsRouterConfig | None = None,
)
```

| Parameter | Type | Description |
| --- | --- | --- |
| `toolkit` | [`Toolkit`](../tools/toolkit.md) | Registry containing tool instances and their `ToolSpec`s. |
| `llm` | [`BaseChatModel`](../models/chat-models/base-model.md) | LLM used for routing and fallback answers. Must implement `ask(...)`. |
| `logger` | `logging.Logger` | Optional logger used when `verbose=True` and for error reporting. |
| `verbose` | `bool` | Prints routing decisions, validation messages, and errors when `True`. |
| `specific_instructions` | `str` | Extra system text appended to the router system prompt (domain policy). |
| `config` | [`ToolsRouterConfig`](#toolsrouterconfig) \| `None` | Advanced flags (streaming default, pruning, retries, LLM defaults). |

---

## Configuration

### `ToolsRouterConfig`

```python
@dataclass
class ToolsRouterConfig:
    allow_input_pruning: bool = True
    repair_with_llm: bool = True
    return_stream_by_default: bool = True
    retry: RouterRetryPolicy = RouterRetryPolicy()
    temperature: float = 0.7
    max_new_tokens: int = 4000
```

| Parameter | Type | Description |
| --- | --- | --- |
| `allow_input_pruning` | `bool` | When `True`, unknown keys in the routed `inputs` are dropped **before** validation. Set `False` for strict mode. |
| `repair_with_llm` | `bool` | When `True`, the agent will ask the LLM to re‑route or repair inputs after failures (bounded by retry policy). |
| `return_stream_by_default` | `bool` | If `True`, `run(...)` streams by default unless `stream=False` is passed. |
| `retry` | [`RouterRetryPolicy`](#routerretrypolicy) | Controls retry counts and backoff for routing/tool execution. |
| `temperature` | `float` | Default temperature for routing calls (overridable per `run(...)`). |
| `max_new_tokens` | `int` | Default token cap for routing calls (overridable per `run(...)`). |

### `RouterRetryPolicy`

```python
@dataclass
class RouterRetryPolicy:
    max_route_retries: int = 2
    max_tool_retries: int = 1
    backoff_sec: float = 0.7
```

| Parameter | Type | Description |
| --- | --- | --- |
| `max_route_retries` | `int` | How many times the router attempts to re‑route when JSON is invalid or no tool is chosen. |
| `max_tool_retries` | `int` | How many times to retry tool execution after a tool error (transient failures). |
| `backoff_sec` | `float` | Linear backoff delay between retries (multiplied by attempt index). |

---

## Methods

### `run(user_query: str, chat_history: list[dict] | None = None, *, stream: bool | None = None, temperature: float | None = None, max_new_tokens: int | None = None, **kwargs) -> str | Iterator[str]`

Routes, validates, and executes a tool for the given `user_query`. Returns a **generator of strings** when streaming, otherwise a **single string**.

```python
router = ToolsRouterAgent(toolkit=tk, llm=llm, verbose=True)
for chunk in router.run("Summarize ./README.md with bullets", stream=True):
    print(chunk, end="")
```

**Important kwargs:**

| Arg | Type | Description |
| --- | --- | --- |
| `stream` | `bool \| None` | Whether to stream output. Defaults to `config.return_stream_by_default`. |
| `temperature` | `float \| None` | Overrides the routing temperature for this call. |
| `max_new_tokens` | `int \| None` | Overrides the routing token cap for this call. |
| `**kwargs` | `Any` | Forwarded to the tool call (merged with routed `inputs`). |

---

## Routing behavior

- The router sends the user message and a **tool catalog** (from `Toolkit.get_tools_description()`) to the LLM with a strict system prompt.  
- The model must return **one‑line JSON** with exactly two keys: `"tool"` and `"inputs"`. Example:

```json
{"tool":"markdown_summarizer","inputs":{"path":"README.md"}}
```

- If JSON parsing fails or `"tool":"none"` is returned, the router retries (`max_route_retries`).  
- If a usable decision cannot be produced, a **helpful fallback message** is generated for the user.

### Input validation & pruning

- If the chosen tool has a `ToolSpec`, the router validates inputs.  
- With `allow_input_pruning=True`, unknown keys are **dropped** before validation. Set `False` for strict behavior (fail fast + attempt repair).

### Tool execution

- The router calls the tool and proxies its output.  
- If the tool returns a `ToolResponse` with a generator, the router **streams** those chunks.  
- If the tool returns a `ToolResponse` with a plain string / non‑stream response, it returns (or yields once) accordingly.  
- Tool errors are retried up to `max_tool_retries` with linear backoff.

---

## System prompt (router)

The router crafts a compact system instruction that lists available tools and enforces strict JSON. You can append **domain policy** via `specific_instructions` (e.g., “prefer read‑only operations” or “never use external network tools”).

---

## Error handling & fallbacks

- **Routing failures:** After retries, the agent produces a concise, user‑friendly message (without exposing internal errors).  
- **Tool failures:** After bounded retries, the agent returns a helpful message indicating that the request couldn’t be completed.  
- **Logging:** With `verbose=True`, routing outputs, parsed decisions, and exceptions are logged for debugging.

---

## Example

```python
from neurosurfer.agents.tools_router import ToolsRouterAgent, ToolsRouterConfig, RouterRetryPolicy
from neurosurfer.tools import Toolkit
from neurosurfer.models.chat_models.openai import OpenAIModel
from my_tools import MarkdownSummarizer, GrepTool

tk = Toolkit()
tk.register_tool(MarkdownSummarizer())
tk.register_tool(GrepTool())

llm = OpenAIModel(model_name="gpt-4o-mini")

router = ToolsRouterAgent(
    toolkit=tk,
    llm=llm,
    verbose=True,
    config=ToolsRouterConfig(
        allow_input_pruning=True,
        repair_with_llm=True,
        return_stream_by_default=True,
        retry=RouterRetryPolicy(max_route_retries=2, max_tool_retries=1, backoff_sec=0.7),
        temperature=0.5,
        max_new_tokens=2000,
    ),
)

# Streaming
for token in router.run("Summarize the key sections of README.md", stream=True):
    print(token, end="")

# Non-stream
text = router.run("Find 'RAG' mentions in ./docs", stream=False)
print("\n\nRESULT:", text)
```

---

## Best practices

- **Keep tool specs accurate**: the better your `ToolSpec`s, the more reliable routing becomes.  
- **Prefer minimal required inputs**: tools should require only what’s essential; optional params can be added later by the agent/tool itself.  
- **Use pruning in early development**: `allow_input_pruning=True` is forgiving while tool schemas evolve. For production, consider strict mode.  
- **Add a clarifier tool**: if your UX supports it, create an `ask_user` tool so the router can resolve missing inputs explicitly.  
- **Audit routing**: log router output for drift detection; periodically evaluate accuracy on real prompts.

---

## Related

- [`Toolkit`](../tools/toolkit.md) — tool registry and descriptions surfaced to the router.  
- [`ReActAgent`](react-agent.md) — full reasoning‑and‑acting loop (choose + observe + iterate).  