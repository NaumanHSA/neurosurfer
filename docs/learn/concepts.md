# Core Concepts

A handful of types recur everywhere in Neurosurfer. Learn these once and every layer reads the same.

## Canonical messages

Providers differ on the wire, but Neurosurfer normalises everything to a **canonical**,
provider-neutral model in `neurosurfer.llm.types`:

- **`Message`** — a `role` (`"user"` or `"assistant"`) plus a list of **content blocks**.
- **Content blocks** — `TextBlock`, `ThinkingBlock`, `ToolUseBlock`, `ToolResultBlock`, `ImageBlock`.
  A single assistant turn can mix thinking, text, and tool calls.
- **`CanonicalResponse`** — one model turn's output: its `content` blocks, `usage`, and
  `stop_reason`. Helpers: `.text()`, `.thinking()`, `.tool_uses()`, `.as_message()`.

```python
from neurosurfer.llm.types import Message, TextBlock

msg = Message(role="user", content=[TextBlock(text="Hello")])
msg.text()          # "Hello"
```

Because history is canonical, the same conversation works across Anthropic, OpenAI, and any
OpenAI-compatible local model — you switch providers without rewriting messages.

## The run lifecycle & events { #events }

Every agent's `run(prompt)` is an **async generator of typed events** (from `neurosurfer.agents`).
Consuming them is how you stream output, drive a UI, or observe a run:

| Event | Meaning |
|---|---|
| `TextDelta` | A chunk of the user-facing answer (`.text`). |
| `ThinkingDelta` | A chunk of reasoning, when the model exposes it (`.text`). |
| `ToolStarted` / `ToolFinished` | A tool call began / returned. |
| `TurnCompleted` | One model turn finished (carries `usage`, `stop_reason`, and the turn's input/output). |
| `ModeChanged` | The permission mode changed mid-run. |
| `Compacted` | History was summarised to stay within the context window. |
| `RunFinished` | The run ended (`status` + final text). |
| `AgentError` | An error surfaced during the run. |

```python
from neurosurfer.agents import TextDelta, ToolStarted, RunFinished

async for ev in agent.run("Summarise the README."):
    if isinstance(ev, TextDelta):
        print(ev.text, end="", flush=True)
    elif isinstance(ev, ToolStarted):
        print(f"\n[tool] {ev.name}")
    elif isinstance(ev, RunFinished):
        print(f"\n[{ev.status}]")
```

### Streaming vs. collecting

- **Stream** — iterate `run(prompt)` for live tokens and tool activity.
- **Collect** — `await agent.run_collect(prompt)` drives the run to completion and returns a
  `RunResult` (`final_text`, `status`, `usage`, `turns`). Use it in scripts that just want the answer.

## Sessions

A **session** groups related runs. Pass `session_id=` when constructing an agent and every run shares
it — in Langfuse those runs appear under one session instead of as N disconnected traces. The CLI
mints one session per conversation and resets it on `/clear`. See
[Observability](../observability/index.md).

## Where to go next

- [Agents](../guides/agents.md) — construct and drive the three agent types.
- [Permissions & Safety](permissions.md) — how tool actions are gated.
- [Architecture](architecture.md) — how these pieces stack.
