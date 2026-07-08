# Agents

Neurosurfer ships three agent types, all built on the same engine and re-exported from
`neurosurfer.agents`:

| Agent | Use it when |
|---|---|
| **`AgenticLoop`** | You want multi-step tool use via the provider's **native** function-calling API. |
| **`ReactAgent`** | Your model has **no native tool API** — it drives tools by parsing text (ReAct). |
| **`Agent`** | You want a **single bounded call**, optionally with tools or structured output. |

## Constructing an agent

Every agent takes the same core keyword arguments:

```python
from pathlib import Path
from neurosurfer.agents import AgenticLoop, Guardrails
from neurosurfer.tools import default_pool

agent = AgenticLoop(
    provider=provider,                       # any Provider (see the Providers guide)
    tools=default_pool(),                     # the tools the agent may call
    system_prompt="You are a helpful assistant. Use tools, then finish.",
    guardrails=Guardrails(),                  # enforced limits (see below)
    io=AutoIO(),                              # how approvals/questions are answered
    cwd=Path.cwd(),                           # working directory for file tools
)
```

### The `io` handler

`io` is an **`IOHandler`** — the object an agent calls when a tool needs a decision (answer a
question, approve a shell command, approve a file write). Interactive apps supply a UI-backed handler
(the CLI uses a Rich terminal one); scripts and notebooks use an auto-approving handler. The full
approval model — the handler methods, `AutoApproveIOHandler`, and the safety caveats — lives in
[Permissions & Safety](../learn/permissions.md#approvals-the-io-handler). Examples below use a headless
`AutoIO()` for brevity.

## AgenticLoop

`AgenticLoop.run(prompt)` is an async generator that streams [events](#events) as the agent thinks,
calls tools, and answers:

```python
import asyncio

async def main():
    agent = AgenticLoop(
        provider=provider, tools=default_pool(),
        system_prompt="Complete the task, then call the finish tool.",
        guardrails=Guardrails(), io=AutoIO(), cwd=Path.cwd(),
    )
    async for event in agent.run("List the .md files here and summarise the README."):
        if hasattr(event, "text"):
            print(event.text, end="", flush=True)

asyncio.run(main())
```

## ReactAgent

Same constructor and streaming interface as `AgenticLoop`, but tool calls are parsed from the
model's text output — use it with local models that lack a native tool-calling API:

```python
from neurosurfer.agents import ReactAgent

agent = ReactAgent(
    provider=local_provider, tools=default_pool(),
    system_prompt="Solve the task using the available tools.",
    guardrails=Guardrails(), io=AutoIO(), cwd=Path.cwd(),
)
```

## One-shot Agent

`Agent` runs a single bounded turn. In **text mode** it returns the final string; with an
`output_schema` it returns a validated Pydantic model. Call `complete()` to run to completion and get
the result directly (or iterate `run()` for events):

```python
from pydantic import BaseModel
from neurosurfer.agents import Agent

class Summary(BaseModel):
    title: str
    points: list[str]

agent = Agent(
    provider=provider, tools=default_pool(),
    system_prompt="Answer concisely.",
    guardrails=Guardrails(), io=AutoIO(), cwd=Path.cwd(),
    output_schema=Summary,
)
result = await agent.complete("Summarise Neurosurfer in three bullet points.")
print(result.title, result.points)   # validated Summary instance
```

Pass `max_tool_rounds=N` to allow a bounded number of tool calls before the final answer.

## Events

Both `run()` generators yield typed events from `neurosurfer.agents` (also under
`neurosurfer.agents.events`):

| Event | Meaning |
|---|---|
| `TextDelta` | A chunk of the answer (`.text`). |
| `ThinkingDelta` | A chunk of reasoning, when the model exposes it (`.text`). |
| `ToolStarted` / `ToolFinished` | A tool call began / returned. |
| `TurnCompleted` | One model turn finished (carries usage + stop reason). |
| `ModeChanged` | The permission mode changed mid-run. |
| `Compacted` | The context was summarised to stay within the window. |
| `RunFinished` | The run ended (status + final text). |
| `AgentError` | An error surfaced during the run. |

```python
from neurosurfer.agents import TextDelta, ToolStarted, RunFinished

async for ev in agent.run(prompt):
    if isinstance(ev, ToolStarted):
        print(f"\n[tool] {ev}")
    elif isinstance(ev, TextDelta):
        print(ev.text, end="", flush=True)
    elif isinstance(ev, RunFinished):
        print(f"\n[done] {ev}")
```

## Permissions and guardrails

Every agent takes a `guardrails=` (what it's allowed to do) and a `mode=` (the runtime posture, e.g.
`"plan"` / `"default"` / `"accept_edits"` / `"bypass"`):

```python
agent = AgenticLoop(
    provider=provider, tools=default_pool(),
    system_prompt="…", io=AutoIO(), cwd=Path.cwd(),
    guardrails=Guardrails(write_scope=["out/**"], shell_policy="denied"),
    mode="default",
)
```

The full field reference, the permission modes, and the approval flow are covered in
[Permissions & Safety](../learn/permissions.md).

## Sub-agents

Agents can spawn scoped sub-agents (via the `spawn_agent` tool) to parallelise work, bounded by the
guardrails above. See the [Sub-agents guide](subagents.md).

## Context management

Long runs stay within the context window via automatic compaction (`ContextManager`) while
`DurableState` pins the plan/todos/decisions that must survive it. Pass `context_manager=` and/or
`durable=`; see [Context & Memory](context.md).
