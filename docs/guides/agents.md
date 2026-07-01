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

`io` is an [`IOHandler`](../guides/tools.md) — the object an agent calls when a tool needs a
decision (answer a question, approve a shell command, approve a file write). Interactive apps supply
a UI-backed handler (the CLI uses a Rich-based one). For **scripts and notebooks**, supply a small
auto-approving handler:

```python
class AutoIO:
    """Auto-approving IOHandler for scripts and notebooks."""
    async def ask(self, question: str, options=None) -> str:
        return (options or ["yes"])[0]
    async def request_plan_approval(self, plan: str) -> tuple[bool, str]:
        return True, ""
    async def request_shell_approval(self, command: str, reason: str) -> bool:
        return True
    async def request_write_approval(self, path: str, summary: str) -> str:
        return "once"
    def notify(self, message: str) -> None:
        pass
```

!!! warning "Auto-approval runs tools without prompting"
    `AutoIO` approves every action. Only use it in trusted, sandboxed contexts, and lean on
    [`Guardrails`](#permissions-and-guardrails) (`write_scope`, `shell_policy`, `path_deny`) to
    contain what tools can touch.

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

`Guardrails` (a Pydantic model) enforces what an agent may do:

| Field | Default | Purpose |
|---|---|---|
| `write_scope` | `["**"]` | Glob(s) the agent may write to. |
| `shell_policy` | `"gated"` | `gated` (ask), `readonly`, or `denied`. |
| `network_policy` | `"gated"` | `gated`, `open`, or `denied`. |
| `mcp_policy` | `"gated"` | Gating for MCP tool calls. |
| `path_allow` / `path_deny` | `["**"]` / `.env`, `secrets`, `.git`, … | Readable-path allow/deny globs. |
| `max_turns` | `200` | Hard cap on model turns. |
| `max_subagent_depth` | `2` | How deep sub-agents may nest. |
| `max_concurrent_subagents` | `4` | Parallel sub-agent cap. |

The agent's `mode` (a `PermissionMode`) sets the overall posture:

- `"plan"` — the agent must present a plan for approval before acting.
- `"default"` — gated actions prompt through `io`.
- `"accept_edits"` — file edits are auto-approved.
- `"bypass"` — skip gating (trusted/automated contexts only).

```python
agent = AgenticLoop(
    provider=provider, tools=default_pool(),
    system_prompt="…", io=AutoIO(), cwd=Path.cwd(),
    guardrails=Guardrails(write_scope=["out/**"], shell_policy="denied"),
    mode="default",
)
```

## Sub-agents

Agents can spawn scoped sub-agents (via the `spawn_agent` tool and `SubAgentRunner`) to parallelise
work; nesting depth and concurrency are bounded by the guardrails above. See the
[Tools guide](tools.md) for `spawn_agent`.

## Context management

`ContextManager` keeps a run within the model's context window by **auto-compacting** older history
into a summary as it approaches the limit (emitting a `Compacted` event), and `DurableState` pins
plan/todos/decisions outside the compactable history so they survive summarisation. Pass a
`context_manager=` and/or `durable=` to customise; the defaults are sensible for most runs.
