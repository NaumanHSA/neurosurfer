# Sub-agents

A sub-agent is a **scoped agent an agent can spawn** to do focused work — explore a codebase,
analyse a file, draft a section — optionally in parallel. Sub-agents keep the parent's context clean
(their chatter stays in their own run) and are bounded by the parent's guardrails.

## Spawning from within a run

The parent calls the built-in **`spawn_agent`** tool. Nesting depth and parallelism are capped by
[`Guardrails`](../learn/permissions.md#guardrails):

| Guardrail | Default | Effect |
|---|---|---|
| `max_subagent_depth` | `2` | How deeply sub-agents may nest. |
| `max_concurrent_subagents` | `4` | How many may run in parallel. |

When tracing is on, each sub-agent nests under its parent's trace automatically (shared `trace_id`,
child span) — see [Observability](../observability/index.md).

## Defining a persona

A sub-agent **persona** is a `SubAgentDefinition` (`neurosurfer.agents.subagents`) — a named role with
its own system prompt and a tool allow-list:

```python
from neurosurfer.agents.subagents.defs import SubAgentDefinition, register

RESEARCHER = SubAgentDefinition(
    agent_type="researcher",
    when_to_use="Gather and summarise information from the web before writing.",
    system_prompt="You research thoroughly and cite sources. Finish with a summary.",
    allowed_tools=["search", "http", "read_file"],   # ["*"] = inherit all
    disallowed_tools=["run_command"],                  # subtract even if allowed
    model_preference="haiku",                          # "haiku" | "inherit" | None
)
register(RESEARCHER)
```

| Field | Purpose |
|---|---|
| `agent_type` | The name the parent spawns by. |
| `when_to_use` | Guidance the parent model sees when choosing a sub-agent. |
| `system_prompt` | The persona's prompt (a string, or a callable returning one). |
| `allowed_tools` | `["*"]` inherits the parent pool; a list restricts to those names. |
| `disallowed_tools` | Names to exclude even if `allowed_tools` would permit them. |
| `model_preference` | `"haiku"` (fast/cheap), `"inherit"` (parent's model), or `None` (default). |

Registered personas are looked up via `get_agent(agent_type)` and listed with `all_agents()`.

## Built-in personas

The Neurosurfer coding assistant (under `neurosurfer.app.agents`) ships four ready-made personas that
self-register on import — a useful reference for writing your own:

- **`explore`** — read-only codebase search and discovery.
- **`analyzer`** — focused analysis of specific files or questions.
- **`verifier`** — check that a change does what it should.
- **`writer`** — draft or edit prose/content.

## When to reach for sub-agents

- **Fan-out** — analyse many files or sources in parallel, then aggregate.
- **Context hygiene** — keep a noisy subtask (large file reads, search dumps) out of the main thread.
- **Specialisation** — give a subtask a tighter tool set and a cheaper model.

For the `spawn_agent` tool signature and the rest of the tool pool, see the
[Tools guide](tools.md).
