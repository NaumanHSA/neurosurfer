# Permissions & Safety

Agents can read files, run shell commands, hit the network, and call MCP tools. Neurosurfer gates
those actions with two independent knobs: **`Guardrails`** (the fixed limits) and the **permission
mode** (the runtime posture), with a human (or a handler) in the loop for anything risky.

## Guardrails

`Guardrails` (a Pydantic model, `neurosurfer.agents.Guardrails`) enforces what an agent may do,
independent of what the model tries:

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

```python
from neurosurfer.agents import Guardrails

guardrails = Guardrails(
    write_scope=["out/**"],     # only write under out/
    shell_policy="denied",       # never run shell
    network_policy="open",       # allow HTTP/search without prompting
)
```

Guardrails are the **backstop**: even in an auto-approving context, `write_scope`/`path_deny`/
`shell_policy` still contain what a tool can touch.

## Permission modes

The agent's `mode` (a `PermissionMode`) sets the overall posture and can change mid-run (emitting a
`ModeChanged` event):

- **`"plan"`** — the agent must present a plan and get approval before acting.
- **`"default"`** — gated actions prompt through the `io` handler.
- **`"accept_edits"`** — file edits are auto-approved; other gated actions still prompt.
- **`"bypass"`** — skip gating entirely (trusted/automated contexts only).

```python
agent = AgenticLoop(
    provider=provider, tools=default_pool(),
    system_prompt="…", io=io, cwd=Path.cwd(),
    guardrails=guardrails, mode="plan",
)
```

## Approvals: the `io` handler

When a gated action needs a decision, the agent calls its **`IOHandler`** (`io=`). That's the single
seam between "the agent wants to do X" and "someone said yes":

- `request_shell_approval(command, reason)` — approve a shell command.
- `request_write_approval(path, summary)` — approve a file write (`"once"` / `"always"` / `"no"`).
- `request_plan_approval(plan)` — approve a plan in `"plan"` mode.
- `ask(question, options)` — the `ask_user` tool.
- `notify(message)` — surface status.

Interactive apps supply a UI-backed handler (the CLI uses a Rich terminal one, `TerminalIOHandler`).
For **scripts and notebooks**, use an auto-approving handler — `AutoApproveIOHandler` from
`neurosurfer.tools`, or a tiny custom class:

```python
class AutoIO:
    """Auto-approving IOHandler for trusted, sandboxed contexts."""
    async def ask(self, question, options=None): return (options or ["yes"])[0]
    async def request_plan_approval(self, plan): return True, ""
    async def request_shell_approval(self, command, reason): return True
    async def request_write_approval(self, path, summary): return "once"
    def notify(self, message): pass
```

!!! warning "Auto-approval runs tools without prompting"
    An auto-approving handler says yes to everything. Only use it in trusted, sandboxed contexts,
    and lean on `Guardrails` (`write_scope`, `shell_policy`, `path_deny`) to bound the blast radius.

## How it fits together

`mode` decides *whether* to ask; `Guardrails` decides *what's even allowed*; the `io` handler decides
*the answer when asked*. A denied guardrail can't be overridden by approving in `io` — the limit
wins. This is also how sub-agents stay bounded: depth and concurrency come straight from the parent's
guardrails (see [Sub-agents](../guides/subagents.md)).
