# The Agent

`ArchitectAgent` is a single planner with a toolbelt. It replaces the fixed eight-node pipeline: it
reads its [self-knowledge](knowledge.md), builds the graph incrementally, checks itself with the
validator, authors missing tools through a sandbox, and either registers a valid workflow or
declares the request blocked with a clear reason.

```python
from neurosurfer.architect import ArchitectAgent, WorkflowInfeasible

agent = ArchitectAgent(provider)
path = await agent.build("Summarise a CSV and write the result to a file")
```

## The terminal contract

Enforced after the loop ends, so a build cannot finish ambiguously:

| Outcome | Result |
|---|---|
| A registered path | `build()` returns it. |
| A blocked reason | `build()` raises `WorkflowInfeasible`. |
| Neither | `RuntimeError` carrying the agent's last text. |

There is no fourth case where a build "finishes" having produced nothing.

## `build()`

```python
async def build(
    intent: str,
    *,
    answers: dict[str, str] | None = None,
    refines: str | None = None,
    plan: Any = None,
) -> str
```

| Argument | Effect |
|---|---|
| `intent` | The plain-English request. |
| `answers` | Answers to clarifying questions asked on a previous attempt. |
| `refines` | Start from an existing **registered** workflow, reading the intent as a change to it rather than a fresh build. |
| `plan` | Skip the planning call and build this plan — how a plan reviewed in one request gets built by the next without being re-invented in between. |

## Construction

```python
ArchitectAgent(provider, *, verify="required", review="warn", plan=True, max_turns=40, ...)
```

### `verify`

| Value | Meaning |
|---|---|
| `"required"` | **Default.** The build must pass [verification](verification.md) before it registers. |
| `"encouraged"` | Prompted to verify, not gated on it. |
| `"off"` | No verification. |

The default is `required` because `encouraged` is prompt-only, and prompt-only is exactly what a
weaker model ignores: two bug-report transcripts went validate → ok → register with **no**
`test_workflow` call at all.

The earlier reason for not defaulting to `required` was that a 9B model stalled under it. The
grounding gates and prescriptive errors exist to fix the stalling, so it is now affordable.
Lowering it is an explicit caller decision.

### `review`

`"off"` · `"warn"` (default) · `"required"` — whether a design [review](how-it-works.md) pass runs
and whether it blocks.

### `plan`

Plan-first is **on by default**. It costs one call and grounds every external step before a node
exists, which is worth far more than it costs. It is switchable because a caller driving the agent
with a scripted provider is testing the agent, not the planner.

### Callbacks

Every interactive decision is a callback, so the same agent drives a CLI, a server, or a test:

| Callback | Asked for |
|---|---|
| `approve_tool` | Approval before an authored tool is registered. |
| `approve_mcp` | Approval before an MCP server is installed. |
| `approve_plan` | Approval of the plan before it is built. |
| `request_authorization` | An API key or account authorization. |
| `request_secrets` | Values a workflow will need supplied. |
| `ask_question` | A clarifying question. |
| `node_event` | Progress, per node. |
| `notify` | Free-text progress for a human. |

A build that parks on one of these and resumes when answered is what the
[Architect API](../server/architect-api.md) exposes over HTTP.

## The toolbelt

What the agent can actually do. Grouped by what each is for:

**Build the graph**

| Tool | Does |
|---|---|
| `set_workflow` | Name, description, declared inputs. Called first. |
| `add_node` / `update_node` / `remove_node` | One node at a time. |
| `set_outputs` | Declare the result node(s). |
| `view_workflow` | Read back the staged graph. |

**Ground it**

| Tool | Does |
|---|---|
| `find_capability` | Find a tool for a need, by [declared tag](../guides/tool-registry.md). |
| `describe_capability` | How a construct works — node kinds, fields, constraints. |
| `neurosurfer_docs` | Retrieve from **these docs**. See [Self-knowledge](knowledge.md). |
| `acknowledge_capability` | Record that a flagged gap was considered. |
| `author_tool` | Write a tool that does not exist, sandboxed and approved. |
| `install_mcp_server` / `list_mcp_tools` | Import a server that provides the gap. |

**Prove it**

| Tool | Does |
|---|---|
| `validate_workflow` | Structural check. Fix every issue, repeat until VALID. |
| `test_workflow` | **Actually runs it** on derived inputs and judges the output. |
| `register_workflow` | Register, once valid and tested. |

**Stop**

| Tool | Does |
|---|---|
| `declare_blocked` | The request is impossible as described — with a precise reason. |
| `drop_plan_step` | Explicitly abandon a planned step rather than silently skipping it. |

Every planned step must be built or explicitly dropped. Silence is not an outcome.

## When a build stalls

Small models narrate mid-build and stop without calling a tool. The agent issues up to six
continuation **nudges** after a premature text-only stop before giving up, which recovers the loop
rather than failing a build that was one turn from finishing.

## Fixing a failed verification

The operating procedure the agent follows is deliberately conservative: **fix the smallest thing
first.** Sharpen the failing node's instructions, correct its `depends_on` wiring, or swap a tool.

Only add a node or a control-flow construct if the intent truly needs a step that is missing — do
**not** escalate a working `router` into a `loop`, or bolt on extra nodes, to patch what is really a
prompt bug.

## Next

- [Grounding & Refusal](grounding.md) — how capability checks gate a build.
- [Verification](verification.md) — what `test_workflow` actually does.
- [Self-knowledge](knowledge.md) — where `find_capability` and `neurosurfer_docs` read from.
