# Grounding & Refusal

Before designing a node, the Architect asks whether the thing that node would have to do is
something anything here can actually do. If not, it says so — and a clear refusal is a far better
answer than a workflow that validates and then hallucinates.

## The question that decides everything

**Does this step reach outside the model?**

Reading a file, fetching a URL, calling an API, touching an inbox, sending a message, running a
command — none of that can be done by an LLM step, however well it is worded. Such a node **must**
have a tool.

A node whose goal is *"read the file"* and whose tool list is empty is one LLM call being asked to
produce the contents of a file it cannot open. That graph validates clean and hallucinates at run
time, which is the failure mode this layer exists to remove.

## Grounding against declared tags

`find_capability` resolves a need against a **declared capability tag** — `file.write` →
`write_file` — rather than against words a tool's description happens to share.

This replaced prose matching, which picked wrong in ways that were hard to see: a
database-migration tool for "generate chart images", a hosted gateway for a database on
`localhost`. See the [Tool Registry](../guides/tool-registry.md).

When the resolver finds nothing, that is a **real answer**: no tool here has that capability. It is
a job to author or import, and it names itself.

## Filling a gap

| Route | Tool | Notes |
|---|---|---|
| Write it | `author_tool` | Generated, sandboxed, and registered only with approval. |
| Import it | `install_mcp_server` | Found via [MCP Discovery](../guides/mcp-discovery.md). |

Both are gated by callbacks, so nothing is installed or registered behind the user's back. See
[The Agent](agent.md#callbacks).

!!! warning "Authored tools are a reviewed starting point"
    Generated tool source is not code to ship unread.

## The two checks

The validator enforces grounding with two rules of very different confidence. See
[Validation](../graph/validation.md#capability-grounding) for the full statement.

- **A toolless `react` node** — structural and unambiguous. A `react` node is *defined* as an LLM
  that calls tools in a loop; with no tools it is a `base` node in a costume.
- **A suspected capability** — lexical, and therefore hedged by four rules learned from real false
  positives on our own workflows.

The trade behind those hedges is asymmetric and worth knowing when reading a warning: **a missed
warning leaves us where we started, while a wrong one derails a build.** On the run that motivated
the last of those rules, a model acknowledged a bogus warning correctly and then declared the whole
workflow blocked anyway.

## Refusal is an outcome, not a failure

```python
from neurosurfer.architect import ArchitectAgent, WorkflowInfeasible

try:
    path = await ArchitectAgent(provider).build(intent)
except WorkflowInfeasible as e:
    print("cannot build this:", e)      # a precise reason, not a shrug
```

`declare_blocked` is called when the request is impossible **as described** — it needs credentials
or resources that were not provided, or an unsafe capability.

Two distinct "not built" outcomes are worth separating when you handle them:

- **Infeasible** — nothing here can do it. Authoring or importing a tool is the path forward.
- **Blocked** — it could be done, but not with what was supplied. Supplying the missing thing is
  the path forward.

Both raise `WorkflowInfeasible`; the message distinguishes them.

## Why refusing early is cheaper

A capability that does not exist is discovered either now, for the cost of a lookup, or at run time
after every node before it has been designed, registered, and paid for. The plan-first pass exists
to move that discovery as early as it can go — before a node exists at all.

## Next

- [Verification](verification.md) — proving the design that grounding allowed.
- [Tool Registry](../guides/tool-registry.md) — the vocabulary being resolved against.
- [MCP Discovery](../guides/mcp-discovery.md) — importing what is missing.
