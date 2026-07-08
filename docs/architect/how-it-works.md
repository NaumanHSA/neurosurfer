# How It Works

The Architect is a **meta-workflow**: a fixed pipeline of LLM-driven nodes whose *output* is another
workflow. Understanding the stages helps you read its results — and know why a build sometimes comes
back as *infeasible* rather than wrong.

## The build pipeline

`ArchitectBuilder.run(intent)` executes a graph of nodes, roughly:

```
clarify ─▶ discover ─▶ plan ─▶ tool_design ─▶ write_nodes ─▶ assemble
```

| Stage | What it does |
|---|---|
| **clarify** | (Optional, interactive) Runs a short requirements interview to sharpen the intent. Skippable by supplying `answers=`. |
| **discover** | Reasons about the goal and the available tool catalog to decide what the workflow needs. |
| **plan** | Designs the graph — the nodes, their order, and dependencies — using only tools that actually exist (the real catalog is injected into the prompt so it can't invent tool names). |
| **tool_design** | Produces a **CapabilityPlan**: the feasibility verdict plus specs for any tools that are missing but required. |
| **write_nodes** | Writes each node's concrete logic. |
| **assemble** | Assembles the pieces into a Workflow package and **validates** it. Registration is withheld if validation fails. |

The whole build runs with a debug trace captured, so you can inspect exactly what each node was
prompted with and what it returned.

## Grounded in the real tool catalog

A recurring failure mode for "LLM builds a workflow" systems is **inventing tools that don't exist**.
The Architect avoids this by interpolating the real `available_tools` catalog into the plan and
`write_nodes` prompts — so generated graphs reference tools the runtime can actually resolve. Its own
build nodes run with a tight allow-list (e.g. `web_search`, `write_workflow_node`).

## Authoring missing tools

When the plan needs a capability no existing tool provides, the Architect doesn't just fail — it can
**author the tool**:

1. `tool_design` produces a rich spec for the missing tool.
2. The Architect drafts an implementation and runs it in a **sandbox**.
3. Your `approve_tool(draft, sandbox_result)` callback is invoked with the draft and the sandbox
   outcome. The tool is registered **only if you return `True`**.

This is the highest-variance part of the module — treat an authored tool as a **reviewable draft**,
not trusted code. If you don't supply `approve_tool`, a build that hits a capability gap surfaces the
gap instead of silently adding code.

## Feasibility: staged vs. infeasible

Two distinct "not built" outcomes:

- **Infeasible** — `tool_design` judges the workflow can't be built as described (e.g. it needs a
  capability that can't be provided). `run()` raises `WorkflowInfeasible` with the reason.
- **Staged but not registered** — the package was assembled but failed validation. The Architect
  re-validates and either surfaces a clean error (hard failures) or, for a missing-tool gap, authors
  the tool with your approval and then registers.

## Why quality varies

Because every stage is LLM-driven, the output tracks the **model** you pass to `ArchitectBuilder`.
A stronger model plans better graphs and writes better nodes; a weaker one produces plans that may
validate but underperform. This is the main reason the module is still marked
[experimental](index.md) — the scaffolding is solid, but the generated content is only as good as the
model behind it, and complex intents still stress it. Treat the output as a first draft to refine.

## Next

- [Building Workflows](building.md) — the API to drive all of this.
- [Graph & Workflows](../guides/graph-workflows.md) — the runtime that executes what the Architect
  produces.
