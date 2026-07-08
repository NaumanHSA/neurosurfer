# Context & Memory

Long runs eventually bump against the model's context window. Neurosurfer handles this automatically:
**`ContextManager`** summarises old history before it overflows, and **`DurableState`** pins the
things that must survive that summary — the plan, todos, and key decisions.

## Automatic compaction

Pass a `context_manager=` when constructing an agent and it watches the running token count. As the
conversation approaches the window limit, it **compacts**: the older message prefix is replaced with
a single summary message, while a tail of recent turns is kept verbatim to avoid an abrupt break. A
`Compacted` event is emitted so UIs and traces can show it happened.

```python
from neurosurfer.agents import AgenticLoop
from neurosurfer.agents.context import ContextManager

agent = AgenticLoop(
    provider=provider, tools=tools,
    system_prompt="…", guardrails=guardrails, io=io, cwd=cwd,
    context_manager=ContextManager(caps=provider_capabilities),
)
```

The threshold is derived from the provider's capabilities
(`auto_compact_threshold(context_window, max_output_tokens)`), so it adapts to the model you're on.
The defaults are sensible for most runs — you rarely need to tune this.

!!! note "Compaction is lossy by design"
    Summarising trades detail for headroom. That's why anything you can't afford to lose goes into
    `DurableState`, not the compactable history.

## Durable state

`DurableState` holds a small block of **pinned context** that is injected on every turn and is
**never** summarised away:

```python
from neurosurfer.agents.context import DurableState

durable = DurableState()
durable.set_plan("Refactor auth", "1. extract service\n2. add tests\n3. migrate callers")
durable.set_todos([{"text": "extract service", "done": False}])
durable.add_decision("Use JWT, not sessions — see ADR-004.")

agent = AgenticLoop(..., durable=durable)
```

| Method | Pins |
|---|---|
| `set_plan(title, text)` | The current plan. |
| `set_todos(items)` | A todo list (list of dicts). |
| `set_manifest(text)` | A free-form manifest / working notes. |
| `add_decision(text)` | An append-only log of decisions. |

`to_context_block()` renders the durable block that gets injected; `is_empty()` tells you whether
there's anything to inject.

## Why two mechanisms

- **`ContextManager`** keeps the run *within the window* — it's about surviving long conversations.
- **`DurableState`** keeps the run *on track* — it's about not forgetting the goal after a compaction.

Together they let an agent work for many turns without either overflowing the context or losing the
thread. Both are optional; omit them for short, bounded runs.
