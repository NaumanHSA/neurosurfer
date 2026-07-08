# Structured Output

Sometimes you don't want prose — you want a **typed object** you can trust. Neurosurfer gets
validated Pydantic models out of a model two ways: the one-shot `Agent` with an `output_schema`, or
the lower-level `structured_completion` helper.

## One-shot `Agent` with `output_schema`

Give `Agent` a Pydantic model and `complete()` returns a validated instance instead of a string:

```python
from pydantic import BaseModel
from pathlib import Path
from neurosurfer.agents import Agent, Guardrails
from neurosurfer.tools import default_pool

class Summary(BaseModel):
    title: str
    points: list[str]

agent = Agent(
    provider=provider, tools=default_pool(),
    system_prompt="Answer concisely.",
    guardrails=Guardrails(), io=io, cwd=Path.cwd(),
    output_schema=Summary,
)

result = await agent.complete("Summarise Neurosurfer in three bullet points.")
print(result.title, result.points)     # a validated Summary instance
```

The agent may still use tools first (bound `max_tool_rounds`), then produces the final structured
answer. Without `output_schema` the same `Agent` returns plain text.

## Low-level: `structured_completion`

When you don't need a full agent — no tools, no run loop, just "give me this object from this prompt"
— call `structured_completion` directly against a provider:

```python
from neurosurfer.agents.runtime.structured import structured_completion

result = await structured_completion(
    provider,
    Summary,                      # the target Pydantic type
    user="Summarise Neurosurfer.",
    system="You are concise.",    # optional
    max_attempts=3,               # repair invalid outputs up to N times
)
```

### How it works

The model is handed a single tool, `submit_result`, whose input schema **is** your Pydantic model,
and is instructed to call it exactly once. If the call doesn't validate, the error is fed back and
the model retries — up to `max_attempts`. On exhaustion it raises `StructuredCompletionError`.

!!! tip "Works on models without a native tool API too"
    Because the schema is enforced via a submit tool and a repair loop, structured output is reliable
    even on smaller local models — set a low temperature (the default is `0.2`) for best results.

## Choosing between them

| Use | When |
|---|---|
| `Agent(output_schema=…)` | You want tools *and* a typed final answer in one object. |
| `structured_completion(…)` | A pure prompt → object transform, no tools, minimal overhead. |
