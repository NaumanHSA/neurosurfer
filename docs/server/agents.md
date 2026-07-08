# Serving Agents

The gateway's most useful trick: take any Neurosurfer agent and expose it as an **OpenAI model**.
Point an existing OpenAI client (or the Neurosurfer CLI, or a chat UI) at your server and it talks to
your agent through `/v1/chat/completions`.

## Register an agent as a model

```python
from neurosurfer.app.server import NeurosurferServer
from neurosurfer.agents import AgenticLoop, Guardrails
from neurosurfer.tools import default_pool
from neurosurfer.llm import build_provider_from_profile

provider = build_provider_from_profile()      # from env / active profile
agent = AgenticLoop(
    provider=provider, tools=default_pool(),
    system_prompt="You are a helpful assistant.",
    guardrails=Guardrails(), io=AutoIO(),      # headless handler for server use
)

server = NeurosurferServer()
server.register_agent(agent, model_id="my-agent", description="My tool-using agent")
server.run()                                    # serves on 0.0.0.0:8000
```

Now the agent answers as the model `my-agent`:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "my-agent", "stream": true,
       "messages": [{"role": "user", "content": "List the files here."}]}'
```

`/v1/models` lists it; `/v1/chat/completions` runs it (SSE streaming when `stream: true`).

### `register_agent` options

| Argument | Default | Purpose |
|---|---|---|
| `agent` | — | An `AgenticLoop`, `ReactAgent`, `Agent`, or any object with `run(prompt)`. |
| `model_id` | — | The model name clients request. |
| `description` | `"Neurosurfer agent"` | Shown in `/v1/models`. |
| `owned_by` | `"neurosurfer"` | `/v1/models` owner field. |
| `max_model_len` | `8192` | Advertised context length. |
| `run_fn` | `None` | Override how the agent is invoked (advanced). |
| `result_to_text` | default | Map the agent's result to the response text (advanced). |

## Use an auto-approving handler

A server has no human at a terminal, so give served agents a headless `io` that never blocks on
approvals — `AutoApproveIOHandler` from `neurosurfer.tools`, plus tight
[`Guardrails`](../learn/permissions.md) to bound what tools can do. Never serve an agent with
interactive-only IO; requests would hang on the first gated action.

## Mixing agents and upstreams

One server can host several models at once — multiple agents under different `model_id`s, plus
proxied upstream models. See [Backends](backends.md) for upstream proxying and
[Hooks](hooks.md) for rewriting requests/responses. To run without writing Python, use the
[`neurosurfer serve`](../reference/cli.md) command — see [Deployment](deployment.md).
