# Backends

A **backend** is what a model ID resolves to. The gateway's `ModelRouter` maps each incoming
`model` name to a backend and forwards the request. There are two kinds.

## Upstream backends

`UpstreamBackend` proxies an existing OpenAI-compatible server. Register one (or several) and each
becomes a model ID:

```python
from neurosurfer.app.server import NeurosurferServer, UpstreamBackend

server = NeurosurferServer()
server.register_backend(
    UpstreamBackend(name="local", base_url="http://localhost:8001/v1"),
    default=True,      # this backend handles unknown model IDs
)
server.register_backend(
    UpstreamBackend(name="openai", base_url="https://api.openai.com/v1", api_key="sk-..."),
    default=False,
)
```

Requests, streaming, and errors are passed through transparently, so existing OpenAI clients work
unchanged.

## Agent backends

Register a native agent to expose it as a model ID. Any `AgenticLoop`, `ReactAgent`, or `Agent` (or
any object with a `run(prompt)` method) works:

```python
from neurosurfer.agents import AgenticLoop
from neurosurfer.app.server import NeurosurferServer

server = NeurosurferServer()
server.register_agent(
    AgenticLoop(provider=provider),
    model_id="research-agent",
    description="Web-research agent",
)
```

`register_agent()` accepts:

| Argument | Purpose |
|---|---|
| `agent` (positional) | The agent instance. |
| `model_id` (keyword) | The model name clients request. |
| `description`, `owned_by` | Metadata shown on `/v1/models`. |
| `max_model_len` | Advertised context length. |
| `run_fn` | Override how the agent is invoked per request. |
| `result_to_text` | Override how the agent's result becomes response text. |

Under the hood this wraps the agent in an `AgentBackend` (with an `AgentSpec`) and registers it with
the router — you can also build those directly for full control.

## Routing

`ModelRouter` and `RouteTarget` (both exported from `neurosurfer.app.server`) are the low-level
routing primitives. Registering backends and agents is the common path; reach for the router
directly only when you need custom dispatch (e.g. routing one model ID to different backends by
request content — see [Hooks](hooks.md)).
