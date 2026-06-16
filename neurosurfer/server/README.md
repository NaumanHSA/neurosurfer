# Neurosurfer Server (OpenAI-Compatible Gateway)

This module exposes an OpenAI-compatible API so you can use Open-WebUI (or any OpenAI client)
while still injecting Neurosurfer custom logic (RAG, tool routing, graph agents, etc.).

## Supported endpoints

- `GET  /v1/models`
- `POST /v1/chat/completions` (streaming + non-streaming)

## Key ideas

- **Backends**: upstream proxy (vLLM/OpenAI) and agent backends.
- **Hooks**: request/response middleware at the *OpenAI* boundary.
- **Concurrency**: fully async I/O; blocking agent execution runs in a worker thread.

## Minimal usage

```python
from neurosurfer.server import NeurosurferServer
from neurosurfer.server.backends import UpstreamBackend
from neurosurfer.agents.graph import GraphAgent

server = NeurosurferServer(host="0.0.0.0", port=8000)

server.register_backend(
    UpstreamBackend(
        name="vllm",
        base_url="http://localhost:8001/v1",
        api_key="abc",
        models_mode="proxy",
    ),
    default=True,
)

agent = GraphAgent(...)
server.register_agent(agent, model_id="ns/graph/rag_code_docs_workflow")

server.run()
```
