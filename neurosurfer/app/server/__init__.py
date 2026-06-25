"""OpenAI-compatible gateway server.

Exposes a FastAPI application with ``/v1/chat/completions``, ``/v1/models``,
and ``/health`` routes.  Backends can be upstream LLM servers (vLLM, LM Studio,
Ollama, OpenAI) or native :mod:`neurosurfer.agents` instances.

Minimal quickstart::

    from neurosurfer.app.server import NeurosurferServer, UpstreamBackend

    server = NeurosurferServer(port=8000)
    server.register_backend(
        UpstreamBackend(name="local", base_url="http://localhost:8001/v1")
    )
    server.run()

Register a native agent::

    from neurosurfer.agents import AgenticLoop
    from neurosurfer.app.server import NeurosurferServer

    loop = AgenticLoop(provider=..., tools=[...])
    server = NeurosurferServer()
    server.register_agent(loop, model_id="my-agent")
    server.run()

Requires the ``[serve]`` extra: ``pip install 'neurosurfer[serve]'``.
"""

from .backends.agent import AgentBackend, AgentSpec
from .backends.upstream import UpstreamBackend
from .config import ServerSettings
from .errors import OpenAIHTTPError
from .gateway import NeurosurferServer
from .hooks.base import Hook, HookContext
from .hooks.builtin import StripReasoningHook, SystemPromptInjectorHook
from .registry import ModelRouter, RouteTarget

__all__ = [
    "AgentBackend",
    "AgentSpec",
    "Hook",
    "HookContext",
    "ModelRouter",
    "NeurosurferServer",
    "OpenAIHTTPError",
    "RouteTarget",
    "ServerSettings",
    "StripReasoningHook",
    "SystemPromptInjectorHook",
    "UpstreamBackend",
]
