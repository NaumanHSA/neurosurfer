"""Neurosurfer OpenAI-Compatible Gateway Server.

Drop-in replacement for the previous `neurosurfer.server` module.

Exposes an OpenAI-compatible API surface so clients like Open-WebUI can connect:
- GET  /v1/models
- POST /v1/chat/completions

Public API:
    from neurosurfer.server import NeurosurferServer
"""

from .gateway import NeurosurferServer

__all__ = ["NeurosurferServer"]
