"""Coding-assistant personas (the product's built-in sub-agents).

Importing this package registers every built-in persona into the engine
registry (``neurosurfer.agents.subagents.defs``) as a side effect of each module
calling ``register()`` at import time. The framework engine ships no personas;
they live here, in the product layer.
"""

from __future__ import annotations

# Re-export the engine registry primitives for convenience.
from neurosurfer.agents.subagents.defs import (
    SubAgentDefinition,
    all_agents,
    get_agent,
    register,
)

# Side-effect imports: each module calls register() at import time.
from . import analyzer, explore, verifier, writer

__all__ = [
    "SubAgentDefinition",
    "all_agents",
    "get_agent",
    "register",
    "analyzer",
    "explore",
    "verifier",
    "writer",
]
