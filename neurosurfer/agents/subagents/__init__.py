"""Sub-agents: definition primitive + registry, and the spawning runner."""
from .defs import (  # noqa: F401
    SubAgentDefinition,
    all_agents,
    get_agent,
    register,
)
from .runner import MAX_DEPTH, SubAgentRunner  # noqa: F401
