"""Sub-agent definition primitive + registry (framework engine).

A ``SubAgentDefinition`` declares an agent role that can be spawned by a parent
agent. This module is part of the **engine** — it provides the generic type and
a registry mechanism but ships **no concrete personas**. Products (e.g. the
coding assistant under ``neurosurfer.app``) register their personas here at
import time, keeping the framework usable without any product.

A definition carries:
- A typed ``agent_type`` key used when spawning via ``spawn_agent`` tool.
- The system prompt (static string or zero-arg callable for lazy generation).
- A tool allow/deny list so the spawner can filter the parent's full ToolPool.
  ``["*"]`` means all tools are permitted before applying ``disallowed_tools``.
- An optional model preference tag ("haiku" → cheapest/fastest; "inherit" →
  same model as parent; ``None`` → engine default).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

# ──────────────────────────────────────────────────────────────────────────────
# Definition
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SubAgentDefinition:
    agent_type: str
    when_to_use: str
    system_prompt: str | Callable[[], str]
    # Tool allow-list: ["*"] = inherit all; list of names = only those tools.
    allowed_tools: list[str] = field(default_factory=lambda: ["*"])
    # Tools to exclude even if they would be permitted by allowed_tools.
    disallowed_tools: list[str] = field(default_factory=list)
    # "haiku" = fast/cheap tier; "inherit" = same model as parent; None = default.
    model_preference: str | None = None

    def get_system_prompt(self) -> str:
        if callable(self.system_prompt):
            return self.system_prompt()
        return self.system_prompt

    def resolve_tools(self, pool_names: list[str]) -> list[str]:
        """Return the tool names this agent may use given the parent pool."""
        if self.allowed_tools == ["*"]:
            allowed = set(pool_names)
        else:
            allowed = set(self.allowed_tools) & set(pool_names)
        disallowed = set(self.disallowed_tools)
        return [n for n in pool_names if n in allowed and n not in disallowed]


# ──────────────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────────────

_REGISTRY: dict[str, SubAgentDefinition] = {}


def register(defn: SubAgentDefinition) -> None:
    _REGISTRY[defn.agent_type] = defn


def get_agent(agent_type: str) -> SubAgentDefinition | None:
    return _REGISTRY.get(agent_type)


def all_agents() -> list[SubAgentDefinition]:
    return list(_REGISTRY.values())
