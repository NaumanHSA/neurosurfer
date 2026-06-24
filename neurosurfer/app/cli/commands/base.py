"""Slash-command registry.

A ``SlashCommand`` binds a name (typed as ``/name``) to an async handler. The
``CommandRegistry`` powers both dispatch and the prompt_toolkit completer (it
exposes name/summary pairs and prefix matching). Keeping commands in one registry
means adding a new one is a single ``register()`` call — the CLI grows without the
module saturating.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from ..context import CLIContext

CommandHandler = Callable[[CLIContext, list[str]], Awaitable[None]]


@dataclass
class SlashCommand:
    name: str
    summary: str
    handler: CommandHandler
    aliases: list[str] = field(default_factory=list)


class CommandRegistry:
    def __init__(self) -> None:
        self._by_name: dict[str, SlashCommand] = {}

    def register(self, cmd: SlashCommand) -> None:
        self._by_name[cmd.name] = cmd
        for alias in cmd.aliases:
            self._by_name[alias] = cmd

    def get(self, name: str) -> SlashCommand | None:
        return self._by_name.get(name.lstrip("/"))

    def all(self) -> list[SlashCommand]:
        # Unique commands (aliases point at the same object), in registration order.
        seen: list[SlashCommand] = []
        for cmd in self._by_name.values():
            if cmd not in seen:
                seen.append(cmd)
        return seen

    def matches(self, prefix: str) -> list[SlashCommand]:
        """Commands whose primary name starts with ``prefix`` (for completion)."""
        prefix = prefix.lstrip("/")
        return [c for c in self.all() if c.name.startswith(prefix)]
