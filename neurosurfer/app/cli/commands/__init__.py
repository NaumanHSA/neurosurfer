"""Slash-command registry assembly."""

from __future__ import annotations

from .base import CommandRegistry, SlashCommand


def build_registry() -> CommandRegistry:
    from . import memory, misc, provider, session, theme, workflow

    registry = CommandRegistry()
    registry.register(provider.COMMAND)
    registry.register(workflow.COMMAND)
    registry.register(memory.COMMAND)
    registry.register(session.COMMAND)
    registry.register(theme.COMMAND)
    misc.register_misc(registry)
    return registry


__all__ = ["CommandRegistry", "SlashCommand", "build_registry"]
