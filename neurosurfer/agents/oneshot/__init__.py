"""Agent — the simple one-shot agent (single bounded interaction).

The implementation lives in :mod:`agent`; this package re-exports the public name
so ``from neurosurfer.agents.oneshot import Agent`` (and the shorter
``from neurosurfer.agents import Agent``) both remain stable.
"""
from .agent import Agent  # noqa: F401

__all__ = ["Agent"]
