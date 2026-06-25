"""AgenticLoop — the multi-step native-tool-use agent loop.

The loop itself lives in :mod:`loop`; this package re-exports the public name so
``from neurosurfer.agents.agentic_loop import AgenticLoop`` (and the shorter
``from neurosurfer.agents import AgenticLoop``) both remain stable.
"""
from .loop import AgenticLoop  # noqa: F401

__all__ = ["AgenticLoop"]
