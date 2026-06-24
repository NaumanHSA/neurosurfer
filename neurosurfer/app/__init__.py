"""The coding-assistant product, built on the neurosurfer framework.

The framework (agents engine, llm, tools, rag, graph, …) ships only generic
primitives. This ``app`` package is the coding-assistant product layered on top:
its personas, app-flavored tools, and (later) CLI.

Importing ``neurosurfer.app`` registers the product's built-in personas into
the engine registry, so a product entry point only needs ``import neurosurfer.app``.
"""

from __future__ import annotations

# Registers the coding-assistant personas (explore/analyzer/writer/verifier)
# and product tools (present_plan self-registers into the tool registry).
from . import agents  # noqa: F401
from . import tools  # noqa: F401

__all__ = ["agents", "tools"]
