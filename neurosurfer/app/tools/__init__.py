"""Coding-assistant product tools.

Importing this package makes the product's tools available: ``present_plan``
self-registers into the framework tool registry on import (so it appears in
``all_tools()``/``default_pool()`` once the product is loaded). ``register_task``
is assembled explicitly into the agent pool by ``tasks.runner`` (it needs a
tasks directory), so it is not auto-registered.
"""

from __future__ import annotations

from .present_plan import PresentPlanTool
from .register_task import RegisterTaskArgs, RegisterTaskTool

__all__ = ["PresentPlanTool", "RegisterTaskTool", "RegisterTaskArgs"]
