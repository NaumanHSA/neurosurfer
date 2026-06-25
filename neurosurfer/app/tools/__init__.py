"""Coding-assistant product tools.

Importing this package makes the product's tools available: ``present_plan``
self-registers into the framework tool registry on import (so it appears in
``all_tools()``/``default_pool()`` once the product is loaded).
"""

from __future__ import annotations

from .present_plan import PresentPlanTool

__all__ = ["PresentPlanTool"]
