"""ReactAgent — the text-parsing ReAct loop for providers without native tool use.

Split into :mod:`agent` (the loop), :mod:`parser` (tolerant output parsing), and
:mod:`prompt` (the ReAct system prompt). The parsing/prompt helpers are re-exported
for tests and advanced callers.
"""
from .agent import ReactAgent  # noqa: F401
from .parser import _parse_action_input, _parse_react_output  # noqa: F401
from .prompt import _build_react_system  # noqa: F401

__all__ = ["ReactAgent"]
